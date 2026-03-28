"""
KudaHitamQuant KV cache v2: Asymmetric attention with Triton Optimization (Pure Multi-Pass).
"""
import os
import sys

# Monkeypatch for transformers bnb integration bug 
try:
    import transformers.integrations
    if not hasattr(transformers.integrations, "validate_bnb_backend_availability"):
        def dummy_val(*args, **kwargs): pass
        transformers.integrations.validate_bnb_backend_availability = dummy_val
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Stability flags for CUDA 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Triton Kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# --- GILA MODE: RAW CUDA JIT BRIDGE (LAZY) ---
_KudaHitamCUDA = None
CUDA_EXT_AVAILABLE = False
_CUDA_BUILD_ATTEMPTED = False
_CENTROID_CACHE = {}

def load_cuda_ext():
    global _KudaHitamCUDA, CUDA_EXT_AVAILABLE, _CUDA_BUILD_ATTEMPTED
    if _CUDA_BUILD_ATTEMPTED: return _KudaHitamCUDA
    _CUDA_BUILD_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load
        import os, subprocess
        def find_nvcc():
            try:
                subprocess.check_output(["nvcc", "--version"])
                return True
            except:
                for p in ["/usr/local/cuda/bin", "/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.1/bin", "/usr/local/cuda-11.8/bin"]:
                    if os.path.exists(os.path.join(p, "nvcc")):
                        os.environ["PATH"] += ":" + p
                        return True
            return False

        current_dir = os.getcwd()
        try: 
            current_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"[KudaHitam] Script directory detected: {current_dir}")
        except NameError: 
            print(f"[KudaHitam] __file__ not found, using CWD: {current_dir}")
        
        _src = os.path.join(current_dir, "KudaHitamCUDA.cu")
        print(f"[KudaHitam] Searching for KudaHitamCUDA.cu at: {_src}")
        
        if not os.path.exists(_src):
            print(f"[KudaHitam] [!] ERROR: KudaHitamCUDA.cu NOT FOUND. Fallback mode will be active.")
            return None
            
        nvcc_path = find_nvcc()
        if not nvcc_path:
            print(f"[KudaHitam] [!] ERROR: NVCC (CUDA Compiler) not found in PATH. Check your CUDA installation.")
            return None
        else:
            print(f"[KudaHitam] NVCC detected.")

        build_dir = os.path.join(current_dir, "cuda_build")
        os.makedirs(build_dir, exist_ok=True)
        
        try:
            print(f"[KudaHitam] Starting JIT Compilation (Mode: Gila Mode V7.6 FP16 Native Engine)...")
            _KudaHitamCUDA = load(name="KudaHitamCUDA_V76", sources=[_src], verbose=False, with_cuda=True, build_directory=build_dir)
            CUDA_EXT_AVAILABLE = True
            print("[KudaHitam] [✓] ULTRA-GILA MODE ACTIVE: Monolithic Native FP16 (V7.6) fully loaded.")
        except Exception as e:
            print(f"[KudaHitam] [X] JIT Compilation failed! Error detail:\n{str(e)}")
            print("[KudaHitam] Falling back to Triton/PyTorch engine.")
            return None
            
        return _KudaHitamCUDA
    except Exception as e: 
        print(f"[KudaHitam] Unexpected error in load_cuda_ext: {e}")
        return None

# --- PURE TRITON KERNELS ---

if TRITON_AVAILABLE:
    @triton.jit
    def fwht_blocked_kernel(x_ptr, y_ptr, n_rows, stride_x, D: tl.constexpr, BLOCK_ROWS: tl.constexpr):
        """Amortized Ultra-Fast Zero-Transpose Bitwise FWHT."""
        pid = tl.program_id(0)
        row_base = pid * BLOCK_ROWS
        offs = tl.arange(0, D)
        
        # Precompute masks to avoid constexpr[0] indexing bug
        m0 = tl.reshape(tl.where(tl.arange(0, 2) == 0, 1.0, 0.0), [1, 2, 1])
        m1 = tl.reshape(tl.where(tl.arange(0, 2) == 1, 1.0, 0.0), [1, 2, 1])
        
        for k in range(BLOCK_ROWS):
            row_idx = row_base + k
            if row_idx < n_rows:
                ptr = x_ptr + row_idx * stride_x + offs
                x = tl.load(ptr).to(tl.float32)
                
                # Zero-Transpose Butterfly (Unrolled Sylvester) - Masked Sum Approach
                t = tl.reshape(x, [128, 2, 1]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [64, 2, 2]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [32, 2, 4]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [16, 2, 8]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [8, 2, 16]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [4, 2, 32]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [2, 2, 64]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                t = tl.reshape(x, [1, 2, 128]); a = tl.sum(t * m0, 1); b = tl.sum(t * m1, 1); x = tl.reshape(tl.join(a + b, a - b), [256])
                
                scale = 1.0 / 16.0 # sqrt(256)
                tl.store(y_ptr + row_idx * stride_x + offs, (x * scale).to(x_ptr.dtype.element_ty))

    @triton.jit
    def quantize_kernel_pure(
        x_ptr, centroids_ptr, out_ptr,
        n_elements, n_centroids,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused quantization with register-level unrolling."""
        pid = tl.program_id(0)
        offsets = (pid << 10) + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        min_dist = tl.full((BLOCK_SIZE,), 1e18, dtype=tl.float32)
        min_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for i in range(n_centroids):
            centroid = tl.load(centroids_ptr + i)
            dist = tl.abs(vals - centroid)
            curr_mask = dist < min_dist
            min_dist = tl.where(curr_mask, dist, min_dist)
            min_idx = tl.where(curr_mask, i, min_idx)
        tl.store(out_ptr + offsets, min_idx, mask=mask)

# --- WRAPPER FUNCTIONS ---

def fwht(x: torch.Tensor):
    if not x.is_cuda: return fwht_pytorch(x)
        
    cuda_ext = load_cuda_ext()
    if CUDA_EXT_AVAILABLE and cuda_ext:
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).contiguous()
        cuda_ext.forward(x)
        return x.reshape(orig_shape)
        
    # Priority 2: Bitwise-Optimized Triton (Blocked)
    if not TRITON_AVAILABLE: return fwht_pytorch(x)
        
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous(); D = x.shape[-1]
    if (D & (D - 1)) != 0: return fwht_pytorch(x)
        
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        BLOCK_ROWS = 64
        grid = (triton.cdiv(x.shape[0], BLOCK_ROWS),)
        fwht_blocked_kernel[grid](x, y, x.shape[0], x.stride(0), D=D, BLOCK_ROWS=BLOCK_ROWS)
    return y.reshape(orig_shape)

def fwht_pytorch(x: torch.Tensor):
    orig_shape = x.shape; D = orig_shape[-1]
    if (D & (D - 1)) != 0:
        next_pow2 = 1 << (D - 1).bit_length(); x = torch.nn.functional.pad(x, (0, next_pow2 - D)); D = next_pow2
    x = x.reshape(-1, D); i = 1
    while i < D:
        x = x.view(-1, D // (i << 1), 2, i); a, b = x[:, :, 0, :], x[:, :, 1, :]
        new_a, new_b = a + b, a - b; x[:, :, 0, :], x[:, :, 1, :] = new_a, new_b; i <<= 1
    return x[..., :orig_shape[-1]].reshape(orig_shape)

# --- COMPRESSORS ---

class KudahitamCompressorV2:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim; self.bits = bits; self.device = device
        
        # Consistent Rademacher Vector (D) regardless of cache
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.d = torch.sign(torch.randn(head_dim, generator=gen)).to(device)

        # Optimized Centroid Logic: Use Global Cache 
        cache_key = (bits, head_dim)
        if cache_key in _CENTROID_CACHE:
            self.centroids = _CENTROID_CACHE[cache_key].to(device)
        else:
            from scipy import integrate; n_levels = 1 << bits; sigma = 1.0 / math.sqrt(head_dim)
            def pdf(x): return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))
            lo, hi = -3.5 * sigma, 3.5 * sigma
            centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
            for _ in range(200):
                boundaries = [(centroids[i] + centroids[i+1]) / 2.0 for i in range(n_levels - 1)]; edges = [lo * 3] + boundaries + [hi * 3]; new_c = []
                for i in range(n_levels): a, b = edges[i], edges[i+1]; num, _ = integrate.quad(lambda x: x*pdf(x), a, b); den, _ = integrate.quad(pdf, a, b); new_c.append(num / den if den > 1e-15 else centroids[i])
                if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10: break
                centroids = new_c
            self.centroids = torch.tensor(centroids, dtype=torch.float32).to(device)
            _CENTROID_CACHE[cache_key] = self.centroids.cpu()


    @torch.no_grad()
    def compress(self, states: torch.Tensor, offload: bool = True) -> dict:
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; shape = [int(v) for v in states.shape]; flat = states.reshape(-1, shape[-1]).half()
        
        # Priority: Ultra-Gila Mode (Ultra-Fused: Norm + Scale + FWHT + Quant)
        cuda_ext = load_cuda_ext()
        # V8.0 God Kernel: Fused Indices, Norms, K_MSE, R_Norms, and Signs
        indices, vec_norms, k_mse, r_norm, signs = cuda_ext.ultra_fused_full_fusion(flat.contiguous(), self.d.float().contiguous(), self.centroids.to(dev).float().contiguous())
        
        return {
            "indices": indices,
            "norms": vec_norms.squeeze(-1),
            "k_mse": k_mse.view(shape),
            "r_norm": r_norm.squeeze(-1).reshape(shape[:-1]),
            "signs": signs.reshape(shape)
        }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); signs = compressed["signs"].to(dev).float(); r_norm = compressed["r_norm"].to(dev).float()
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        q_proj = fwht(queries.float() * self.d); qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))
        scale = 1.0 / math.sqrt(self.head_dim); return term1 + scale * qjl_ip * r_norm.unsqueeze(-2)

class KudahitamCompressorHBBA:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu", hbba_4bit_ratio: float = 0.25):
        self.head_dim = head_dim; self.seed = seed; self.device = device; self.hbba_4bit_ratio = hbba_4bit_ratio
        self.d = (torch.randint(0, 2, (head_dim,), generator=torch.Generator().manual_seed(seed)) * 2 - 1).float().to(device)
        self.centroids_table = None; self.n_centroids_map = None; self.is_calibrated = False

    def _calibrate_hbba(self, sample_rotated: torch.Tensor):
        dev = sample_rotated.device; D = self.head_dim
        # 1. Variance-based allocation
        variances = sample_rotated.var(0); num_4bit = int(D * self.hbba_4bit_ratio)
        top_indices = torch.topk(variances, num_4bit).indices
        
        self.n_centroids_map = torch.full((D,), 2, dtype=torch.int32, device=dev); self.n_centroids_map[top_indices] = 16
        
        # 2. Atomic CUDA Calibrator (Lloyd-Max in GPU SRAM) - V8.7
        cuda_ext = load_cuda_ext()
        self.centroids_table = cuda_ext.hbba_calibrate_cuda(sample_rotated.float().contiguous(), self.n_centroids_map.contiguous())
        self.is_calibrated = True

    @torch.no_grad()
    def calibrate(self, states: torch.Tensor):
        if self.is_calibrated: return
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; D = states.shape[-1]
        
        # 2k Contiguous Sample (as requested)
        flat = states.reshape(-1, D)[:2000].float()
        norm = torch.norm(flat, dim=-1, keepdim=True); rotated = fwht((flat / (norm+1e-8)) * self.d.to(dev)) / math.sqrt(D); self._calibrate_hbba(rotated)

    @torch.no_grad()
    def compress(self, states: torch.Tensor, offload: bool = True) -> dict:
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; shape = states.shape; flat = states.reshape(-1, shape[-1]).half()
        cuda_ext = load_cuda_ext()
        if not self.is_calibrated: self.calibrate(states)
        indices, vec_norms, k_mse, r_norm, signs = cuda_ext.ultra_fused_hbba_fusion(flat.contiguous(), self.d.to(dev).contiguous(), self.centroids_table, self.n_centroids_map)
        return { "indices": indices, "norms": vec_norms.squeeze(-1), "k_mse": k_mse.view(shape), "r_norm": r_norm.squeeze(-1).reshape(shape[:-1]), "signs": signs.view(shape), "shape": tuple(shape) }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); signs = compressed["signs"].to(dev).float(); r_norm = compressed["r_norm"].to(dev).float()
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1)); q_proj = fwht(queries.float() * self.d.to(dev)); qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1)); scale = 1.0 / math.sqrt(self.head_dim); return term1 + scale * qjl_ip * r_norm.unsqueeze(-2)

# Baseline unchanged (Gaussian)
class KudahitamCompressorGaussian:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu"):
        self.head_dim = head_dim; self.bits = bits; self.device = device
        gen = torch.Generator(device="cpu").manual_seed(seed); G = torch.randn(head_dim, head_dim, generator=gen); Q, R = torch.linalg.qr(G); d_s = torch.sign(torch.diag(R)); d_s[d_s==0] = 1.0; self.Pi = (Q * d_s.unsqueeze(0)).to(device)
        from scipy import integrate; n_levels = 1 << bits; sigma = 1.0 / math.sqrt(head_dim)
        def pdf(x): return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))
        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i+1]) / 2.0 for i in range(n_levels - 1)]; edges = [lo * 3] + boundaries + [hi * 3]; new_c = []
            for i in range(n_levels): a, b = edges[i], edges[i+1]; num, _ = integrate.quad(lambda x: x*pdf(x), a, b); den, _ = integrate.quad(pdf, a, b); new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10: break
            centroids = new_c
        self.centroids = torch.tensor(centroids, dtype=torch.float32).to(device)
    @torch.no_grad()
    def compress(self, states: torch.Tensor, offload: bool = True) -> dict:
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; shape = [int(v) for v in states.shape]; flat = states.reshape(-1, shape[-1]).float()
        if self.Pi.device != dev: self.Pi = self.Pi.to(dev)
        if self.centroids.device != dev: self.centroids = self.centroids.to(dev)
        vec_norms = torch.norm(flat, dim=-1, keepdim=True); rotated = (flat / (vec_norms + 1e-8)) @ self.Pi.T
        indices = (rotated.unsqueeze(-1) - self.centroids).abs().argmin(-1).to(torch.uint8)
        k_mse = (self.centroids[indices.long()] @ self.Pi) * vec_norms
        return { "indices": indices, "norms": vec_norms.squeeze(-1).to(torch.float16), "rank": len(shape), "shape": tuple(shape), "k_mse": k_mse.to(torch.float16).reshape(shape) }
    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); return torch.matmul(queries.float(), k_mse.transpose(-2, -1))


# =============================================================================
# BENCHMARK SUITE (STABLE)
# =============================================================================

MODEL_NAME = "Qwen/Qwen3.5-2B"

def main():
    print("=" * 150); print(f"      KUDAHITAM-QUANT: PURE TRITON MULTI-PASS BENCHMARK (STABLE)"); print("=" * 150)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Use bnb with compute_dtype and safe loading
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto")
    except Exception as e:
        print(f"Primary load failed ({e}), attempting fallback load without explicit compute_dtype...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
    
    head_dim = 256
    try:
        with torch.no_grad():
            out = model(**tok("Hello", return_tensors="pt").to(model.device), use_cache=True)
            for k in out.past_key_values.key_cache:
                if k is not None: head_dim = (k[0].shape[-1] if isinstance(k, (tuple, list)) else k.shape[-1]); break
    except Exception as e: print(f"Warning: head_dim detection failed ({e}), using default 256.")

    print(f"Warming up GPU kernels ({model.device})..."); d_k = torch.randn(1, 1, 128, head_dim).to(model.device).half(); d_q = torch.randn(1, 1, 1, head_dim).to(model.device).half()
    for _ in range(5):
        for CC in [KudahitamCompressorV2, KudahitamCompressorGaussian]:
            comp = CC(head_dim, 1, seed=0, device=model.device); c = comp.compress(d_k, offload=False); _ = comp.asymmetric_attention_scores(d_q, c)
    
    benchmark_tasks = [
        ("Reasoning", "Five people (A, B, C, D, E) are sitting in a row. A is not next to B. C is next to D. If E is at the end, and B is exactly two seats from D, list all possible seating arrangements and explain why each fits the constraints. Analyze the logical state space step-by-step."),
        ("Math", "State and prove Minkowski's Theorem in the geometry of numbers, then explain its application in finding the shortest vector in a lattice (SVP). Detail the relationship between centrally symmetric convex bodies and lattice constants."),
        ("Story", "Construct a multi-layered narrative about a clockmaker in 18th-century Prague who builds a mechanical bird capable of predicting the future through a lunar-cycle cipher. Describe the granddaughter's role in decoding the fate of the city."),
        ("Coding", "Implement a production-ready Lock-Free Concurrent Skip List in C++20 using atomic pointers and hazard pointers for safe memory management, including search and lock-free deletion with memory_order_release.")
    ]

    print("-" * 150); print(f"{'Ctx':7s} | {'Field':10s} | {'Strategy/Bit Mode':34s} | {'Acc (V2/F)':12s} | {'Acc (G)':12s} | {'Comp(F)':8s} | {'Mem'}")
    for ctx in [10000, 40000,104000]:
        for task_name, task_desc in benchmark_tasks:
            # Diverse filler for realistic context
            fillers = [
                "The KudaHitam engine utilizes a structured projection matrix for KV-cache compression.",
                "Fast Walsh-Hadamard Transforms (FWHT) offer O(D log D) complexity which is ideal for LLMs.",
                "Spectral quantization methods focus on the frequency domain properties of the hidden states.",
                "QJL theory suggests that random projections can preserve distances with high probability.",
                "Implementing custom Triton kernels allows for fused operations and reduced VRAM bandwidth usage.",
                "Context windows in modern transformers are limited by the quadratic cost of attention.",
                "Activation outliers are a known challenge for low-bit weight and activation quantization.",
                "Softmax attention requires high precision to maintain mathematical properties of the distribution."
            ]
            filler_text = " ".join(fillers * (ctx // 200 + 1))
            prompt = task_desc + " | Research: " + filler_text
            in_ids = tok(prompt + " Question: Analyze the task based on provided context.", return_tensors="pt", truncation=True, max_length=ctx).input_ids.to(model.device); pkv = None
            with torch.no_grad():
                for i in range(0, in_ids.shape[1], 2048): out = model(in_ids[:, i:i+2048], past_key_values=pkv, use_cache=True); pkv = out.past_key_values
            gpu_c = []
            for k in pkv.key_cache:
                if k is None: gpu_c.append(None)
                elif isinstance(k, (tuple, list)): gpu_c.append([x.to(model.device) if x is not None else None for x in k])
                else: gpu_c.append(k.to(model.device))
            
            # --- ACTIVE INDICES & RANKING ---
            active_indices = [l for l, k in enumerate(gpu_c) if k is not None]
            layer_variances = []
            for l_idx in active_indices:
                keys = gpu_c[l_idx]
                if isinstance(keys, (tuple, list)): keys = keys[0]
                curr_h_dim = keys.shape[-1]
                flat = keys.reshape(-1, curr_h_dim).float()
                v_dims = flat.var(0); ratio = (v_dims.max() / (v_dims.mean() + 1e-8)).item()
                layer_variances.append((l_idx, ratio))
            layer_variances.sort(key=lambda x: x[1], reverse=True)
            ranked_active = [x[0] for x in layer_variances]

            # Final Table Collection
            if not hasattr(main, 'all_results'): main.all_results = []

            strategies = [("1-bit Baseline", 1), ("2-bit Baseline", 2), ("HBBA-Hybrid (1/4-bit 25%)", 1.75)]
            for s_name, b_count in strategies:
                res_row = {}; mem_total = 0
                for ver in ["V2", "Gaussian"]:
                    if "HBBA" in s_name and ver == "Gaussian":
                        res_row[ver] = {"acc": 0.0, "ms": 0.0}; continue
                        
                    CompClass = KudahitamCompressorV2 if ver == "V2" else KudahitamCompressorGaussian; cos_l, comp_l = [], []
                    for l_idx, keys in enumerate(gpu_c):
                        if keys is None: continue 
                        if isinstance(keys, (tuple, list)): keys = keys[0]
                        D = keys.shape[-1]; q = (keys[:, :, -1:, :] if keys.ndim == 4 else keys[:, -1:, :]).float()
                        real = torch.matmul(q, keys.float().transpose(-2, -1))
                        if "HBBA" in s_name:
                            comp = KudahitamCompressorHBBA(D, b_count, seed=l_idx, device=model.device, hbba_4bit_ratio=0.25)
                            comp.calibrate(keys)
                            b_eff = 1.75
                            l_base = (b_eff * D / 8) + 2
                            mem_total += l_base * keys.shape[1]
                        else:
                            comp = CompClass(D, b_count, seed=l_idx, device=model.device)
                            mem_total += (b_count * D / 8 + 2) * keys.shape[1]
                    res_row[ver] = {"acc": (torch.stack(cos_l).mean().item()), "ms": (sum(comp_l)/len(comp_l))*1000}
                print(f"Done: {ctx} | {task_name} | {s_name}")
                main.all_results.append((ctx, task_name, s_name, res_row['V2']['acc'], res_row['Gaussian']['acc'], res_row['V2']['ms'], res_row['Gaussian']['ms'], mem_total))

    # --- FINAL CONSOLIDATED DISPLAY ---
    print("\n" + "=" * 165)
    print(f"{'Ctx':7s} | {'Field':10s} | {'Strategy/Bit Mode':34s} | {'Acc (V2/F)':12s} | {'Acc (G)':12s} | {'Comp(V2)':8s} | {'Comp(G)':8s} | {'Mem'}")
    print("-" * 165)
    for r in main.all_results:
        print(f"{r[0]:5d} | {r[1]:10s} | {r[2]:34s} | {r[3]:.4f}     | {r[4]:.4f}     | {r[5]:8.2f} | {r[6]:8.2f} | {int(r[7]):7d}")
    print("=" * 165)
if __name__ == "__main__":
    main()
