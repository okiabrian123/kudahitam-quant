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
        try: current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError: pass
        
        _src = os.path.join(current_dir, "KudaHitamCUDA.cu")
        if os.path.exists(_src) and find_nvcc():
            build_dir = os.path.join(current_dir, "cuda_build")
            os.makedirs(build_dir, exist_ok=True)
            try:
                _KudaHitamCUDA = load(name="KudaHitamCUDA", sources=[_src], verbose=False, with_cuda=True, build_directory=build_dir)
                CUDA_EXT_AVAILABLE = True
                print("[KudaHitam] Gila Mode Activated: Raw CUDA Warp-Shuffles Enabled.")
            except:
                try: 
                    _KudaHitamCUDA = load(name="KudaHitamCUDA", sources=[_src], verbose=False, with_cuda=True, build_directory=build_dir, is_python_module=True)
                    CUDA_EXT_AVAILABLE = True
                except: pass
        return _KudaHitamCUDA
    except: return None

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

    @triton.jit
    def fractional_quantize_kernel(
        x_ptr, c0_ptr, c1_ptr, out_ptr,
        n_elements, n_cols, split_at,
        n_c0, n_c1,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid << 10) + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        col_idx = offsets % n_cols
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        is_c0 = col_idx < split_at
        
        m_dist0 = tl.full((BLOCK_SIZE,), 1e18, dtype=tl.float32)
        m_idx0 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for i in range(n_c0):
            c = tl.load(c0_ptr + i)
            d = tl.abs(vals - c)
            curr_m = d < m_dist0
            m_dist0 = tl.where(curr_m, d, m_dist0)
            m_idx0 = tl.where(curr_m, i, m_idx0)
            
        m_dist1 = tl.full((BLOCK_SIZE,), 1e18, dtype=tl.float32)
        m_idx1 = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for i in range(n_c1):
            c = tl.load(c1_ptr + i)
            d = tl.abs(vals - c)
            curr_m = d < m_dist1
            m_dist1 = tl.where(curr_m, d, m_dist1)
            m_idx1 = tl.where(curr_m, i, m_idx1)
            
        final_idx = tl.where(is_c0, m_idx0, m_idx1)
        tl.store(out_ptr + offsets, final_idx, mask=mask)

    @triton.jit
    def mask_outliers_kernel(
        x_ptr, idx_ptr,
        n_rows, n_cols, n_indices,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        idx_off = tl.arange(0, BLOCK_SIZE)
        idx_mask = idx_off < n_indices
        
        cols = tl.load(idx_ptr + idx_off, mask=idx_mask)
        # Use atomic or simple store if rows are independent
        row_off = row_idx * n_cols
        tl.store(x_ptr + row_off + cols, 0.0, mask=idx_mask)

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
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu",
                 outlier_threshold: Optional[float] = None,
                 protected_dim_count: int = 0,
                 is_domain_layer: bool = False,
                 use_outlier: bool = True,
                 domain_p_count: int = 64,
                 p_bits: int = 4,
                 use_spaced: bool = False,
                 use_spectral: bool = False, use_vwh: bool = False, use_dynamic_codebook: bool = False, use_fractional: bool = False, use_ultra: bool = False):
        self.head_dim = head_dim; self.bits = bits; self.device = device; self.outlier_threshold = outlier_threshold; self.protected_dim_count = protected_dim_count; self.protected_indices = None
        self.is_domain_layer = is_domain_layer; self.use_outlier = use_outlier; self.domain_p_count = domain_p_count; self.p_bits = p_bits; self.use_spaced = use_spaced; self.use_spectral = use_spectral; self.use_vwh = use_vwh; self.use_dynamic_codebook = use_dynamic_codebook; self.use_fractional = use_fractional; self.use_ultra = use_ultra
        
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
        # Recursive QJL for protection (Mini-Turbo)
        self.p_centroids = None; self.p_Pi = None; self.p_norms = None
        # Walsh-Sequency ordering for Spectral JPEG
        _idx = torch.arange(head_dim); _gray = _idx ^ (_idx >> 1); _rev = 0
        for _i in range(int(math.log2(head_dim))): _rev = (_rev << 1) | ((_gray >> _i) & 1)
        self.walsh_indices = _rev.to(device)

    def _init_mini_qjl(self, p_dim: int, seed: int, device: str, p_bits: int):
        if p_dim <= 0: return
        key = (p_dim, p_bits)
        if not hasattr(self, "_p_cache"): self._p_cache = {}
        if key in self._p_cache:
            self.p_Pi, self.p_centroids = self._p_cache[key]; return
        
        gen = torch.Generator(device="cpu").manual_seed(seed + 99)
        G = torch.randn(p_dim, p_dim, generator=gen); Q, R = torch.linalg.qr(G); d_s = torch.sign(torch.diag(R)); d_s[d_s==0] = 1.0; self.p_Pi = (Q * d_s.unsqueeze(0)).to(device)
        
        # Restore Gaussian Centroids for robust outlier protection
        sigma = 1.0 / math.sqrt(p_dim); n_levels = 1 << p_bits; lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = torch.linspace(lo, hi, n_levels).to(device)
        self.p_centroids = centroids; self._p_cache[key] = (self.p_Pi, self.p_centroids)

    @torch.no_grad()
    def compress(self, states: torch.Tensor, offload: bool = True) -> dict:
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; shape = [int(v) for v in states.shape]; flat = states.reshape(-1, shape[-1]).float()
        # Random sampling for variance (1024 tokens for speed + accuracy)
        s_size = min(1024, flat.shape[0]); s_idx = torch.randint(0, flat.shape[0], (s_size,), device=dev) if s_size < flat.shape[0] else slice(None)
        
        if self.use_vwh:
            v_dims = flat[s_idx].var(0); self.vwh_weights = 1.0 / (torch.sqrt(v_dims) + 1e-8); flat = flat * self.vwh_weights
        
        if not self.use_outlier: p_count = 0
        elif self.is_domain_layer: p_count = self.domain_p_count
        else: p_count = self.protected_dim_count
            
        if p_count > 0:
            if self.protected_indices is None or len(self.protected_indices) != p_count:
                v_local = flat[s_idx].var(0)
                if self.use_spectral: self.protected_indices = self.walsh_indices[:p_count]
                elif self.use_spaced: self.protected_indices = torch.linspace(0, flat.shape[-1]-1, steps=p_count, device=dev).long()
                else: self.protected_indices = torch.topk(v_local, p_count).indices
            p_vals_raw = flat[:, self.protected_indices]; self._init_mini_qjl(p_count, 0, str(dev), self.p_bits)
            p_norms = torch.norm(p_vals_raw, dim=-1, keepdim=True); p_rotated = (p_vals_raw / (p_norms + 1e-8)) @ self.p_Pi.T
            p_indices = (p_rotated.unsqueeze(-1) - self.p_centroids).abs().argmin(-1).to(torch.uint8)
            flat_q = flat.clone()
            if TRITON_AVAILABLE and flat_q.is_cuda:
                with torch.cuda.device(dev):
                    grid = (flat_q.shape[0],)
                    mask_outliers_kernel[grid](flat_q, self.protected_indices, flat_q.shape[0], flat_q.shape[1], p_count, BLOCK_SIZE=triton.next_power_of_2(p_count))
            else: flat_q[:, self.protected_indices] = 0.0
        else: p_indices = p_norms = None; flat_q = flat
            
        vec_norms = torch.norm(flat_q, dim=-1, keepdim=True)
        # Priority: Gila Mode Fused Compression (FWHT + Quantization)
        cuda_ext = load_cuda_ext()
        if CUDA_EXT_AVAILABLE and cuda_ext and flat_q.is_cuda and not self.use_fractional:
            # Prepare inputs
            if self.use_dynamic_codebook:
                # We need to compute _a for dynamic codebook - this requires first pass or partial
                # For dynamic codebook, we still use the two-step process for now
                # OR we compute _a via a reduction kernel first.
                pass
            
            if not self.use_dynamic_codebook:
                # 1. Scale input by self.d and normalize
                x_scaled = (flat_q / (vec_norms + 1e-8)) * self.d
                # 2. Fused FWHT + Argmin
                indices = cuda_ext.fused_compress(x_scaled.float().contiguous(), self.centroids.float().contiguous())
                rotated = None # Not needed anymore
                k_mse = fwht(self.centroids[indices.long()]) * self.d * vec_norms
            else:
                rotated = fwht((flat_q.float() / (vec_norms + 1e-8)) * self.d)
                _a = rotated.abs().mean(); centroids = torch.tensor([-_a, _a], device=dev)
                indices = (rotated.unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
                k_mse = fwht(centroids[indices.long()]) * self.d * vec_norms
        else:
            # Fallback to Triton/PyTorch
            rotated = fwht((flat_q.float() / (vec_norms + 1e-8)) * self.d)
            if self.use_dynamic_codebook:
                _a = rotated.abs().mean(); centroids = torch.tensor([-_a, _a], device=dev)
            else: centroids = self.centroids
            
            if self.use_fractional:
                split_at = 64 if self.use_ultra else 128
                c_h = 0.9816 * (_sigma if 'orig_sigma' in locals() else rotated.abs().mean() / 0.79788)
                c2 = torch.tensor([-c_h, 0.0, c_h], device=dev)
                idx0 = (rotated[:, :split_at].unsqueeze(-1) - c2).abs().argmin(-1).to(torch.uint8)
                idx1 = (rotated[:, split_at:].unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
                indices = torch.cat([idx0, idx1], dim=-1)
                k_mse = fwht(torch.cat([c2[indices[:, :split_at].long()], centroids[indices[:, split_at:].long()]], dim=-1)) * self.d * vec_norms
            else:
                indices = (rotated.unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
                k_mse = fwht(centroids[indices.long()]) * self.d * vec_norms
            
        if self.use_vwh: k_mse = k_mse / self.vwh_weights
        residual = flat_q - k_mse; r_norm = torch.norm(residual, dim=-1); projected = fwht(residual * self.d); signs = (projected >= 0).to(torch.int8) * 2 - 1
        return { "indices": indices, "norms": vec_norms.squeeze(-1).to(torch.float16), "p_indices": p_indices, "p_norms": p_norms.squeeze(-1).to(torch.float16) if p_norms is not None else None, "p_idx": self.protected_indices, "rank": len(shape), "shape": tuple(shape), "r_norm": r_norm.to(torch.float16).reshape(shape[:-1]), "k_mse": k_mse.to(torch.float16).reshape(shape), "signs": signs.reshape(shape), "is_domain": self.is_domain_layer }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); signs = compressed["signs"].to(dev).float(); r_norm = compressed["r_norm"].to(dev).float()
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        q_proj = fwht(queries.float() * self.d); qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))
        scale = 1.0 / math.sqrt(self.head_dim); out = term1 + scale * qjl_ip * r_norm.unsqueeze(-2)
        
        if compressed.get("p_indices") is not None:
            idx = compressed["p_idx"]; B, Sk = compressed["shape"][0], compressed["shape"][-2]
            p_indices = compressed["p_indices"].long(); p_norms = compressed["p_norms"].float().reshape(-1, 1)
            self._init_mini_qjl(len(idx), 0, str(dev), self.p_bits) # Re-initialize p_Pi/p_centroids if needed
            p_recon = (self.p_centroids[p_indices] @ self.p_Pi) * p_norms
            k_p = p_recon.view(B, -1, Sk, len(idx)); q_p = queries[..., idx].float()
            term3 = torch.matmul(q_p, k_p.transpose(-2, -1)); out += term3
        return out

# Baseline unchanged (Gaussian)
class KudahitamCompressorGaussian:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu", 
                 outlier_threshold: Optional[float] = None,
                 protected_dim_count: int = 0):
        self.head_dim = head_dim; self.bits = bits; self.device = device; self.outlier_threshold = outlier_threshold; self.protected_dim_count = protected_dim_count; self.protected_indices = None
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
        if self.protected_dim_count > 0:
            if self.protected_indices is None: v_dims = flat.var(0); self.protected_indices = torch.topk(v_dims, self.protected_dim_count).indices
            p_vals = flat[:, self.protected_indices].to(torch.float16); flat_q = flat.clone(); flat_q[:, self.protected_indices] = 0.0
        else: p_vals = None; flat_q = flat
            
        vec_norms = torch.norm(flat_q, dim=-1, keepdim=True)
        rotated = (flat_q / (vec_norms + 1e-8)) @ self.Pi.T
        indices = (rotated.unsqueeze(-1) - self.centroids).abs().argmin(-1).to(torch.uint8)
        k_mse = (self.centroids[indices.long()] @ self.Pi) * vec_norms
        return { "indices": indices, "norms": vec_norms.squeeze(-1).to(torch.float16), "p_vals": p_vals, "p_idx": self.protected_indices, "rank": len(shape), "shape": tuple(shape), "k_mse": k_mse.to(torch.float16).reshape(shape) }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1)); out = term1
        if compressed.get("p_vals") is not None:
            idx = compressed["p_idx"]; B_idx, Sk_idx = compressed["shape"][0], compressed["shape"][-2]; q_p = queries[..., idx].float(); k_p = compressed["p_vals"].to(dev).float().reshape(B_idx, -1, Sk_idx, len(idx)).squeeze(1); out += torch.matmul(q_p, k_p.transpose(-2, -1))
        return out


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

    print(f"Warming up GPU kernels ({model.device})..."); d_k = torch.randn(1, 1, 128, head_dim).to(model.device); d_q = torch.randn(1, 1, 1, head_dim).to(model.device)
    for _ in range(5):
        for CC in [KudahitamCompressorV2, KudahitamCompressorGaussian]:
            comp = CC(head_dim, 1, seed=0, device=model.device); c = comp.compress(d_k, offload=False); _ = comp.asymmetric_attention_scores(d_q, c)
    
    benchmark_tasks = [
        ("Reasoning", "Five people (A, B, C, D, E) are sitting in a row. A is not next to B. C is next to D. If E is at the end, and B is exactly two seats from D, list all possible seating arrangements and explain why each fits the constraints. Analyze the logical state space step-by-step."),
        ("Math", "State and prove Minkowski's Theorem in the geometry of numbers, then explain its application in finding the shortest vector in a lattice (SVP). Detail the relationship between centrally symmetric convex bodies and lattice constants."),
        ("Story", "Construct a multi-layered narrative about a clockmaker in 18th-century Prague who builds a mechanical bird capable of predicting the future through a lunar-cycle cipher. Describe the granddaughter's role in decoding the fate of the city."),
        ("Coding", "Implement a production-ready Lock-Free Concurrent Skip List in C++20 using atomic pointers and hazard pointers for safe memory management, including search and lock-free deletion with memory_order_release.")
    ]

    for ctx in [10000, 40000]:
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
            messages = [{"role": "user", "content": task_desc}]
            input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            in_ids = tok(input_text + " | Research: " + filler_text, return_tensors="pt", truncation=True, max_length=ctx).input_ids.to(model.device); pkv = None
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

            # Positional Strategies Indices (for 6 active layers)
            n_act = len(active_indices); mid = n_act // 2
            # Strategies: (Name, Bits, Outlier_Idxs, Use_Outlier, Outlier_P, Outlier_Bits, Frac_Idxs)
            strategies = [
                ("1-bit Baseline", 1, [], False, 0, 4, []),
                ("2-bit Baseline", 2, [], False, 0, 4, []),
            ]

            for s_name, b_count, d_set, use_out, d_p, o_bits, f_set in strategies:
                res_row = {}; mem_total = 0
                for ver in ["V2", "Gaussian"]:
                    CompClass = KudahitamCompressorV2 if ver == "V2" else KudahitamCompressorGaussian; cos_l, comp_l = [], []
                    for l_idx, keys in enumerate(gpu_c):
                        if keys is None: continue 
                        if isinstance(keys, (tuple, list)): keys = keys[0]
                        B, H, S, D = keys.shape
                        q = (keys[:, :, -1:, :] if keys.ndim == 4 else keys[:, -1:, :]).float()
                        real = torch.matmul(q, (keys_f := keys.float()).transpose(-2, -1))
                        is_d = (l_idx in d_set); is_f = (l_idx in f_set)
                        
                        # Logic: Domain dimensions ONLY for d_set, else 0 outlier for Selective feel.
                        curr_p_dim = d_p if is_d else 0; curr_o_bits = o_bits
                        
                        if ver == "V2":
                            curr_use_frac = is_f; curr_use_ultra = (is_f and "Ultra" in s_name)
                            comp = CompClass(D, b_count, seed=l_idx, device=model.device, is_domain_layer=is_d, use_outlier=use_out, domain_p_count=curr_p_dim, p_bits=curr_o_bits, use_spaced=("Spaced" in s_name), use_spectral=("Spectral" in s_name), use_vwh=("VWH" in s_name), use_dynamic_codebook=("Dynamic" in s_name), use_fractional=curr_use_frac, use_ultra=curr_use_ultra)
                            b_eff = 1.25 if curr_use_ultra else (1.5 if curr_use_frac else b_count)
                            l_base = (b_eff * D / 8) + 2
                            l_out = ((curr_p_dim * curr_o_bits / 8) + 2) if (use_out and is_d) else 0
                            mem_total += (l_base + l_out) * H
                        else:
                            p_count_g = curr_p_dim if (is_d and use_out) else (16 if use_out else 0)
                            comp = CompClass(D, b_count, seed=l_idx, device=model.device, protected_dim_count=p_count_g)
                            
                        torch.cuda.synchronize(); t0 = time.perf_counter(); c = comp.compress(keys, offload=False); torch.cuda.synchronize(); comp_l.append(time.perf_counter() - t0); s = comp.asymmetric_attention_scores(q, c); cos_l.append(F.cosine_similarity(real.flatten(), s.flatten(), dim=0))
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
