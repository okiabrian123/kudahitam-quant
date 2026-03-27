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

# --- PURE TRITON KERNELS ---

if TRITON_AVAILABLE:
    @triton.jit
    def fwht_stage_kernel(x_ptr, stride_row, step, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        """Stable multi-pass FWHT stage (Bitwise Optimized)."""
        row_idx = tl.program_id(0)
        col_idx = (tl.program_id(1) << 10) + tl.arange(0, BLOCK_SIZE)
        mask = (col_idx < D) & ((col_idx & step) == 0)
        
        # Use shift for row offset if stride_row is a power of 2
        # For simplicity in this general kernel, we stick to multiplication 
        # but the user can pass a LOG_STRIDE if they want absolute zero overhead.
        row_offset = row_idx * stride_row
        ptr_l = x_ptr + row_offset + col_idx
        ptr_r = x_ptr + row_offset + (col_idx + step)
        val_l = tl.load(ptr_l, mask=mask)
        val_r = tl.load(ptr_r, mask=mask)
        tl.store(ptr_l, val_l + val_r, mask=mask)
        tl.store(ptr_r, val_l - val_r, mask=mask)

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
    """Pure Triton multi-pass FWHT."""
    if not TRITON_AVAILABLE or not x.is_cuda:
        return fwht_pytorch(x)
    orig_shape = x.shape
    D = orig_shape[-1]
    if (D & (D - 1)) != 0:
        next_pow2 = 1 << (D - 1).bit_length(); x = F.pad(x, (0, next_pow2 - D)); D = next_pow2
    x = x.reshape(-1, D).contiguous()
    n_rows = x.shape[0]; stride = x.stride(0); step = 1
    with torch.cuda.device(x.device):
        while step < D:
            # BLOCK_SIZE=1024 used for bitwise synergy
            grid = (n_rows, triton.cdiv(D >> 1, 1024))
            fwht_stage_kernel[grid](x, stride, step, D, BLOCK_SIZE=1024)
            step <<= 1
    return x[..., :orig_shape[-1]].reshape(orig_shape)

def fwht_pytorch(x: torch.Tensor):
    orig_shape = x.shape; D = orig_shape[-1]
    if (D & (D - 1)) != 0:
        next_pow2 = 1 << (D - 1).bit_length(); x = F.pad(x, (0, next_pow2 - D)); D = next_pow2
    x = x.reshape(-1, D); i = 1
    while i < D:
        x = x.view(-1, D // (i << 1), 2, i); a, b = x[:, :, 0, :], x[:, :, 1, :]
        new_a, new_b = a + b, a - b; x[:, :, 0, :], x[:, :, 1, :] = new_a, new_b; i <<= 1
    return x[..., :orig_shape[-1]].reshape(orig_shape)

# --- COMPRESSORS ---

class KudaHitamQuantCompressorV2:
    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu",
                 outlier_threshold: Optional[float] = None,
                 protected_dim_count: int = 0):
        self.head_dim = head_dim; self.bits = bits; self.device = device; self.outlier_threshold = outlier_threshold; self.protected_dim_count = protected_dim_count; self.protected_indices = None
        gen = torch.Generator(device="cpu").manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen); Q, R = torch.linalg.qr(G); d_s = torch.sign(torch.diag(R)); d_s[d_s==0] = 1.0; self.Pi = (Q * d_s.unsqueeze(0)).to(device)
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
        self.d = torch.sign(torch.randn(head_dim, generator=gen)).to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor, offload: bool = True) -> dict:
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; shape = [int(v) for v in states.shape]; flat = states.reshape(-1, shape[-1]).float()
        if self.protected_dim_count > 0:
            if self.protected_indices is None: v_dims = flat.var(0); self.protected_indices = torch.topk(v_dims, self.protected_dim_count).indices
            p_vals = flat[:, self.protected_indices].to(torch.float16).cpu() if offload else flat[:, self.protected_indices].to(torch.float16); flat_q = flat.clone(); flat_q[:, self.protected_indices] = 0.0
        else: p_vals = None; flat_q = flat
        vec_norms = torch.norm(flat_q, dim=-1, keepdim=True); rotated = (flat_q / (vec_norms + 1e-8)) @ self.Pi.to(dev).T; centroids = self.centroids.to(dev); indices = torch.empty_like(rotated, dtype=torch.int32).contiguous()
        if TRITON_AVAILABLE and rotated.is_cuda:
            with torch.cuda.device(dev):
                rot_f = rotated.contiguous(); cent_f = centroids.contiguous(); grid = (triton.cdiv(rotated.numel(), 1024),); quantize_kernel_pure[grid](rot_f, cent_f, indices, rotated.numel(), len(centroids), BLOCK_SIZE=1024); indices = indices.to(torch.uint8)
        else: indices = (rotated.unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
        k_mse = (centroids[indices.long()] @ self.Pi.to(dev)) * vec_norms; residual = flat_q - k_mse; r_norm = torch.norm(residual, dim=-1); projected = fwht(residual * self.d.to(dev)); signs = (projected >= 0).to(torch.int8) * 2 - 1
        def mb_cpu(x): return x.cpu() if (offload and x is not None) else x
        return { "indices": mb_cpu(indices), "norms": mb_cpu(vec_norms.squeeze(-1).to(torch.float16)), "p_vals": p_vals, "p_idx": self.protected_indices, "rank": len(shape), "shape": tuple(shape), "r_norm": mb_cpu(r_norm.to(torch.float16).reshape(shape[:-1])), "out_mask": mb_cpu(None), "k_mse": mb_cpu(k_mse.to(torch.float16).reshape(shape)), "signs": mb_cpu(signs.reshape(shape)) }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); signs = compressed["signs"].to(dev).float(); r_norm = compressed["r_norm"].to(dev).float()
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1)); q_proj = fwht(queries.float() * self.d.to(dev)); qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1)); scale = math.sqrt(math.pi / 2) / self.head_dim; out = term1 + scale * qjl_ip * r_norm.unsqueeze(-2)
        if compressed.get("p_vals") is not None:
            idx = compressed["p_idx"]; B_idx, Sk_idx = compressed["shape"][0], compressed["shape"][-2]; q_p = queries[..., idx].float(); k_p = compressed["p_vals"].to(dev).float().reshape(B_idx, -1, Sk_idx, len(idx)).squeeze(1); term3 = torch.matmul(q_p, k_p.transpose(-2, -1)); out += term3
        return out

class KudaHitamQuantCompressorGaussian:
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
        gen2 = torch.Generator(device="cpu").manual_seed(seed + 10000); self.S = torch.randn(head_dim, head_dim, generator=gen2).to(device)

    @torch.no_grad()
    def compress(self, states: torch.Tensor, offload: bool = True) -> dict:
        if isinstance(states, (list, tuple)): states = states[0]
        dev = states.device; shape = [int(v) for v in states.shape]; flat = states.reshape(-1, shape[-1]).float()
        if self.protected_dim_count > 0:
            if self.protected_indices is None: v_dims = flat.var(0); self.protected_indices = torch.topk(v_dims, self.protected_dim_count).indices
            p_vals = flat[:, self.protected_indices].to(torch.float16).cpu() if offload else flat[:, self.protected_indices].to(torch.float16); flat_q = flat.clone(); flat_q[:, self.protected_indices] = 0.0
        else: p_vals = None; flat_q = flat
        vec_norms = torch.norm(flat_q, dim=-1, keepdim=True); rotated = (flat_q / (vec_norms + 1e-8)) @ self.Pi.to(dev).T; centroids = self.centroids.to(dev); indices = torch.empty_like(rotated, dtype=torch.int32).contiguous()
        if TRITON_AVAILABLE and rotated.is_cuda:
            with torch.cuda.device(dev):
                rot_f = rotated.contiguous(); cent_f = centroids.contiguous(); grid = (triton.cdiv(rotated.numel(), 1024),); quantize_kernel_pure[grid](rot_f, cent_f, indices, rotated.numel(), len(centroids), BLOCK_SIZE=1024); indices = indices.to(torch.uint8)
        else: indices = (rotated.unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
        k_mse = (reconstructed := centroids[indices.long()] @ self.Pi.to(dev)) * vec_norms; residual = flat_q - k_mse; r_norm = torch.norm(residual, dim=-1); projected = residual @ self.S.to(dev).T; signs = (projected >= 0).to(torch.int8) * 2 - 1
        def mb_cpu(x): return x.cpu() if (offload and x is not None) else x
        return { "indices": mb_cpu(indices), "norms": mb_cpu(vec_norms.squeeze(-1).to(torch.float16)), "p_vals": p_vals, "p_idx": self.protected_indices, "rank": len(shape), "shape": tuple(shape), "r_norm": mb_cpu(r_norm.to(torch.float16).reshape(shape[:-1])), "out_mask": mb_cpu(None), "k_mse": mb_cpu(k_mse.to(torch.float16).reshape(shape)), "signs": mb_cpu(signs.reshape(shape)) }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        dev = queries.device; k_mse = compressed["k_mse"].to(dev).float(); signs = compressed["signs"].to(dev).float(); r_norm = compressed["r_norm"].to(dev).float()
        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1)); q_proj = queries.float() @ self.S.to(dev).T; qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1)); scale = math.sqrt(math.pi / 2) / self.head_dim; out = term1 + scale * qjl_ip * r_norm.unsqueeze(-2)
        if compressed.get("p_vals") is not None:
            idx = compressed["p_idx"]; B_idx, Sk_idx = compressed["shape"][0], compressed["shape"][-2]; q_p = queries[..., idx].float(); k_p = compressed["p_vals"].to(dev).float().reshape(B_idx, -1, Sk_idx, len(idx)).squeeze(1); term3 = torch.matmul(q_p, k_p.transpose(-2, -1)); out += term3
        return out

# =============================================================================
# BENCHMARK SUITE
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
        for CC in [KudaHitamQuantCompressorV2, KudaHitamQuantCompressorGaussian]:
            comp = CC(head_dim, 1, seed=0, device=model.device); c = comp.compress(d_k, offload=False); _ = comp.asymmetric_attention_scores(d_q, c)
    
    print("-" * 150); print(f"{'Context':8s} | {'Bits':10s} | {'Acc (F)':10s} | {'Acc (G)':10s} | {'Comp(F) ms':12s} | {'Comp(G) ms':12s}")
    for ctx in [10000, 40000]:
        prompt = f"Quantum memory optimization session. Standardizing the KV cache for very long context window. " + "Context injection: The primary method is KudaHitamQuant using Triton-optimized Multi-Pass FWHT. " * (ctx // 100)
        in_ids = tok(prompt + " Question: What is the primary method?", return_tensors="pt").input_ids.to(model.device); pkv = None
        with torch.no_grad():
            for i in range(0, in_ids.shape[1], 2048): out = model(in_ids[:, i:i+2048], past_key_values=pkv, use_cache=True); pkv = out.past_key_values
        gpu_c = []
        for k in pkv.key_cache:
            if k is None: gpu_c.append(None)
            elif isinstance(k, (tuple, list)): gpu_c.append([x.to(model.device) if x is not None else None for x in k])
            else: gpu_c.append(k.to(model.device))
        for b in [1, 2]:
            res_row = {}
            for ver in ["V2", "Gaussian"]:
                CompClass = KudaHitamQuantCompressorV2 if ver == "V2" else KudaHitamQuantCompressorGaussian; cos_l, comp_l = [], []
                for l_idx, keys in enumerate(gpu_c):
                    if keys is None: continue
                    if isinstance(keys, (tuple, list)): keys = keys[0]
                    q = (keys[:, :, -1:, :] if keys.ndim == 4 else keys[:, -1:, :]).float()
                    real = torch.matmul(q, (keys_f := keys.float()).transpose(-2, -1))
                    comp = CompClass(keys.shape[-1], b, seed=l_idx, device=model.device); torch.cuda.synchronize(); t0 = time.perf_counter(); c = comp.compress(keys, offload=False); torch.cuda.synchronize(); comp_l.append(time.perf_counter() - t0); s = comp.asymmetric_attention_scores(q, c); cos_l.append(F.cosine_similarity(real.flatten(), s.flatten(), dim=0).item())
                res_row[ver] = {"acc": sum(cos_l)/len(cos_l), "ms": (sum(comp_l)/len(comp_l))*1000}
            print(f"{ctx:8d} | {b:2d}-bit QJL | {res_row['V2']['acc']:.4f}     | {res_row['Gaussian']['acc']:.4f}     | {res_row['V2']['ms']:12.2f} | {res_row['Gaussian']['ms']:12.2f}")
if __name__ == "__main__":
    main()
