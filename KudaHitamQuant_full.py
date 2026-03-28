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
    def fwht_kernel(x_ptr, y_ptr, n_rows, stride_x, D: tl.constexpr):
        """Single-pass shared-memory FWHT (Orthonormal & Fast)."""
        pid = tl.program_id(0)
        block_start = pid * stride_x
        
        # Load entire row (D=256) into shared memory
        offs = tl.arange(0, D)
        sdata = tl.load(x_ptr + block_start + offs).to(tl.float32)
        
        # Butterfly stages (Vectorized with masking to avoid indexing errors)
        # We unroll manually to ensure each stage has constant shapes for tl.reshape
        
        # Stage 0: h=1
        s3d = tl.reshape(sdata, [128, 2, 1])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [128, 2, 1])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [128, 2, 1])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 1: h=2
        s3d = tl.reshape(sdata, [64, 2, 2])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [64, 2, 2])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [64, 2, 2])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 2: h=4
        s3d = tl.reshape(sdata, [32, 2, 4])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [32, 2, 4])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [32, 2, 4])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 3: h=8
        s3d = tl.reshape(sdata, [16, 2, 8])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [16, 2, 8])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [16, 2, 8])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 4: h=16
        s3d = tl.reshape(sdata, [8, 2, 16])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [8, 2, 16])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [8, 2, 16])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 5: h=32
        s3d = tl.reshape(sdata, [4, 2, 32])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [4, 2, 32])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [4, 2, 32])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 6: h=64
        s3d = tl.reshape(sdata, [2, 2, 64])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [2, 2, 64])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [2, 2, 64])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])
        
        # Stage 7: h=128
        s3d = tl.reshape(sdata, [1, 2, 128])
        u = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 0, s3d, 0.0), axis=1, keep_dims=True), [1, 2, 128])
        v = tl.broadcast_to(tl.sum(tl.where(tl.arange(0, 2)[None, :, None] == 1, s3d, 0.0), axis=1, keep_dims=True), [1, 2, 128])
        sdata = tl.reshape(tl.where(tl.arange(0, 2)[None, :, None] == 0, u + v, u - v), [256])

        # Normalize to orthonormal (1/sqrt(D))
        scale = 1.0 / tl.sqrt(float(D))
        tl.store(y_ptr + block_start + offs, (sdata * scale).to(x_ptr.dtype.element_ty))

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
    if not TRITON_AVAILABLE or not x.is_cuda: return fwht_pytorch(x)
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous(); D = x.shape[-1]
    if (D & (D - 1)) != 0: return fwht_pytorch(x) # Fallback if not power-of-2
    y = torch.empty_like(x)
    with torch.cuda.device(x.device):
        grid = (x.shape[0],)
        fwht_kernel[grid](x, y, x.shape[0], x.stride(0), D=D)
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
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.d = torch.sign(torch.randn(head_dim, generator=gen)).to(device)
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
            
        vec_norms = torch.norm(flat_q, dim=-1, keepdim=True); rotated = fwht((flat_q.float() / (vec_norms + 1e-8)) * self.d)
        _a = rotated.abs().mean(); _sigma = _a / 0.79788
        if self.use_dynamic_codebook: centroids = torch.tensor([-_a, _a], device=dev)
        else: centroids = self.centroids
        
        if self.use_fractional:
            split_at = 64 if self.use_ultra else 128
            c_h = 0.9816 * _sigma; c2 = torch.tensor([-c_h, 0.0, c_h], device=dev)
            if TRITON_AVAILABLE and rotated.is_cuda:
                with torch.cuda.device(dev):
                    i_tmp = torch.empty_like(rotated, dtype=torch.int32).contiguous()
                    grid = (triton.cdiv(rotated.numel(), 1024),)
                    fractional_quantize_kernel[grid](rotated.contiguous(), c2.contiguous(), centroids.contiguous(), i_tmp, rotated.numel(), rotated.shape[1], split_at, len(c2), len(centroids), BLOCK_SIZE=1024)
                    indices = i_tmp.to(torch.uint8)
            else:
                idx0 = (rotated[:, :split_at].unsqueeze(-1) - c2).abs().argmin(-1).to(torch.uint8)
                idx1 = (rotated[:, split_at:].unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
                indices = torch.cat([idx0, idx1], dim=-1)
            k_mse = fwht(torch.cat([c2[indices[:, :split_at].long()], centroids[indices[:, split_at:].long()]], dim=-1)) * self.d * vec_norms
        else:
            indices = torch.empty_like(rotated, dtype=torch.uint8)
            if TRITON_AVAILABLE and rotated.is_cuda:
                with torch.cuda.device(dev):
                    i_tmp = torch.empty_like(rotated, dtype=torch.int32).contiguous()
                    grid = (triton.cdiv(rotated.numel(), 1024),); quantize_kernel_pure[grid](rotated.contiguous(), centroids.contiguous(), i_tmp, rotated.numel(), len(centroids), BLOCK_SIZE=1024)
                    indices = i_tmp.to(torch.uint8)
            else: indices = (rotated.unsqueeze(-1) - centroids).abs().argmin(-1).to(torch.uint8)
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
        if self.protected_dim_count > 0:
            if self.protected_indices is None: v_dims = flat.var(0); self.protected_indices = torch.topk(v_dims, self.protected_dim_count).indices
            p_vals = flat[:, self.protected_indices].to(torch.float16); flat_q = flat.clone(); flat_q[:, self.protected_indices] = 0.0
        else: p_vals = None; flat_q = flat
            
        vec_norms = torch.norm(flat_q, dim=-1, keepdim=True)
        rotated = (flat_q / (vec_norms + 1e-8)) @ self.Pi.to(dev).T
        indices = (rotated.unsqueeze(-1) - self.centroids.to(dev)).abs().argmin(-1).to(torch.uint8)
        k_mse = (self.centroids.to(dev)[indices.long()] @ self.Pi.to(dev)) * vec_norms
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

    print("-" * 150); print(f"{'Ctx':7s} | {'Field':10s} | {'Strategy/Bit Mode':34s} | {'Acc (V2/F)':12s} | {'Acc (G)':12s} | {'Comp(F)':8s} | {'Mem'}")
    for ctx in [10000, 40000]:
        for task_name, task_desc in benchmark_tasks:
            messages = [{"role": "user", "content": task_desc}]
            try:
                input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            except:
                input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            in_ids = tok(input_text + " | Context Block: " * (ctx // 50), return_tensors="pt").input_ids.to(model.device); pkv = None
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
                ("Ranked-Top-1 (1L 64/0)",   1, ranked_active[:1], True, 64, 4, []),
                ("Pos-Mid-4bit (1L 32/0)",   1, [active_indices[mid]], True, 64, 4, []),
                ("Pos-Mid-4bit (1L 16/0)",   1, [active_indices[mid]], True, 32, 4, []),
                ("Pos-Mid-4bit (1L 32/0)",   1, [active_indices[mid]], True, 16, 4, []),
                ("Pos-Mid-3bit (1L 16/0)",   1, [active_indices[mid]], True, 64, 3, []), 
                ("Pos-Mid-Only (1L 64/0)",   1, [active_indices[mid]], True, 64, 4, []),
                ("Pos-Mid-3bit (1L 64/0)",   1, [active_indices[mid]], True, 64, 3, []), 
                ("Pos-Mid-5bit (1L 64/0)",   1, [active_indices[mid]], True, 64, 5, []),
                ("Pos-Mid-5bit (1L 32/0)",   1, [active_indices[mid]], True, 32, 5, []),
                ("Pos-Mid-5bit (1L 8/0)",    1, [active_indices[mid]], True, 8, 5, []),
                ("Pos-Mid-6bit (1L 16/0)",   1, [active_indices[mid]], True, 16, 6, []),
                ("Pos-Mid-6bit (1L 8/0)",    1, [active_indices[mid]], True, 8, 6, []),
                ("Pos-Mid-8bit (1L 4/0)",    1, [active_indices[mid]], True, 4, 8, []),
                ("Pos-Mid-Adj-4bit (2L 32/0)",    1, active_indices[mid:mid+2], True, 32, 4, []),
                ("Pos-Mid-Space-5bit (2L 32/0)",  1, [active_indices[mid-1], active_indices[mid+1]], True, 32, 8, []),
                ("Pos-Mid-Full (1L 3-bit)",  1, [active_indices[mid]], True, 256, 3, []), 
                ("Pos-Mid-Full (1L 4-bit)",  1, [active_indices[mid]], True, 256, 4, []), 
                ("Pos-Top-Bottom (2L 64/0)", 1, [active_indices[0], active_indices[-1]], True, 64, 4, []),
                ("Pos-Mid-Adj (2L 64/0)",    1, active_indices[mid:mid+2], True, 64, 4, []),
                ("Pos-Mid-Space (2L 64/0)",  1, [active_indices[mid-1], active_indices[mid+1]], True, 64, 4, []),
                ("Pos-Near-Ends (2L 64/0)",  1, [active_indices[1], active_indices[-2]], True, 64, 4, []),
                ("Pos-Mid-Spaced (1L 64/0)", 1, [active_indices[mid]], True, 64, 4, []),
                ("Pos-Mid-Spectral (1L 64/0)", 1, [active_indices[mid]], True, 64, 4, []),
                ("VWH-1bit (0-overhead)", 1, [], False, 0, 4, []),
                ("HBAA-Adaptive (1L 128-budget)", 1, [active_indices[mid]], True, 64, 4, []),
                ("Dynamic-1bit (0-overhead)", 1, [], False, 0, 4, []),
                ("Fractional-1.5bit (+20B)", 1, [], False, 0, 4, active_indices),
                ("Fractional-Mid-5bit (1L 32/0)", 1, [active_indices[mid]], True, 32, 5, [active_indices[mid]]),
                ("Selective-Fractional (2L 0/0)", 1, active_indices[mid-1:mid+1], False, 0, 4, active_indices[mid-1:mid+1]),
                ("Selective-Fractional-Mid (2L 8/8)", 1, active_indices[mid-1:mid+1], True, 8, 8, [active_indices[i] for i in [0, 2, 4, 5]]),
                ("Ultra-Mid-5bit (1L 32/0)", 1, [active_indices[mid]], True, 32, 5, [active_indices[mid]])
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
                            
                        torch.cuda.synchronize(); t0 = time.perf_counter(); c = comp.compress(keys, offload=False); torch.cuda.synchronize(); comp_l.append(time.perf_counter() - t0); s = comp.asymmetric_attention_scores(q, c); cos_l.append(F.cosine_similarity(real.flatten(), s.flatten(), dim=0).item())
                    res_row[ver] = {"acc": sum(cos_l)/len(cos_l), "ms": (sum(comp_l)/len(comp_l))*1000}
                print(f"Done: {ctx} | {task_name} | {s_name}")
                main.all_results.append((ctx, task_name, s_name, res_row['V2']['acc'], res_row['Gaussian']['acc'], res_row['V2']['ms'], mem_total))

    # --- FINAL CONSOLIDATED DISPLAY ---
    print("\n" + "=" * 150)
    print(f"{'Ctx':7s} | {'Field':10s} | {'Strategy/Bit Mode':34s} | {'Acc (V2/F)':12s} | {'Acc (G)':12s} | {'Comp(F)':8s} | {'Mem'}")
    print("-" * 150)
    for r in main.all_results:
        print(f"{r[0]:5d} | {r[1]:10s} | {r[2]:34s} | {r[3]:.4f}     | {r[4]:.4f}     | {r[5]:8.2f} | {int(r[6]):7d}")
    print("=" * 150)
if __name__ == "__main__":
    main()
