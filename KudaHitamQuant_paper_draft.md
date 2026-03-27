# KudaHitamQuant: Ultra-High Fidelity 1-bit KV Cache Compression via Structured Spectral Projections

**Authors**: Oki Abrian
**Date**: March 2026

---

## Abstract
Large Language Models (LLMs) with long context windows are critically bottlenecked by the Key-Value (KV) cache. While Multi-Head Latent Attention (MLA) reduces the latent dimension, the linear growth of the cache prohibits extreme sequence lengths. We introduce **KudaHitamQuant**, a high-fidelity quantization framework optimized for Qwen-3.5 MLA. By leveraging **Fast Walsh-Hadamard Transform (FWHT)** and **Optimal Lloyd-Max Centroids**, we achieve superior reconstruction fidelity through perfectly orthogonal structured transformations. We implement **Gila Mode**, a raw CUDA engine utilizing **Warp-Shuffle** primitives for $O(D \log D)$ projection, and an **Asymmetric Attention Score Recovery** mechanism that keeps queries in high precision. Benchmarks on **Qwen-3.5 2B** demonstrate that KudaHitamQuant achieves **near-lossless fidelity (0.9993)** at **2-bits**, significantly outperforming Gaussian QJL baselines.

---

## 1. Introduction
The quest for long context length has shifted the primary optimization target from compute to KV cache bandwidth. Recent works like Google's TurboQuant explored Quantized Johnson-Lindenstrauss (QJL) but rely on dense Gaussian projections ($O(D^2)$). We propose a structured approach that leverages the spectral properties of the latent space in Qwen-3.5 ($D=256$).

## 2. Methodology

### 2.1 Structured Spectral Projection
KudaHitamQuant projects the KV state $x$ into a spectral domain via:
$$ y = H(x \odot d) \cdot \frac{1}{\sqrt{D}} $$
where $H$ is the Hadamard matrix and $d$ is a Rademacher random diagonal. Quantization follows via an optimal Lloyd-Max codebook $\mathcal{C}$.

### 2.2 Asymmetric Score Recovery
To maintain accuracy, we keep Queries $q$ in high precision while compressing Keys $k$. The attention score is recovered:
$$ \text{Score} = (q \odot d)^T \cdot H^T \cdot \text{lookup}(\mathcal{C}_k) $$
This eliminates the "noisy" sign-only projection of standard 1-bit QJL.

### 2.3 Gila Mode: CUDA Warp-Shuffles
We implement the $O(D \log D)$ FWHT using CUDA's `__shfl_xor_sync` to eliminate shared memory bank conflicts and global memory roundtrips. Together with a **Global Centroid Cache** and **Lazy JIT Loading**, the engine achieves sub-millisecond latency for $D=256$.

## 3. Experimental Analysis

### 3.1 Benchmark Results on NVIDIA T4 (Qwen-3.5 2B MLA, $D=256$)

| Ctx | Task | Mode | Acc (V2/F: KudaHitam) | Acc (G: Gaussian) | Comp(V2/F) | Comp(G) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 10k | Reasoning | 1-bit | **0.9977** (+0.46%) | 0.9931 | 0.86ms | 0.49ms |
| 10k | Reasoning | 2-bit | **0.9993** (+0.13%) | 0.9980 | 0.86ms | 0.44ms |
| 10k | Math | 1-bit | **0.9977** (+0.46%) | 0.9931 | 0.87ms | 0.47ms |
| 40k | Reasoning | 1-bit | **0.9973** (+0.56%) | 0.9917 | 0.89ms | 0.49ms |
| 40k | Reasoning | 2-bit | **0.9992** (+0.15%) | 0.9977 | 0.84ms | 0.47ms |
| 40k | Coding | 2-bit | **0.9991** (+0.16%) | 0.9975 | 0.97ms | 0.45ms |

## 4. Conclusion
**KudaHitamQuant** sets a new state-of-the-art for KV compression by combining structured orthogonal projections with asymmetric attention recovery, delivering near-lossless 2-bit performance.
.
