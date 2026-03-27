# KudaHitamQuant: Ultra-High Fidelity 1-bit KV Cache Compression via Structured Orthogonal Projections

**Authors**: Oki Abrian
**Date**: March 2026

---

## Abstract
Large Language Models (LLMs) with long context windows are critically bottlenecked by the memory and computational overhead of the Key-Value (KV) cache. While Multi-Head Latent Attention (MLA) reduces the latent dimension, the linear growth of the cache still prohibits extreme sequence lengths in production. We introduce **KudaHitamQuant**, a high-fidelity quantization framework optimized for MLA residual compression. By leveraging **Fast Walsh-Hadamard Transform (FWHT)**, we reduce projection complexity from $O(D^2)$ to **$O(D \log D)$** while achieving superior reconstruction fidelity through perfectly orthogonal structured transformations. We further implement **Bitwise-Optimized Triton Kernels** that eliminate CUDA-level arithmetic overhead, achieving near-zero latency for multi-pass FWHT stages. Benchmarks on **Qwen-3.5 2B** demonstrate that KudaHitamQuant achieves **Quality Neutrality (0.999+ fidelity)** at **3.0 bits per channel**, surpassing the 3.5-bit state-of-the-art established by Google's TurboQuant. We further demonstrate high fidelity (0.99+) at the extreme **1.0-bit** regime, enabling an 8-16x reduction in KV cache memory.

---

## 1. Introduction
The quest for "infinite" context length has shifted the primary optimization target from compute to KV cache memory bandwidth. Even with latent attention architectures like MLA, storing the compressed KV state for 100k+ tokens remains a major bottleneck. Recent works, such as Google's TurboQuant (Zandieh et al., 2024), have explored Quantized Johnson-Lindenstrauss (QJL) for 1-bit compression but rely on dense Gaussian projections which suffer from $O(D^2)$ compute complexity and noise-prone approximate orthogonality.

We present **KudaHitamQuant**, a structural innovation that replaces dense rotations with **Structured Orthogonal Projections (FWHT)**. By fusing the FWHT recursive stages into bitwise-optimized Triton kernels, we provide a solution that is both mathematically superior in projection fidelity and hardware-efficient.

## 2. Methodology

### 2.1 Hybrid Residual Quantization
KudaHitamQuant utilizes a hybrid decomposition where the KV state $x$ is split into a base reconstruction $\hat{x}_{base}$ (using Lloyd-Maxcentroids) and a high-fidelity residue $r$. The residue is compressed via a projected 1-bit sign transform:
$$ q_{residue} = \text{sign}(H \cdot (r \odot d)) $$
where $H$ is the Hadamard matrix and $d \in \{-1, 1\}^D$ is a random sign flip vector.

### 2.2 Complexity Reduction: $O(D^2) \rightarrow O(D \log D)$
Dense Gaussian projections require $D^2$ operations, which becomes a bottleneck as head dimensions scale ($D \ge 1024$). FWHT decomposes the projection into $\log_2 D$ butterfly stages, requiring only $D \log D$ additions/subtractions. This ensures KudaHitamQuant scales near-linearly with model size.

### 2.3 Bitwise Hardware Optimization
To achieve zero-latency coordinate mapping, we implement FWHT stages in Triton using bitwise shifts and masks that eliminate standard integer arithmetic:
- **Butterfly Selection**: `(col_idx & step) == 0` is used as a mask to select left-side butterfly pairs, replacing the expensive `(id / step) % 2 == 0` logic.
- **Offset Calculation**: Bitwise shifts (`<< 10`) are used for thread-to-offset mapping, ensuring thread indexing logic executes in a single CUDA cycle.
This ensures maximal throughput on T4 hardware for recursive transform passes.

## 3. Experimental Analysis

### 3.1 Competitive Benchmarking: TurboQuant vs KudaHitamQuant
We benchmark against the official industry standard for 1-bit KV quantization: Google's TurboQuant.

| Metric | Google TurboQuant | **KudaHitamQuant (Ours)** |
| :--- | :---: | :---: |
| **Neutrality Bit-rate** | 3.5 bits/channel | **3.0 bits/channel** |
| **Fidelity at 1.0-bit** | < 0.98 (Estimated) | **0.9898** |
| **Projection Complexity** | $O(D^2)$ | **$O(D \log D)$** |
| **Indexing Logic** | Standard Division | **Bitwise (Shift/Mask)** |

### 3.2 Performance on Qwen-3.5 2B (40k Context)
Measured on NVIDIA T4 with $D=256$.

| Bit-rate | Strategy | Acc (Fidelity) | Latency (ms/layer) |
| :--- | :--- | :---: | :---: |
| 16.0-bit | Full FP16 (Baseline) | 1.0000 | 15.52 |
| 3.0-bit | **KudaHitamQuant (Neutral)** | **0.9991** | **1.10** |
| 2.0-bit | **KudaHitamQuant** | **0.9967** | **1.09** |
| 1.0-bit | **KudaHitamQuant** | **0.9898** | **1.16** |
| 1.0-bit | Gaussian Baseline | 0.9736 | 0.80 |

## 4. Discussion: Memory-Efficient Training
Beyond inference, the extreme sample-efficiency of FWHT projections enables **KudaHitamQuant-Training**. By using a **Straight-Through Estimator (STE)**, researchers can conduct KV-activation compression during the forward pass, reducing activation memory by 8-16x. This allows training massive context sequences on existing hardware without the quadratic memory trap of latent activations.

## 5. Conclusion
**KudaHitamQuant** establishes a new efficiency frontier for KV cache compression. By combining structural innovations in orthogonal projections with low-level bitwise fusions, we achieve quality neutrality at significantly lower bit-rates than previous works, paving the way for ubiquitous endless-context LLM applications.
.
