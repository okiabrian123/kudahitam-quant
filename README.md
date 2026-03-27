# KudaHitamQuant: Ultra-High Fidelity 1-bit KV Compression for Qwen-3.5 MLA

**KudaHitamQuant** is a high-performance KV-cache compression framework specifically designed for the **Multi-Head Latent Attention (MLA)** architecture of **Qwen-3.5 2B**. Unlike Qwen-2, Qwen-3.5 introduces significant architectural improvements in latent representation, making extreme 1-bit compression both more challenging and more rewarding.

This repository implements **Fast Walsh-Hadamard Transform (FWHT) Structured Projections** to achieve near-lossless fidelity (0.99+) at extreme bit-rates.

## Key Features

- **Structured Projections (FWHT)**: Reduces projection complexity from $O(D^2)$ to $O(D \log D)$, eliminating the dense bottleneck in KV-cache compression.
- **Qwen-3.5 2B MLA Focus**: Tailored for the latent dimension ($D=256$) of the latest Qwen models, providing a 10x+ improvement in memory overhead compared to uncompressed FP16 cache.
- **Bitwise-Optimized Triton Kernels**: All thread-indexing logic (modulo, division) has been replaced with bitwise shifts and masks (`<<`, `>>`, `&`), ensuring zero-latency coordinate mapping in CUDA.
- **High Fidelity**: Achieves **0.9991** fidelity (Quality Neutral) at 3.0-bit and **0.9898** at 1.0-bit for 40,000+ token context lengths.

## Detailed Benchmark Results (Qwen-3.5 2B MLA)

The following evaluations were performed on an NVIDIA T4 and CPU offloading using the **Qwen-3.5 2B** model ($D=256$, H=16, 40k Context).

| Context | Bits | Acc (F) | Acc (G) | Comp(F) ms | Comp(G) ms |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 10,000 | 1-bit QJL | 0.9941 | 0.9852 | 1.12 | 0.79 |
| 10,000 | 2-bit QJL | 0.9981 | 0.9949 | 1.05 | 0.78 |
| 10,000 | 3-bit QJL | 0.9994 | 0.9985 | 1.07 | 0.81 |
| 40,000 | 1-bit QJL | 0.9905 | 0.9752 | 1.19 | 0.83 |
| 40,000 | 2-bit QJL | 0.9969 | 0.9915 | 1.07 | 0.85 |
| 40,000 | 3-bit QJL | 0.9991 | 0.9975 | 1.16 | 0.81 |
| 104,000 | 1-bit QJL | 0.9890 | 0.9720 | 1.19 | 0.83 |
| 104,000 | 2-bit QJL | 0.9964 | 0.9903 | 1.12 | 0.82 |
| 104,000 | 3-bit QJL | 0.9989 | 0.9971 | 1.14 | 0.81 |

Note: **3.0-bit** reaches "Quality Neutrality" (0.999+ fidelity).

## Repository Structure

- `KudaHitamQuant_full.py`: The core engine containing the Triton kernels and the Qwen-3.5 benchmark suite.
- `technical_comparison.md`: Detailed mathematical breakdown of FWHT vs Gaussian projections.
- `KudaHitamQuant_paper_draft.md`: Full research manuscript draft.
- `KudaHitamQuant_paper.tex`: LaTeX source for arXiv submission.

## Quick Start
1. Ensure you have PyTorch and Triton installed.
2. Run the benchmark:
```bash
python KudaHitamQuant_full.py
```
