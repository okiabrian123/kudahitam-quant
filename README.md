# Kudahitam-Quant: Ultra-High Fidelity 1-bit KV Compression for Qwen-3.5 MLA

**Kudahitam-Quant** is a high-performance KV-cache compression framework specifically designed for the **Multi-Head Latent Attention (MLA)** architecture of **Qwen-3.5 2B**. Unlike Qwen-2, Qwen-3.5 introduces significant architectural improvements in latent representation, making extreme 1-bit compression both more challenging and more rewarding.

This repository implements **Fast Walsh-Hadamard Transform (FWHT) Structured Projections** to achieve near-lossless fidelity (0.99+) at extreme bit-rates.

## Key Features

- **Structured Projections (FWHT)**: Reduces projection complexity from $O(D^2)$ to $O(D \log D)$, eliminating the dense bottleneck in KV-cache compression.
- **Qwen-3.5 2B MLA Focus**: Tailored for the latent dimension ($D=256$) of the latest Qwen models, providing a 10x+ improvement in memory overhead compared to uncompressed FP16 cache.
- **Bitwise-Optimized Triton Kernels**: All thread-indexing logic (modulo, division) has been replaced with bitwise shifts and masks (`<<`, `>>`, `&`), ensuring zero-latency coordinate mapping in CUDA.
- **High Fidelity**: Achieves **0.9991** fidelity (Quality Neutral) at 3.0-bit and **0.9898** at 1.0-bit for 40,000+ token context lengths.

## Detailed Benchmark Results (Qwen-3.5 2B MLA)

The following evaluations were performed on an NVIDIA H100 GPU using the **Qwen-3.5 2B** model ($D=256$, H=16, 40k Context).

| Strategy | Bit-rate | Acc (Fidelity) | Latency (ms/layer) |
| :--- | :---: | :---: | :---: |
| Google TurboQuant (Neutral) | 3.5 | ~1.0000 | 2.10 |
| **Kudahitam (Neutral)** | **3.0** | **0.9991** | **1.10** |
| **Kudahitam-Quant** | **2.0** | **0.9967** | **1.09** |
| **Kudahitam-Quant** | **1.0** | **0.9898** | **1.16** |
| Gaussian 1-bit QJL | 1.0 | 0.9736 | 0.80 |

Note: **3.0-bit** reaches "Quality Neutrality" (0.999+ fidelity).

## Repository Structure

- `kudahitam_qwen3_5_full.py`: The core engine containing the Triton kernels and the Qwen-3.5 benchmark suite.
- `technical_comparison.md`: Detailed mathematical breakdown of FWHT vs Gaussian projections.
- `kudahitam_quant_paper_draft.md`: Full research manuscript draft.
- `kudahitam_quant_paper.tex`: LaTeX source for arXiv submission.

## Quick Start
1. Ensure you have PyTorch and Triton installed.
2. Run the benchmark:
```bash
python kudahitam_qwen3_5_full.py
```
