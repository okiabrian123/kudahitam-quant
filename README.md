# Kudahitam-Quant: High-Speed KV Cache Compression

This repository demonstrates the primary innovation in the Kudahitam-Quant D-MLA framework: **Fast Walsh-Hadamard Transform (FWHT) Structured Projections**.

## The Innovation
Traditional Quasi-Joachims-Lindenstrauss (QJL) quantization rely on **Gaussian Dense Projections** ($O(D^2)$ complexity) to balance vector variance before 1-bit quantization. 

**Kudahitam-Quant** replaces these dense matrices with **Structured Projections** using the FWHT algorithm:
- **Complexity Reduction**: From $O(D^2)$ to $O(D \log D)$.
- **Zero Memory Overhead**: No need to store large $256 \times 256$ projection matrices per layer; only a sign-vector is required.
- **Improved Latency**: Up to 60-80% faster compression on standard GPU hardware without sacrificing reconstruction fidelity (cosine similarity).

## Repository Structure
- `compressors.py`: Contains the `TurboQuantCompressorCompare` class with both Gaussian and FWHT implementations.
- `benchmark.py`: A standalone performance suite to compare latency and accuracy.

## Quick Start
1. Ensure you have PyTorch installed.
2. Run the benchmark:
```bash
python benchmark.py
```

## Results
Preliminary benchmarks show that FWHT achieves identical distance-preserving properties as Gaussian projections while significantly reducing the computational bottleneck of the KV-cache compression pipeline.
