# Technical Comparison: Gaussian vs Kudahitam-Quant

This document provides a formal comparison between traditional **Gaussian QJL** and our optimized **Kudahitam-Quant**.

## 1. Complexity Comparison
| Metric | Gaussian QJL (Original) | Kudahitam-Quant (Innovation) |
| :--- | :--- | :--- |
| **Projection Complexity** | $O(D^2)$ | **$O(D \log D)$** |
| **Operation Type** | Dense Matrix-Vector Multiplication | **Recursive Sum-Difference Transform** |
| **Storage (per layer)** | $256 \times 256 \times \text{fp}32 = 256 \text{ KB}$ | **$256 \times 1 \text{ bit} = 32 \text{ bytes (Sign Vector)}$** |

## 2. Performance Analysis (Qwen-3.5 2B / 40,000 Tokens)
Based on benchmarks with **Qwen-3.5 2B** (MLA architecture) on NVIDIA H100:

| Strategy | Latency (ms/layer) | Cosine Similarity (Fidelity) | Accuracy Status |
| :--- | :---: | :---: | :---: |
| **FP16 (Baseline)** | 15.52 ms | 1.0000 | Baseline |
| **Gaussian 1-bit** | **0.80 ms** | 0.9736 | Noisy |
| **Kudahitam 1-bit** | **1.16 ms** | **0.9898** | **High Fidelity** |
| **Kudahitam 3-bit** | **1.10 ms** | **0.9991** | **Quality Neutral** |

## 3. Why FWHT is Superior?
In the traditional dense approach, every KV-cache compression required a large $D \times D$ matrix multiplication. This not only consumed GPU compute cycles but also introduced significant memory-access overhead for the large projection matrix.

**Kudahitam-Quant** eliminates this:
1. **Memory Access**: It reads only a single 256-dim sign vector, reducing I/O bandwidth.
2. **Compute**: FWHT uses only additions and subtractions, which are much cheaper than multiplications for hardware.
3. **Scaling**: While Gaussian scaling is quadratic ($O(D^2)$), FWHT scaling is near-linear ($O(D \log D)$), making it the only viable choice for future models with $D=1024$ or $D=2048$.

## Conclusion
The transition from Gaussian to FWHT is not just a 'trick'; it is a structural innovation that makes KV-cache compression practical for long-context generation in production environments.
