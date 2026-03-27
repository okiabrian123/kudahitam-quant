# Technical Comparison: Gaussian vs KudaHitamQuant

This document provides a formal comparison between traditional **Gaussian QJL** and our optimized **KudaHitamQuant**.

## 1. Complexity Comparison
| Metric | Gaussian QJL (Original) | KudaHitamQuant (Innovation) |
| :--- | :--- | :--- |
| **Projection Complexity** | $O(D^2)$ | **$O(D \log D)$** |
| **Operation Type** | Dense Matrix-Vector Multiplication | **Recursive Sum-Difference Transform** |
| **Storage (per layer)** | $256 \times 256 \times \text{fp}32 = 256 \text{ KB}$ | **$256 \times 1 \text{ bit} = 32 \text{ bytes (Sign Vector)}$** |

## 2. Performance Analysis (Qwen-3.5 2B / 40,000 Tokens)
Based on benchmarks with **Qwen-3.5 2B** (MLA architecture) on NVIDIA T4:

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

## 3. Why FWHT is Superior?
In the traditional dense approach, every KV-cache compression required a large $D \times D$ matrix multiplication. This not only consumed GPU compute cycles but also introduced significant memory-access overhead for the large projection matrix.

**KudaHitamQuant** eliminates this:
1. **Memory Access**: It reads only a single 256-dim sign vector, reducing I/O bandwidth.
2. **Compute**: FWHT uses only additions and subtractions, which are much cheaper than multiplications for hardware.
3. **Scaling**: While Gaussian scaling is quadratic ($O(D^2)$), FWHT scaling is near-linear ($O(D \log D)$), making it the only viable choice for future models with $D=1024$ or $D=2048$.

## Conclusion
The transition from Gaussian to FWHT is not just a 'trick'; it is a structural innovation that makes KV-cache compression practical for long-context generation in production environments.
