#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode V8.3 (Final Handover)
 * Monolithic: FP16 Input -> Registers -> Quant -> Reconstruction -> FP32 Output
 * [V8.0] Fused Indices, Norms, k_mse, Signs, and Residual Norms.
 * [V8.1] Fixed Scaling parity with Triton hardcoded 1/16.
 * [V8.2] Sign-bit accuracy correction for D=128 (D/256 factor).
 * [V8.3] Professional annotation and final stabilization.
 */

__global__ void ultra_fused_full_fusion_kernel_v83(
    const half* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids, 
    uint8_t* __restrict__ out_idx, 
    float* __restrict__ out_norms, 
    float* __restrict__ out_kmse,
    int8_t* __restrict__ out_signs,
    float* __restrict__ out_r_norms,
    int D, int N, int n_centroids) 
{
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = (global_thread_id * 8) / D;
    if (row_id >= N) return;

    int lane_id = threadIdx.x & 31;
    int threads_per_row = D / 8; 
    int row_start_lane = (lane_id / threads_per_row) * threads_per_row;
    int lane_in_row = lane_id % threads_per_row;

    float r[8];
    float sum_sq = 0.0f;
    
    // 1. Vectorized FP16 Load
    const half2* x_ptr = reinterpret_cast<const half2*>(x) + row_id * (D/2) + lane_in_row * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        half2 val2 = x_ptr[i];
        r[i*2] = __half2float(val2.x);
        r[i*2 + 1] = __half2float(val2.y);
        sum_sq += r[i*2]*r[i*2] + r[i*2+1]*r[i*2+1];
    }

    // 2. Isolated Norm Reduction
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    float norm = __shfl_sync(0xffffffff, sum_sq > 0 ? sqrtf(sum_sq) : 1e-8f, row_start_lane);
    if (lane_in_row == 0) out_norms[row_id] = norm;
    
    // 3. Forward Projection
    const float* d_row = d + lane_in_row * 8;
    #pragma unroll
    for(int k=0; k<8; ++k) r[k] = (r[k] * d_row[k]) / norm;

    // 4. Butterfly Transform
    #pragma unroll
    for(int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for(int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                if ((j & step) == 0) {
                    float a = r[i + j]; float b = r[i + (j | step)];
                    r[i + j] = a + b; r[i + (j | step)] = a - b;
                }
            }
        }
    }
    for(int t_step = 1; t_step < threads_per_row; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) { float a = r[j]; float b = r[j + 4]; r[j] = a + b; r[j + 4] = a - b; }

    // 5. Quantize & Sign Extraction [V8.3]
    float f_scale = 1.0f / 16.0f; // [V8.1] Aligned to Triton
    float s_factor = (float)D / 256.0f; // [V8.2] Sign Correction Factor
    uint8_t out_c[8];
    float sum_sq_resid = 0.0f;
    #pragma unroll
    for(int k = 0; k < 8; ++k) {
        float projected = r[k] * f_scale;
        uint8_t best_c = 0; float min_dist = 1e18f;
        for(int c = 0; c < n_centroids; ++c) {
            float dist = fabsf(projected - centroids[c]);
            if (dist < min_dist) { min_dist = dist; best_c = (uint8_t)c; }
        }
        out_c[k] = best_c;
        float resid = projected - s_factor * centroids[best_c]; // [V8.3] Sign bit logic
        sum_sq_resid += (projected - centroids[best_c]) * (projected - centroids[best_c]);
        if (out_signs != nullptr) out_signs[row_id * D + lane_in_row * 8 + k] = (resid >= 0) ? 1 : -1;
        r[k] = centroids[best_c]; 
    }

    // 6. Isolated Residual Norm Reduction
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum_sq_resid += __shfl_down_sync(0xffffffff, sum_sq_resid, offset);
    }
    float r_norm = __shfl_sync(0xffffffff, sqrtf(sum_sq_resid), row_start_lane);
    if (lane_in_row == 0) out_r_norms[row_id] = r_norm;
    
    if (out_idx != nullptr) {
        uint32_t* out_idx_ptr = reinterpret_cast<uint32_t*>(out_idx) + row_id * (D/4) + lane_in_row * 2;
        union { uint8_t b[4]; uint32_t v; } pack0, pack1;
        pack0.b[0] = out_c[0]; pack0.b[1] = out_c[1]; pack0.b[2] = out_c[2]; pack0.b[3] = out_c[3];
        pack1.b[0] = out_c[4]; pack1.b[1] = out_c[5]; pack1.b[2] = out_c[6]; pack1.b[3] = out_c[7];
        out_idx_ptr[0] = pack0.v; out_idx_ptr[1] = pack1.v;
    }

    // 7. Inverse Butterfly
    #pragma unroll
    for(int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for(int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                if ((j & step) == 0) {
                    float a = r[i + j]; float b = r[i + (j | step)];
                    r[i + j] = a + b; r[i + (j | step)] = a - b;
                }
            }
        }
    }
    for(int t_step = 1; t_step < threads_per_row; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) { float a = r[j]; float b = r[j + 4]; r[j] = a + b; r[j + 4] = a - b; }

    // 8. Reconstruct & Write FP32
    float m_base = (1.0f / 16.0f) * norm; 
    float* out_ptr = out_kmse + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_ptr[i] = r[i] * m_base * d_row[i];
}

std::vector<torch::Tensor> ultra_fused_full_fusion_cuda(
    torch::Tensor x, torch::Tensor d, torch::Tensor centroids) 
{
    const int N = x.size(0);
    const int D = x.size(1);
    const int n_centroids = centroids.size(0);
    
    auto idx_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto norm_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sign_options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());

    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_kmse = torch::empty({N, D}, norm_options);
    torch::Tensor out_signs = torch::empty({N, D}, sign_options);
    torch::Tensor out_r_norms = torch::empty({N, 1}, norm_options);

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N * D + 2047) / 2048);

    ultra_fused_full_fusion_kernel_v83<<<blocks, threads, 0, stream>>>(
        (const half*)x.data_ptr<at::Half>(), 
        d.data_ptr<float>(), 
        centroids.data_ptr<float>(), 
        out_idx.data_ptr<uint8_t>(), 
        out_norms.data_ptr<float>(), 
        out_kmse.data_ptr<float>(),
        out_signs.data_ptr<int8_t>(),
        out_r_norms.data_ptr<float>(),
        D, N, n_centroids
    );
    return {out_idx, out_norms, out_kmse, out_signs, out_r_norms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ultra_fused_full_fusion", &ultra_fused_full_fusion_cuda, "KudaHitam Engine V8.3 [FULL FUSION]");
}
