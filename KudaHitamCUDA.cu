#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode V7.6 (Native FP16 Engine)
 * Monolithic: FP16 Input -> Registers -> Quant -> Reconstruction -> FP16 Output
 * Formula: Out = FWHT(LloydMax(FWHT(X_norm * D), centroids)) * D * ||X||
 * Bypasses Python-level FP16->FP32 casting floor.
 */

__global__ void ultra_fused_full_fusion_kernel_v8(
    const half* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids, 
    uint8_t* __restrict__ out_idx, 
    float* __restrict__ out_norms, 
    float* __restrict__ out_kmse,
    float* __restrict__ out_r_norms,
    int8_t* __restrict__ out_signs,
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
    float orig[8];
    float sum_sq = 0.0f;
    
    // 1. Vectorized FP16 Load
    const half2* x_ptr = reinterpret_cast<const half2*>(x) + row_id * (D/2) + lane_in_row * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        half2 val2 = x_ptr[i];
        orig[i*2] = __half2float(val2.x);
        orig[i*2 + 1] = __half2float(val2.y);
        sum_sq += orig[i*2]*orig[i*2] + orig[i*2+1]*orig[i*2+1];
    }

    // 2. Norm Reduction
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    float norm = __shfl_sync(0xffffffff, sum_sq > 0 ? sqrtf(sum_sq) : 1e-8f, row_start_lane);
    if (lane_in_row == 0) out_norms[row_id] = norm;
    
    // 3. Forward Projection Prep
    const float* d_row = d + lane_in_row * 8;
    #pragma unroll
    for(int k=0; k<8; ++k) r[k] = (orig[k] * d_row[k]) / norm;

    // 4. Pass 1: FWHT (Forward)
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
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    // 5. Quantize
    float f_scale = 1.0f / sqrtf((float)D); 
    uint8_t out_c[8];
    #pragma unroll
    for(int k = 0; k < 8; ++k) {
        float projected = r[k] * f_scale;
        uint8_t best_c = 0; float min_dist = 1e18f;
        for(int c = 0; c < n_centroids; ++c) {
            float dist = fabsf(projected - centroids[c]);
            if (dist < min_dist) { min_dist = dist; best_c = (uint8_t)c; }
        }
        out_c[k] = best_c;
        r[k] = centroids[best_c]; 
    }
    
    if (out_idx != nullptr) {
        uint32_t* out_idx_ptr = reinterpret_cast<uint32_t*>(out_idx) + row_id * (D/4) + lane_in_row * 2;
        union { uint8_t b[4]; uint32_t v; } pack0, pack1;
        pack0.b[0] = out_c[0]; pack0.b[1] = out_c[1]; pack0.b[2] = out_c[2]; pack0.b[3] = out_c[3];
        pack1.b[0] = out_c[4]; pack1.b[1] = out_c[5]; pack1.b[2] = out_c[6]; pack1.b[3] = out_c[7];
        out_idx_ptr[0] = pack0.v; out_idx_ptr[1] = pack1.v;
    }

    // 6. Pass 2: FWHT (Inverse/Reconstruct)
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
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    // 7. Residual & Second Norm
    float b_scale = 1.0f / sqrtf((float)D); 
    float m_base = b_scale * norm;
    float res_sum_sq = 0.0f;
    float* out_kmse_ptr = out_kmse + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float k_mse_val = r[i] * m_base * d_row[i];
        out_kmse_ptr[i] = k_mse_val;
        float diff = orig[i] - k_mse_val;
        r[i] = diff * d_row[i]; // Prep for Pass 3 (Signs)
        res_sum_sq += diff * diff;
    }
    
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        res_sum_sq += __shfl_down_sync(0xffffffff, res_sum_sq, offset);
    }
    float r_norm = __shfl_sync(0xffffffff, res_sum_sq > 0 ? sqrtf(res_sum_sq) : 1e-8f, row_start_lane);
    if (lane_in_row == 0) out_r_norms[row_id] = r_norm;

    // 8. Pass 3: FWHT (Signs Projection)
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
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    // 9. Final Sign Write
    int8_t* out_signs_ptr = out_signs + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_signs_ptr[i] = (r[i] >= 0) ? (int8_t)1 : (int8_t)-1;
    }
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
    torch::Tensor out_r_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_signs = torch::empty({N, D}, sign_options);

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N * D + 2047) / 2048); 

    ultra_fused_full_fusion_kernel_v8<<<blocks, threads, 0, stream>>>(
        (const half*)x.data_ptr<at::Half>(), 
        d.data_ptr<float>(), 
        centroids.data_ptr<float>(), 
        out_idx.data_ptr<uint8_t>(), 
        out_norms.data_ptr<float>(), 
        out_kmse.data_ptr<float>(),
        out_r_norms.data_ptr<float>(),
        out_signs.data_ptr<int8_t>(),
        D, N, n_centroids
    );
    return {out_idx, out_norms, out_kmse, out_r_norms, out_signs};
}

void fwht_cuda_forward_warp(torch::Tensor x) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward_warp, "Reserved");
    m.def("ultra_fused_full_fusion", &ultra_fused_full_fusion_cuda, "KudaHitam God Kernel V8.0");
}
