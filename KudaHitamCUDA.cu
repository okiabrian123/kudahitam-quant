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
        sum_sq += orig[i*2]*orig[i*2] + orig[i*2 + 1]*orig[i*2 + 1];
    }

    // 2. Norm Reduction - Strict Isolation
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset, threads_per_row);
    }
    float norm = __shfl_sync(0xffffffff, sum_sq > 0 ? sqrtf(sum_sq) : 1e-8f, 0, threads_per_row);
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
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    // 5. Quantize - Orthonormal Scaling
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
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    // 7. Residual & Second Norm - Correct Orthonormal Scaling 
    float b_scale = 1.0f / sqrtf((float)D); 
    float m_base = b_scale * norm;
    float res_sum_sq = 0.0f;
    float* out_kmse_ptr = out_kmse + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float k_mse_val = r[i] * m_base * d_row[i];
        out_kmse_ptr[i] = k_mse_val;
        float diff = orig[i] - k_mse_val;
        r[i] = diff * d_row[i]; 
        res_sum_sq += diff * diff;
    }
    
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        res_sum_sq += __shfl_down_sync(0xffffffff, res_sum_sq, offset, threads_per_row);
    }
    float r_norm = __shfl_sync(0xffffffff, res_sum_sq > 0 ? sqrtf(res_sum_sq) : 1e-8f, 0, threads_per_row);
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
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
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

__global__ void fwht_cuda_forward_kernel(float* __restrict__ x, int D, int N) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = (global_thread_id * 8) / D;
    if (row_id >= N) return;
    int lane_id = threadIdx.x & 31;
    int threads_per_row = D / 8; 
    int lane_in_row = lane_id % threads_per_row;
    float r[8];
    float* row_ptr = x + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = row_ptr[i];
    #pragma unroll
    for(int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for(int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                if ((j & step) == 0) {
                    float a = r[i+j]; float b = r[i+(j|step)];
                    r[i+j] = a + b; r[i+(j|step)] = a - b;
                }
            }
        }
    }
    for(int t_step = 1; t_step < threads_per_row; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }
    float scale = 1.0f / sqrtf((float)D);
    #pragma unroll
    for (int i = 0; i < 8; ++i) row_ptr[i] = r[i] * scale;
}

void fwht_cuda_forward(torch::Tensor x) {
    auto x_float = x.to(torch::kFloat32); // Safe fallback for legacy FWHT kernel
    const int N = x_float.size(0);
    const int D = x_float.size(1);
    at::cuda::CUDAGuard device_guard(x_float.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N * D + 2047) / 2048); 
    fwht_cuda_forward_kernel<<<blocks, threads, 0, stream>>>(x_float.data_ptr<float>(), D, N);
    
    // Copy back result if it was half (since it's in-place)
    if (x.scalar_type() == torch::kHalf) {
        x.copy_(x_float);
    }
}

__global__ void ultra_fused_hbba_fusion_kernel(
    const half* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids_table, // [D, 16]
    const int* __restrict__ n_centroids_map,   // [D]
    uint8_t* __restrict__ out_idx, 
    half* __restrict__ out_norms, 
    half* __restrict__ out_kmse,
    half* __restrict__ out_r_norms,
    int8_t* __restrict__ out_signs,
    int D, int N) 
{
    // Innovations V8.7: Shared Centroid Cache (LDS)
    __shared__ float s_centroids[256 * 16]; // 16KB for D=256
    
    // Parallel Load Table into SRAM
    int total_elements = D * 16;
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        if (i < 4096) s_centroids[i] = centroids_table[i];
    }
    __syncthreads();

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = (global_thread_id * 8) / D;
    if (row_id >= N) return;

    int lane_id = threadIdx.x & 31;
    int threads_per_row = D / 8; 
    int lane_in_row = lane_id % threads_per_row;

    float r[8];
    float orig[8];
    float sum_sq = 0.0f;
    
    const half2* x_ptr = reinterpret_cast<const half2*>(x) + row_id * (D/2) + lane_in_row * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        half2 val2 = x_ptr[i];
        orig[i*2] = __half2float(val2.x);
        orig[i*2 + 1] = __half2float(val2.y);
        sum_sq += orig[i*2]*orig[i*2] + orig[i*2 + 1]*orig[i*2 + 1];
    }

    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset, threads_per_row);
    }
    float norm = __shfl_sync(0xffffffff, sum_sq > 0 ? sqrtf(sum_sq) : 1e-8f, 0, threads_per_row);
    if (lane_in_row == 0) out_norms[row_id] = __float2half(norm);
    
    const float* d_row = d + lane_in_row * 8;
    #pragma unroll
    for(int k=0; k<8; ++k) r[k] = (orig[k] * d_row[k]) / norm;

    // Pass 1: FWHT
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
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    // HBBA Quantize
    float f_scale = 1.0f / sqrtf((float)D);
    #pragma unroll
    for(int k = 0; k < 8; ++k) {
        int element_idx = lane_in_row * 8 + k;
        int n_centroids = n_centroids_map[element_idx];
        const float* centroids = s_centroids + element_idx * 16;
        
        float projected = r[k] * f_scale;
        uint8_t best_c = 0; float min_dist = 1e18f;
        for(int c = 0; c < n_centroids; ++c) {
            float dist = fabsf(projected - centroids[c]);
            if (dist < min_dist) { min_dist = dist; best_c = (uint8_t)c; }
        }
        out_idx[row_id * D + element_idx] = best_c;
        r[k] = centroids[best_c]; 
    }

    // Pass 2: FWHT (Reconstruct)
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
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    float b_scale = 1.0f / sqrtf((float)D); 
    float m_base = b_scale * norm;
    float res_sum_sq = 0.0f;
    half* out_kmse_ptr = out_kmse + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float k_mse_val = r[i] * m_base * d_row[i];
        out_kmse_ptr[i] = __float2half(k_mse_val);
        float diff = orig[i] - k_mse_val;
        r[i] = diff * d_row[i]; 
        res_sum_sq += diff * diff;
    }
    
    for (int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        res_sum_sq += __shfl_down_sync(0xffffffff, res_sum_sq, offset, threads_per_row);
    }
    float r_norm = __shfl_sync(0xffffffff, res_sum_sq > 0 ? sqrtf(res_sum_sq) : 1e-8f, 0, threads_per_row);
    if (lane_in_row == 0) out_r_norms[row_id] = __float2half(r_norm);

    // Pass 3: FWHT (Signs)
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
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step, threads_per_row);
            if ((lane_in_row & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    uint8_t* out_packed_signs = (uint8_t*)out_signs + row_id * (D/8) + lane_in_row;
    uint8_t pack = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (r[i] >= 0) pack |= (1 << i);
    }
    *out_packed_signs = pack;
}

std::vector<torch::Tensor> ultra_fused_hbba_fusion_cuda(
    torch::Tensor x, torch::Tensor d, torch::Tensor centroids_table, torch::Tensor n_centroids_map) 
{
    const int N = x.size(0);
    const int D = x.size(1);
    
    auto idx_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto norm_options = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
    auto sign_options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());

    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_kmse = torch::empty({N, D}, norm_options);
    torch::Tensor out_r_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_signs = torch::empty({N, D / 8}, sign_options);

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(32); // row isolation
    dim3 blocks(N); 

    ultra_fused_hbba_fusion_kernel<<<blocks, threads, 0, stream>>>(
        (const half*)x.data_ptr<at::Half>(), 
        d.data_ptr<float>(), 
        centroids_table.data_ptr<float>(),
        n_centroids_map.data_ptr<int>(),
        out_idx.data_ptr<uint8_t>(), 
        (half*)out_norms.data_ptr<at::Half>(), 
        (half*)out_kmse.data_ptr<at::Half>(),
        (half*)out_r_norms.data_ptr<at::Half>(),
        (int8_t*)out_signs.data_ptr<int8_t>(),
        D, N
    );
    return {out_idx, out_norms, out_kmse, out_r_norms, out_signs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam Forward FWHT (Orthonormal)");
    m.def("ultra_fused_full_fusion", &ultra_fused_full_fusion_cuda, "KudaHitam God Kernel V8.3 (Complete)");
    m.def("ultra_fused_hbba_fusion", &ultra_fused_hbba_fusion_cuda, "KudaHitam HBBA God Kernel V8.5 (Hybrid 1/4-bit)");
}
