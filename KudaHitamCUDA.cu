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
    const int N = x.size(0);
    const int D = x.size(1);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N * D + 2047) / 2048); 
    fwht_cuda_forward_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), D, N);
}

__global__ void ultra_fused_hbba_fusion_kernel(
    const half* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids_table, // [D, 16]
    const uint32_t* __restrict__ hbba_mask,    // [8] (256 bits)
    uint8_t* __restrict__ out_idx, 
    float* __restrict__ out_norms, 
    float* __restrict__ out_kmse,
    float* __restrict__ out_r_norms,
    half* __restrict__ out_signs,
    int D, int N) 
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
    if (lane_in_row == 0) out_norms[row_id] = norm;
    
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

    __shared__ half s_centroids[256][16];

    // Parallel load centroids to SRAM
    for (int i = threadIdx.x; i < D * 16; i += threads_per_row) {
        s_centroids[i / 16][i % 16] = __float2half(centroids_table[i]);
    }
    __syncthreads();

    // HBBA Quantize
    float f_scale = 1.0f / sqrtf((float)D);
    #pragma unroll
    for(int k = 0; k < 8; ++k) {
        int element_idx = lane_in_row * 8 + k;
        // Bit-Mask Lookup (0=1bit, 1=4bit)
        int is_4bit = (hbba_mask[element_idx >> 5] >> (element_idx & 31)) & 1;
        int n_centroids = is_4bit ? 16 : 2;
        
        float projected = r[k] * f_scale;
        uint8_t best_c = 0; float min_dist = 1e18f;
        for(int c = 0; c < n_centroids; ++c) {
            float dist = fabsf(projected - __half2float(s_centroids[element_idx][c]));
            if (dist < min_dist) { min_dist = dist; best_c = (uint8_t)c; }
        }
        out_idx[row_id * D + element_idx] = best_c;
        r[k] = __half2float(s_centroids[element_idx][best_c]); 
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

    half* out_half_signs = ((half*)out_signs) + row_id * D + lane_in_row * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_half_signs[i] = __float2half(r[i]);
    }
}

std::vector<torch::Tensor> ultra_fused_hbba_fusion_cuda(
    torch::Tensor x, torch::Tensor d, torch::Tensor centroids_table, torch::Tensor mask) 
{
    const int N = x.size(0);
    const int D = x.size(1);
    
    auto idx_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto norm_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sign_options = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());

    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_kmse = torch::empty({N, D}, norm_options);
    torch::Tensor out_r_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_signs = torch::empty({N, D}, sign_options);

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(32); // row isolation
    dim3 blocks(N); 

    ultra_fused_hbba_fusion_kernel<<<blocks, threads, 0, stream>>>(
        (const half*)x.data_ptr<at::Half>(), 
        d.data_ptr<float>(), 
        centroids_table.data_ptr<float>(),
        (const uint32_t*)mask.data_ptr<int>(),
        out_idx.data_ptr<uint8_t>(), 
        out_norms.data_ptr<float>(), 
        out_kmse.data_ptr<float>(),
        out_r_norms.data_ptr<float>(),
        (half*)out_signs.data_ptr<at::Half>(),
        D, N
    );
    return {out_idx, out_norms, out_kmse, out_r_norms, out_signs};
}

__global__ void hbba_calibrate_cuda_kernel(
    const float* __restrict__ sample, // [N, D]
    float* __restrict__ centroids,   // [D, 16]
    const int* __restrict__ n_map,    // [D]
    int N, int D) 
{
    int d = blockIdx.x;
    if (d >= D) return;
    int n_c = n_map[d];

    __shared__ float s_centroids[16];
    __shared__ float s_sums[16];
    __shared__ int s_counts[16];

    // 1. Initial Min-Max for Uniform Init (Streaming from Global)
    float min_val = 1e18f, max_val = -1e18f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = sample[i * D + d];
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
    }
    
    // Block-wide reduction for min/max
    __shared__ float rs_min[128], rs_max[128];
    rs_min[threadIdx.x] = min_val;
    rs_max[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            rs_min[threadIdx.x] = fminf(rs_min[threadIdx.x], rs_min[threadIdx.x + s]);
            rs_max[threadIdx.x] = fmaxf(rs_max[threadIdx.x], rs_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        float b_min = rs_min[0];
        float b_max = rs_max[0];
        for (int i = 0; i < n_c; ++i) {
            s_centroids[i] = b_min + (float)i * (b_max - b_min) / (float)fmaxf(1.0f, (float)(n_c - 1));
        }
    }
    __syncthreads();

    // 2. Lloyd-Max Iterations (Streaming from Global)
    for (int iter = 0; iter < 10; ++iter) {
        if (threadIdx.x < 16) { s_sums[threadIdx.x] = 0.0f; s_counts[threadIdx.x] = 0; }
        __syncthreads();

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float val = sample[i * D + d];
            int best_c = 0; float min_d = 1e18f;
            for (int c = 0; c < n_c; ++c) {
                float dist = fabsf(val - s_centroids[c]);
                if (dist < min_d) { min_d = dist; best_c = c; }
            }
            atomicAdd(&s_sums[best_c], val);
            atomicAdd(&s_counts[best_c], 1);
        }
        __syncthreads();

        if (threadIdx.x < n_c) {
            if (s_counts[threadIdx.x] > 0) s_centroids[threadIdx.x] = s_sums[threadIdx.x] / (float)s_counts[threadIdx.x];
        }
        __syncthreads();
    }

    // 3. Write back
    if (threadIdx.x < n_c) {
        centroids[d * 16 + threadIdx.x] = s_centroids[threadIdx.x];
    }
}

torch::Tensor hbba_calibrate_cuda(torch::Tensor sample, torch::Tensor n_map) {
    int N = sample.size(0);
    int D = sample.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(sample.device());
    torch::Tensor centroids = torch::zeros({D, 16}, options);

    at::cuda::CUDAGuard device_guard(sample.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(D);
    dim3 threads(128);

    hbba_calibrate_cuda_kernel<<<blocks, threads, 0, stream>>>(
        sample.data_ptr<float>(),
        centroids.data_ptr<float>(),
        n_map.data_ptr<int>(),
        N, D
    );
    return centroids;
}

// --- Symmetry Nexus (V8.8.2) Component ---

// Fast Rank Calculation using __popc
__device__ __forceinline__ int get_rank_v2(const uint32_t* mask, int i, int bit_val) {
    int idx = i >> 5; int pos = i & 31; int rank = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < idx) rank += __popc(bit_val ? mask[j] : ~mask[j]);
        else if (j == idx) rank += __popc((bit_val ? mask[j] : ~mask[j]) & ((1U << pos) - 1));
    }
    return rank;
}

__global__ void packed_hbba_fusion_kernel(
    const half* __restrict__ x,       // [N, D]
    const float* __restrict__ d_vec,   // [D]
    const float* __restrict__ centroids_table, // [D, 16]
    const uint32_t* __restrict__ hbba_mask,    // [8] (256 bits)
    uint8_t* __restrict__ out_packed, // [N, 56]
    float* __restrict__ out_norms,    // [N]
    float* __restrict__ out_kmse,     // [N]
    float* __restrict__ out_r_norms,  // [N]
    half* __restrict__ out_signs,     // [N, D]
    int D, int N) 
{
    int row = blockIdx.x;
    if (row >= N) return;

    __shared__ float s_row[256];
    __shared__ half s_centroids[256][16];
    __shared__ uint8_t s_p[56];
    
    // 1. Parallel Load and FWHT
    for (int i = threadIdx.x; i < D; i += blockDim.x) s_row[i] = __half2float(x[row * D + i]) * d_vec[i];
    if (threadIdx.x < 56) s_p[threadIdx.x] = 0;
    __syncthreads();

    // In-place FWHT
    for (int len = 1; len < D; len <<= 1) {
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            int step = i / len;
            if (step % 2 == 0) {
                float a = s_row[i]; float b = s_row[i + len];
                s_row[i] = a + b; s_row[i + len] = a - b;
            }
        }
        __syncthreads();
    }

    // 2. Load Centroids to SRAM
    for (int i = threadIdx.x; i < D * 16; i += blockDim.x) s_centroids[i / 16][i % 16] = __float2half(centroids_table[i]);
    __syncthreads();

    float norm_val = 0;
    for (int i = 0; i < D; ++i) norm_val += s_row[i] * s_row[i];
    norm_val = sqrtf(norm_val);
    float inv_norm = 1.0f / (norm_val + 1e-8f);

    float kmse = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = s_row[i] * inv_norm;
        int is_4bit = (hbba_mask[i >> 5] >> (i & 31)) & 1;
        int n_c = is_4bit ? 16 : 2;
        
        int best_c = 0; float min_dist = 1e18f;
        for (int c = 0; c < n_c; ++c) {
            float dist = fabsf(val - __half2float(s_centroids[i][c]));
            if (dist < min_dist) { min_dist = dist; best_c = c; }
        }
        
        float res = val - __half2float(s_centroids[i][best_c]);
        out_signs[row * D + i] = __float2half(res);
        kmse += res * res;

        // 3. Grouped Bit-Packing (SRAM) - V8.8.2
        int rank = get_rank_v2(hbba_mask, i, is_4bit);
        if (is_4bit) {
            // Group A (Byte 0-31): 2 per byte
            int byte_idx = rank >> 1;
            int shift = (rank & 1) ? 4 : 0;
            atomicOr((unsigned int*)&s_p[byte_idx & ~3], (unsigned int)((best_c & 0xF) << (shift + (byte_idx & 3) * 8)));
        } else {
            // Group B (Byte 32-55): 8 per byte
            int byte_idx = 32 + (rank >> 3);
            int shift = rank & 7;
            atomicOr((unsigned int*)&s_p[byte_idx & ~3], (unsigned int)((best_c & 0x1) << (shift + (byte_idx & 3) * 8)));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) kmse += __shfl_xor_sync(0xffffffff, kmse, offset);
    
    if (threadIdx.x == 0) {
        out_norms[row] = norm_val;
        out_kmse[row] = kmse / D;
        for(int j=0; j<56; ++j) out_packed[row * 56 + j] = s_p[j];
    }
}

__global__ void asymmetric_score_kernel(
    const half* __restrict__ q,       // [D]
    const uint8_t* __restrict__ k_pc, // [N, 56]
    const float* __restrict__ k_norms,// [N]
    const uint32_t* __restrict__ mask,// [8]
    const float* __restrict__ centroids, // [D, 16]
    float* __restrict__ out_scores,   // [N]
    int D, int N) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float score = 0;
    const uint8_t* my_k = &k_pc[row * 56];
    float norm = k_norms[row];

    for (int i = 0; i < D; ++i) {
        int is_4bit = (mask[i >> 5] >> (i & 31)) & 1;
        int rank = get_rank_v2(mask, i, is_4bit);
        int val = 0;
        if (is_4bit) {
            int byte_idx = rank >> 1;
            val = (my_k[byte_idx] >> ((rank & 1) ? 4 : 0)) & 0xF;
        } else {
            int byte_idx = 32 + (rank >> 3);
            val = (my_k[byte_idx] >> (rank & 7)) & 0x1;
        }
        score += __half2float(q[i]) * centroids[i * 16 + val] * norm;
    }
    out_scores[row] = score;
}

// --- Wrappers ---

std::vector<torch::Tensor> packed_hbba_fusion_cuda(
    torch::Tensor x, torch::Tensor d_vec, torch::Tensor centroids, torch::Tensor mask) {
    int N = x.size(0); int D = x.size(1);
    auto options = torch::TensorOptions().device(x.device());
    auto out_packed = torch::zeros({N, 56}, options.dtype(torch::kUInt8));
    auto out_norms = torch::zeros({N}, options.dtype(torch::kFloat32));
    auto out_kmse = torch::zeros({N}, options.dtype(torch::kFloat32));
    auto out_r_norms = torch::zeros({N}, options.dtype(torch::kFloat32));
    auto out_signs = torch::zeros({N, D}, options.dtype(torch::kHalf));

    at::cuda::CUDAGuard device_guard(x.device());
    packed_hbba_fusion_kernel<<<N, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)x.data_ptr<at::Half>(), d_vec.data_ptr<float>(), centroids.data_ptr<float>(),
        (const uint32_t*)mask.data_ptr<int>(), out_packed.data_ptr<uint8_t>(),
        out_norms.data_ptr<float>(), out_kmse.data_ptr<float>(), out_r_norms.data_ptr<float>(), (half*)out_signs.data_ptr<at::Half>(), D, N
    );
    return {out_packed, out_norms, out_kmse, out_r_norms, out_signs};
}

torch::Tensor asymmetric_score_cuda(
    torch::Tensor q, torch::Tensor k_packed, torch::Tensor k_norms, torch::Tensor mask, torch::Tensor centroids) {
    int N = k_packed.size(0); int D = q.size(0);
    auto out_scores = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    at::cuda::CUDAGuard device_guard(q.device());
    int threads = 256; int blocks = (N + threads - 1) / threads;
    asymmetric_score_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)q.data_ptr<at::Half>(), k_packed.data_ptr<uint8_t>(), k_norms.data_ptr<float>(),
        (const uint32_t*)mask.data_ptr<int>(), centroids.data_ptr<float>(), out_scores.data_ptr<float>(), D, N
    );
    return out_scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam Forward FWHT");
    m.def("ultra_fused_full_fusion", &ultra_fused_full_fusion_cuda, "God Kernel V8.3");
    m.def("ultra_fused_hbba_fusion", &ultra_fused_hbba_fusion_cuda, "HBBA God Kernel V8.5");
    m.def("hbba_calibrate_cuda", &hbba_calibrate_cuda, "Atomic HBBA Calibrator V8.7");
    m.def("packed_hbba_fusion", &packed_hbba_fusion_cuda, "Symmetry Nexus Encoder V8.8");
    m.def("asymmetric_score_cuda", &asymmetric_score_cuda, "Symmetry Nexus Scorer V8.8");
}
