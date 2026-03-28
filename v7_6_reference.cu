#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode Ultra-Fused
 * Everything in ONE PASS: Norm -> Scale -> FWHT -> Quant
 */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void fwht_warp(float& val, int mask) {
    for (int step = 1; step <= mask; step <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, step);
        if ((threadIdx.x & step) == 0) val = val + other;
        else val = other - val;
    }
}

__global__ void fwht_kernel_legacy(float* __restrict__ x, int D, int N) {
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;
    int tid = threadIdx.x;
    if (tid >= D) return;
    
    __shared__ float sdata[1024];
    float val = x[row_idx * D + tid];
    
    fwht_warp(val, 16);
    sdata[tid] = val; __syncthreads();
    
    for (int step = 32; step < D; step <<= 1) {
        float other = sdata[tid ^ step];
        if ((tid & step) == 0) val = val + other;
        else val = other - val;
        __syncthreads();
        sdata[tid] = val; __syncthreads();
    }
    
    float scale = 1.0f / sqrtf((float)D);
    x[row_idx * D + tid] = val * scale;
}

__global__ void ultra_fused_compress_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids, 
    uint8_t* __restrict__ out_idx, 
    float* __restrict__ out_norms, 
    int D, int N, int n_centroids) 
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    float r[8];
    float sum_sq = 0.0f;
    
    // Vectorized Load (float4) & Partial Norm
    float4 load0 = reinterpret_cast<const float4*>(x)[global_warp_id * 64 + lane_id];
    float4 load1 = reinterpret_cast<const float4*>(x)[global_warp_id * 64 + lane_id + 32];
    r[0] = load0.x; r[1] = load0.y; r[2] = load0.z; r[3] = load0.w;
    r[4] = load1.x; r[5] = load1.y; r[6] = load1.z; r[7] = load1.w;

    #pragma unroll
    for(int k=0; k<8; ++k) sum_sq += r[k]*r[k];
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    float norm = __shfl_sync(0xffffffff, sum_sq > 0 ? sqrtf(sum_sq) : 1e-8f, 0);
    if (lane_id == 0) out_norms[global_warp_id] = norm;
    
    // Scale by d and norm
    float4 d0 = reinterpret_cast<const float4*>(d)[lane_id];
    float4 d1 = reinterpret_cast<const float4*>(d)[lane_id + 32];
    
    r[0] = (r[0] * d0.x) / norm; r[1] = (r[1] * d0.y) / norm; 
    r[2] = (r[2] * d0.z) / norm; r[3] = (r[3] * d0.w) / norm;
    r[4] = (r[4] * d1.x) / norm; r[5] = (r[5] * d1.y) / norm; 
    r[6] = (r[6] * d1.z) / norm; r[7] = (r[7] * d1.w) / norm;
    
    // Intra-thread FWHT (Steps 1, 2)
    #pragma unroll
    for(int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for(int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                if ((j & step) == 0) {
                    float a = r[i + j];
                    float b = r[i + (j | step)];
                    r[i + j] = a + b;
                    r[i + (j | step)] = a - b;
                }
            }
        }
    }
    
    // Inter-thread FWHT (Steps 4, 8, 16, 32, 64) -> t_step = 1, 2, 4, 8, 16 
    #pragma unroll
    for(int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    
    // Intra-thread FWHT (Step 128)
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j];
        float b = r[j + 4];
        r[j] = a + b;
        r[j + 4] = a - b;
    }
    
    // Quantization
    float scale = 1.0f / 16.0f;
    uint8_t out_c[8];
    #pragma unroll
    for(int k = 0; k < 8; ++k) {
        float projected = r[k] * scale;
        uint8_t best_c = 0;
        float min_dist = 1e18f;
        for(int c = 0; c < n_centroids; ++c) {
            float dist = fabsf(projected - centroids[c]);
            if (dist < min_dist) { min_dist = dist; best_c = (uint8_t)c; }
        }
        out_c[k] = best_c;
    }
    
    // Vectorized Write
    union { uint8_t b[4]; uint32_t v; } pack0, pack1;
    pack0.b[0] = out_c[0]; pack0.b[1] = out_c[1]; pack0.b[2] = out_c[2]; pack0.b[3] = out_c[3];
    pack1.b[0] = out_c[4]; pack1.b[1] = out_c[5]; pack1.b[2] = out_c[6]; pack1.b[3] = out_c[7];
    
    reinterpret_cast<uint32_t*>(out_idx)[global_warp_id * 64 + lane_id] = pack0.v;
    reinterpret_cast<uint32_t*>(out_idx)[global_warp_id * 64 + lane_id + 32] = pack1.v;
}

std::vector<torch::Tensor> ultra_fused_compress_cuda(
    torch::Tensor x, torch::Tensor d, torch::Tensor centroids) 
{
    const int N = x.size(0);
    const int D = x.size(1);
    if (D != 256) throw std::runtime_error("Warp-Only FWHT requires D=256");
    const int n_centroids = centroids.size(0);
    auto idx_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto norm_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N + 7) / 8);
    ultra_fused_compress_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), d.data_ptr<float>(), centroids.data_ptr<float>(), 
        out_idx.data_ptr<uint8_t>(), out_norms.data_ptr<float>(), 
        D, N, n_centroids
    );
    return {out_idx, out_norms};
}

__global__ void ultra_fused_reconstruct_kernel(
    const uint8_t* __restrict__ indices,
    const float* __restrict__ vec_norms,
    const float* __restrict__ centroids,
    const float* __restrict__ d,
    float* __restrict__ out_kmse,
    int D, int N)
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    union { uint8_t b[4]; uint32_t v; } pack0, pack1;
    pack0.v = reinterpret_cast<const uint32_t*>(indices)[global_warp_id * 64 + lane_id];
    pack1.v = reinterpret_cast<const uint32_t*>(indices)[global_warp_id * 64 + lane_id + 32];
    
    float r[8];
    r[0] = centroids[pack0.b[0]]; r[1] = centroids[pack0.b[1]]; 
    r[2] = centroids[pack0.b[2]]; r[3] = centroids[pack0.b[3]];
    r[4] = centroids[pack1.b[0]]; r[5] = centroids[pack1.b[1]]; 
    r[6] = centroids[pack1.b[2]]; r[7] = centroids[pack1.b[3]];

    // Step 1, 2
    #pragma unroll
    for(int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for(int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                if ((j & step) == 0) {
                    float a = r[i + j];
                    float b = r[i + (j | step)];
                    r[i + j] = a + b;
                    r[i + (j | step)] = a - b;
                }
            }
        }
    }
    
    // Step 4..64
    #pragma unroll
    for(int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    
    // Step 128
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j];
        float b = r[j + 4];
        r[j] = a + b;
        r[j + 4] = a - b;
    }
    
    float scale = 1.0f / 16.0f;
    float norm = vec_norms[global_warp_id];
    float4 d0 = reinterpret_cast<const float4*>(d)[lane_id];
    float4 d1 = reinterpret_cast<const float4*>(d)[lane_id + 32];
    
    float m0 = scale * norm * d0.x; float m1 = scale * norm * d0.y;
    float m2 = scale * norm * d0.z; float m3 = scale * norm * d0.w;
    float m4 = scale * norm * d1.x; float m5 = scale * norm * d1.y;
    float m6 = scale * norm * d1.z; float m7 = scale * norm * d1.w;
    
    r[0] *= m0; r[1] *= m1; r[2] *= m2; r[3] *= m3;
    r[4] *= m4; r[5] *= m5; r[6] *= m6; r[7] *= m7;
    
    // Vectorized Write
    float4 out0 = {r[0], r[1], r[2], r[3]};
    float4 out1 = {r[4], r[5], r[6], r[7]};
    reinterpret_cast<float4*>(out_kmse)[global_warp_id * 64 + lane_id] = out0;
    reinterpret_cast<float4*>(out_kmse)[global_warp_id * 64 + lane_id + 32] = out1;
}

torch::Tensor ultra_fused_reconstruct_cuda(
    torch::Tensor indices, torch::Tensor vec_norms, torch::Tensor centroids, torch::Tensor d) 
{
    const int N = indices.size(0);
    const int D = indices.size(1);
    if (D != 256) throw std::runtime_error("Warp-Only FWHT requires D=256");
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(indices.device());
    torch::Tensor out_kmse = torch::empty({N, D}, options);
    at::cuda::CUDAGuard device_guard(indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N + 7) / 8);
    ultra_fused_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
        indices.data_ptr<uint8_t>(), vec_norms.data_ptr<float>(), 
        centroids.data_ptr<float>(), d.data_ptr<float>(), 
        out_kmse.data_ptr<float>(), D, N
    );
    return out_kmse;
}


void fwht_cuda_forward(torch::Tensor x) {
    const int N = x.size(0);
    const int D = x.size(1);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(D);
    dim3 blocks(N);
    fwht_kernel_legacy<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), D, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam FWHT Forward (Legacy)");
    m.def("ultra_fused_compress", &ultra_fused_compress_cuda, "KudaHitam Ultra-Fused Compression");
    m.def("ultra_fused_reconstruct", &ultra_fused_reconstruct_cuda, "KudaHitam Ultra-Fused Reconstruction");
}
