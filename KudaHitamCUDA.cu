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
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;
    int tid = threadIdx.x;
    
    __shared__ float sdata[1024]; 
    
    // 1. Load, Scale by d, and compute partial sum of squares
    float x_val = x[row_idx * D + tid];
    float d_val = d[tid];
    float sq = x_val * x_val;
    
    // Warp-level reduction for sum of squares
    float sum_sq = warp_reduce_sum(sq);
    
    // Block-level reduction (shared memory)
    if ((tid & 31) == 0) sdata[tid >> 5] = sum_sq;
    __syncthreads();
    
    if (tid < 32) {
        sum_sq = (tid < (D >> 5)) ? sdata[tid] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        if (tid == 0) sdata[0] = sqrtf(sum_sq + 1e-8f); // This is the L2 norm
    }
    __syncthreads();
    
    float norm = sdata[0];
    if (tid == 0) out_norms[row_idx] = norm;
    
    // 2. Prepare for FWHT: (x * d) / norm
    float val = (x_val * d_val) / norm;
    
    // 3. FWHT (Gila Mode)
    fwht_warp(val, 16);
    sdata[tid] = val; __syncthreads();
    
    for (int step = 32; step < D; step <<= 1) {
        float other = sdata[tid ^ step];
        if ((tid & step) == 0) val = val + other;
        else val = other - val;
        __syncthreads();
        sdata[tid] = val; __syncthreads();
    }
    
    // 4. Orthonormal scaling + Argmin
    float projected = val * (1.0f / sqrtf((float)D));
    
    float min_dist = 1e18f;
    uint8_t best_idx = 0;
    #pragma unroll
    for (int i = 0; i < n_centroids; ++i) {
        float dist = fabsf(projected - centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = (uint8_t)i;
        }
    }
    
    // 5. Final Store
    out_idx[row_idx * D + tid] = best_idx;
}

void fwht_cuda_forward(torch::Tensor x) {
    const int N = x.size(0);
    const int D = x.size(1);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(N);
    dim3 threads(D);
    fwht_kernel_legacy<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), D, N);
}

std::vector<torch::Tensor> ultra_fused_compress_cuda(
    torch::Tensor x, torch::Tensor d, torch::Tensor centroids) 
{
    const int N = x.size(0);
    const int D = x.size(1);
    const int n_centroids = centroids.size(0);
    auto idx_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto norm_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(N);
    dim3 threads(D);
    ultra_fused_compress_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), d.data_ptr<float>(), centroids.data_ptr<float>(), 
        out_idx.data_ptr<uint8_t>(), out_norms.data_ptr<float>(), 
        D, N, n_centroids
    );
    return {out_idx, out_norms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam FWHT Forward (Legacy)");
    m.def("ultra_fused_compress", &ultra_fused_compress_cuda, "KudaHitam Ultra-Fused Compression");
}
