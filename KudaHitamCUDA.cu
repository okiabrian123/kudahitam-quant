#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode V2 (Fused & Vectorized)
 * Optimized for Qwen-3.5 2B (D=256).
 */

__device__ __forceinline__ void fwht_warp(float& val, int mask) {
    for (int step = 1; step <= mask; step <<= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, step);
        if ((threadIdx.x & step) == 0) {
            val = val + other;
        } else {
            val = other - val;
        }
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

__global__ void fused_compress_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ centroids, 
    uint8_t* __restrict__ out, 
    int D, int N, int n_centroids) 
{
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;
    int tid = threadIdx.x;
    
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
    float projected = val * scale;
    
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
    out[row_idx * D + tid] = best_idx;
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

torch::Tensor fused_compress_cuda(torch::Tensor x, torch::Tensor centroids) {
    const int N = x.size(0);
    const int D = x.size(1);
    const int n_centroids = centroids.size(0);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    torch::Tensor out = torch::empty({N, D}, options);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(N);
    dim3 threads(D);
    fused_compress_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), 
        centroids.data_ptr<float>(), 
        out.data_ptr<uint8_t>(), 
        D, N, n_centroids
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam FWHT Forward (Legacy)");
    m.def("fused_compress", &fused_compress_cuda, "KudaHitam Fused FWHT + Quantization");
}
