#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode (Raw CUDA)
 * Uses Warp-Level Shuffles for maximum speed.
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

__global__ void fwht_kernel_cuda(float* x, int D, int N) {
    // Each block handles one row (D items)
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;
    
    int tid = threadIdx.x;
    if (tid >= D) return;
    
    float val = x[row_idx * D + tid];
    
    // Stage 1: Intra-warp shuffle (for steps 1, 2, 4, 8, 16)
    fwht_warp(val, 16);
    
    // Stage 2: Inter-warp sync using Shared Memory
    __shared__ float sdata[1024]; // Support up to 1024 head_dim
    sdata[tid] = val;
    __syncthreads();
    
    for (int step = 32; step < D; step <<= 1) {
        int partner = tid ^ step;
        float other = sdata[partner];
        if ((tid & step) == 0) {
            val = val + other;
        } else {
            val = other - val;
        }
        __syncthreads();
        sdata[tid] = val;
        __syncthreads();
    }
    
    // Orthonormal scaling
    float scale = 1.0f / sqrtf((float)D);
    x[row_idx * D + tid] = val * scale;
}

void fwht_cuda_forward(torch::Tensor x) {
    const int N = x.size(0);
    const int D = x.size(1);
    
    // Device and Stream awareness for multi-GPU (Kaggle T4 x2)
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    dim3 blocks(N);
    dim3 threads(D); // Assumes D <= 1024 (std head_dim is 128/256)
    
    fwht_kernel_cuda<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), D, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam FWHT Forward (CUDA)");
}
