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

__device__ __forceinline__ void fwht_butterfly_warp(float r[8], int lane_id) {
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
    #pragma unroll
    for(int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }
}

__global__ void ultra_fused_full_fusion_kernel_fp16(
    const half* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids, 
    uint8_t* __restrict__ out_idx, 
    float* __restrict__ out_norms, 
    half* __restrict__ out_kmse,
    int D, int N, int n_centroids) 
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    float r[8];
    float sum_sq = 0.0f;
    
    // 1. Vectorized FP16 Load & Cast to Float in Registers
    // Each thread loads 8 halfs (2 float4 equivalent in terms of elements, but half size)
    const half2* x_ptr = reinterpret_cast<const half2*>(x) + global_warp_id * 128 + lane_id * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        half2 val2 = x_ptr[i];
        r[i*2] = __half2float(val2.x);
        r[i*2 + 1] = __half2float(val2.y);
        sum_sq += r[i*2]*r[i*2] + r[i*2+1]*r[i*2+1];
    }

    for (int offset = 16; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    float norm = __shfl_sync(0xffffffff, sum_sq > 0 ? sqrtf(sum_sq) : 1e-8f, 0);
    if (lane_id == 0) out_norms[global_warp_id] = norm;
    
    // 2. Scale by d and norm
    float4 d_vec0 = reinterpret_cast<const float4*>(d)[lane_id];
    float4 d_vec1 = reinterpret_cast<const float4*>(d)[lane_id + 32];
    r[0] = (r[0] * d_vec0.x) / norm; r[1] = (r[1] * d_vec0.y) / norm; 
    r[2] = (r[2] * d_vec0.z) / norm; r[3] = (r[3] * d_vec0.w) / norm;
    r[4] = (r[4] * d_vec1.x) / norm; r[5] = (r[5] * d_vec1.y) / norm; 
    r[6] = (r[6] * d_vec1.z) / norm; r[7] = (r[7] * d_vec1.w) / norm;

    fwht_butterfly_warp(r, lane_id);
    
    float f_scale = 1.0f / 16.0f; 
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
        union { uint8_t b[4]; uint32_t v; } pack0, pack1;
        pack0.b[0] = out_c[0]; pack0.b[1] = out_c[1]; pack0.b[2] = out_c[2]; pack0.b[3] = out_c[3];
        pack1.b[0] = out_c[4]; pack1.b[1] = out_c[5]; pack1.b[2] = out_c[6]; pack1.b[3] = out_c[7];
        reinterpret_cast<uint32_t*>(out_idx)[global_warp_id * 64 + lane_id] = pack0.v;
        reinterpret_cast<uint32_t*>(out_idx)[global_warp_id * 64 + lane_id + 32] = pack1.v;
    }

    fwht_butterfly_warp(r, lane_id);
    
    float b_scale = 1.0f / 16.0f; 
    float m_base = b_scale * norm;
    
    // Final Scale & Write as FP16
    half2* out_ptr = reinterpret_cast<half2*>(out_kmse) + global_warp_id * 128 + lane_id * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float d_val0, d_val1;
        if (i == 0) { d_val0 = d_vec0.x; d_val1 = d_vec0.y; }
        else if (i == 1) { d_val0 = d_vec0.z; d_val1 = d_vec0.w; }
        else if (i == 2) { d_val0 = d_vec1.x; d_val1 = d_vec1.y; }
        else { d_val0 = d_vec1.z; d_val1 = d_vec1.w; }
        
        float res0 = r[i*2] * m_base * d_val0;
        float res1 = r[i*2+1] * m_base * d_val1;
        out_ptr[i] = __floats2half2_rn(res0, res1);
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
    auto half_options = torch::TensorOptions().dtype(torch::kHalf).device(x.device());

    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_kmse = torch::empty({N, D}, half_options);

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N + 7) / 8);

    ultra_fused_full_fusion_kernel_fp16<<<blocks, threads, 0, stream>>>(
        (const half*)x.data_ptr<at::Half>(), 
        d.data_ptr<float>(), 
        centroids.data_ptr<float>(), 
        out_idx.data_ptr<uint8_t>(), 
        out_norms.data_ptr<float>(), 
        (half*)out_kmse.data_ptr<at::Half>(),
        D, N, n_centroids
    );
    return {out_idx, out_norms, out_kmse};
}

void fwht_cuda_forward_warp(torch::Tensor x) {
    // Legacy support for Standalone FWHT (FP32)
    const int N = x.size(0);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N + 7) / 8);
    // Use the optimized FP16-capable logic but adapted for FP32 if needed.
    // For now, standalone is strictly internal-experimental.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward_warp, "Reserved");
    m.def("ultra_fused_full_fusion", &ultra_fused_full_fusion_cuda, "KudaHitam Native FP16 Engine");
}
