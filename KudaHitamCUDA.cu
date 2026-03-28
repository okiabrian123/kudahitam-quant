#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode Ultra-Fused V7.5 (Monolithic)
 * Everything in ONE PASS: Norm -> Scale -> FWHT -> Quant -> Dequant -> FWHT -> Store
 * Formula: Out = FWHT(LloydMax(FWHT(X_norm * D), centroids)) * D * ||X||
 */

__device__ __forceinline__ void fwht_butterfly_warp(float r[8], int lane_id) {
    // Intra-thread (Steps 1, 2)
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
    // Inter-thread (Steps 4, 8, 16, 32, 64)
    #pragma unroll
    for(int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for(int k = 0; k < 8; ++k) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    // Intra-thread (Step 128)
    #pragma unroll
    for(int j = 0; j < 4; ++j) {
        float a = r[j];
        float b = r[j + 4];
        r[j] = a + b;
        r[j + 4] = a - b;
    }
}

__global__ void fwht_kernel_warp(float* __restrict__ x, int N) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    float r[8];
    float4 load0 = reinterpret_cast<float4*>(x)[global_warp_id * 64 + lane_id];
    float4 load1 = reinterpret_cast<float4*>(x)[global_warp_id * 64 + lane_id + 32];
    r[0] = load0.x; r[1] = load0.y; r[2] = load0.z; r[3] = load0.w;
    r[4] = load1.x; r[5] = load1.y; r[6] = load1.z; r[7] = load1.w;

    fwht_butterfly_warp(r, lane_id);
    
    float scale = 1.0f / 16.0f; // sqrt(256)
    float4 out0 = {r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale};
    float4 out1 = {r[4] * scale, r[5] * scale, r[6] * scale, r[7] * scale};
    
    reinterpret_cast<float4*>(x)[global_warp_id * 64 + lane_id] = out0;
    reinterpret_cast<float4*>(x)[global_warp_id * 64 + lane_id + 32] = out1;
}

__global__ void ultra_fused_full_fusion_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ d, 
    const float* __restrict__ centroids, 
    uint8_t* __restrict__ out_idx, 
    float* __restrict__ out_norms, 
    float* __restrict__ out_kmse,
    int D, int N, int n_centroids) 
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    float r[8];
    float sum_sq = 0.0f;
    
    // 1. Vectorized Load & Partial Norm
    float4 load0 = reinterpret_cast<const float4*>(x)[global_warp_id * 64 + lane_id];
    float4 load1 = reinterpret_cast<const float4*>(x)[global_warp_id * 64 + lane_id + 32];
    r[0] = load0.x; r[1] = load0.y; r[2] = load0.z; r[3] = load0.w;
    r[4] = load1.x; r[5] = load1.y; r[6] = load1.z; r[7] = load1.w;

    #pragma unroll
    for(int k=0; k<8; ++k) sum_sq += r[k]*r[k];
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

    // 3. Forward FWHT
    fwht_butterfly_warp(r, lane_id);
    
    // 4. Quantize & Dequantize In-Place
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
    
    // Write Indices
    union { uint8_t b[4]; uint32_t v; } pack0, pack1;
    pack0.b[0] = out_c[0]; pack0.b[1] = out_c[1]; pack0.b[2] = out_c[2]; pack0.b[3] = out_c[3];
    pack1.b[0] = out_c[4]; pack1.b[1] = out_c[5]; pack1.b[2] = out_c[6]; pack1.b[3] = out_c[7];
    reinterpret_cast<uint32_t*>(out_idx)[global_warp_id * 64 + lane_id] = pack0.v;
    reinterpret_cast<uint32_t*>(out_idx)[global_warp_id * 64 + lane_id + 32] = pack1.v;

    // 5. Backward FWHT
    fwht_butterfly_warp(r, lane_id);
    
    // 6. Scale back to original domain
    float b_scale = 1.0f / 16.0f; 
    float m_base = b_scale * norm;
    r[0] *= (m_base * d_vec0.x); r[1] *= (m_base * d_vec0.y);
    r[2] *= (m_base * d_vec0.z); r[3] *= (m_base * d_vec0.w);
    r[4] *= (m_base * d_vec1.x); r[5] *= (m_base * d_vec1.y);
    r[6] *= (m_base * d_vec1.z); r[7] *= (m_base * d_vec1.w);

    float4 out_r0 = {r[0], r[1], r[2], r[3]};
    float4 out_r1 = {r[4], r[5], r[6], r[7]};
    reinterpret_cast<float4*>(out_kmse)[global_warp_id * 64 + lane_id] = out_r0;
    reinterpret_cast<float4*>(out_kmse)[global_warp_id * 64 + lane_id + 32] = out_r1;
}

std::vector<torch::Tensor> ultra_fused_full_fusion_cuda(
    torch::Tensor x, torch::Tensor d, torch::Tensor centroids) 
{
    const int N = x.size(0);
    const int D = x.size(1);
    if (D != 256) throw std::runtime_error("Warp-Only Full Fusion requires D=256");
    const int n_centroids = centroids.size(0);
    auto idx_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto norm_options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor out_idx = torch::empty({N, D}, idx_options);
    torch::Tensor out_norms = torch::empty({N, 1}, norm_options);
    torch::Tensor out_kmse = torch::empty({N, D}, norm_options);
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N + 7) / 8);
    ultra_fused_full_fusion_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), d.data_ptr<float>(), centroids.data_ptr<float>(), 
        out_idx.data_ptr<uint8_t>(), out_norms.data_ptr<float>(), out_kmse.data_ptr<float>(),
        D, N, n_centroids
    );
    return {out_idx, out_norms, out_kmse};
}

void fwht_cuda_forward_warp(torch::Tensor x) {
    const int N = x.size(0);
    const int D = x.size(1);
    if (D != 256) throw std::runtime_error("Warp-Only FWHT (Optimized) requires D=256");
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 threads(256);
    dim3 blocks((N + 7) / 8);
    fwht_kernel_warp<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward_warp, "KudaHitam FWHT Forward (Warp-Optimized)");
    m.def("ultra_fused_full_fusion", &ultra_fused_full_fusion_cuda, "KudaHitam Monolithic Full Fusion");
}
