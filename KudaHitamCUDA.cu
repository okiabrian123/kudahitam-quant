#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * KudaHitam FWHT - Gila Mode V7.5 (FINAL OPTIMIZED)
 * Everything Warp-Only Register Based. Zero Shared Memory. Zero __syncthreads.
 */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fwht_kernel_warp(float* __restrict__ x, int D, int N) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    float* row_ptr = x + global_warp_id * 256;
    float4* x_vec = reinterpret_cast<float4*>(row_ptr);
    float4 v0 = x_vec[lane_id];      // 0..127
    float4 v1 = x_vec[lane_id + 32]; // 128..255
    float r[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};

    // FWHT Step 1..4 (Intra-thread)
    #pragma unroll
    for (int step = 1; step <= 4; step <<= 1) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if ((i & step) == 0) {
                float a = r[i]; float b = r[i | step];
                r[i] = a + b; r[i | step] = a - b;
            }
        }
    }

    // FWHT Step 8..128 (Inter-thread Shuffle)
    #pragma unroll
    for (int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }

    // FWHT Step 128 (Inter-register)
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    float scale = 1.0f / 16.0f; // 1/D for reconstruct-like behavior (or 1/sqrt(D))
    float4 out0 = {r[0]*scale, r[1]*scale, r[2]*scale, r[3]*scale};
    float4 out1 = {r[4]*scale, r[5]*scale, r[6]*scale, r[7]*scale};
    x_vec[lane_id] = out0;
    x_vec[lane_id + 32] = out1;
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
    
    const float4* x_vec = reinterpret_cast<const float4*>(x + global_warp_id * 256);
    const float4* d_vec = reinterpret_cast<const float4*>(d);
    float4 x0 = x_vec[lane_id];      
    float4 x1 = x_vec[lane_id + 32]; 
    float4 v_d0 = d_vec[lane_id];
    float4 v_d1 = d_vec[lane_id + 32];
    
    float r[8] = {x0.x * v_d0.x, x0.y * v_d0.y, x0.z * v_d0.z, x0.w * v_d0.w,
                  x1.x * v_d1.x, x1.y * v_d1.y, x1.z * v_d1.z, x1.w * v_d1.w};

    float sum_sq = 0;
    #pragma unroll
    for(int i = 0; i < 8; ++i) sum_sq += r[i] * r[i];
    sum_sq = warp_reduce_sum(sum_sq);
    float norm = __shfl_sync(0xffffffff, sqrtf(sum_sq) + 1e-8f, 0);
    if (lane_id == 0) out_norms[global_warp_id] = norm;
    float inv_norm = 1.0f / norm;

    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] *= inv_norm;

    // FWHT Steps 1..4, 8..128, 128
    #pragma unroll
    for (int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for (int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if ((j & step) == 0) {
                    float a = r[i + j]; float b = r[i + (j | step)];
                    r[i + j] = a + b; r[i + (j | step)] = a - b;
                }
            }
        }
    }
    #pragma unroll
    for (int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }

    float scale = 1.0f / 16.0f;
    uint8_t out_c[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float val = r[i] * scale;
        float min_d = 1e18f; uint8_t best_idx = 0;
        for (int j = 0; j < n_centroids; j++) {
            float dist = val - centroids[j]; dist = dist * dist;
            if (dist < min_d) { min_d = dist; best_idx = j; }
        }
        out_c[i] = best_idx;
    }
    
    union { uint8_t b[4]; uint32_t v; } pack0, pack1;
    pack0.b[0] = out_c[0]; pack0.b[1] = out_c[1]; pack0.b[2] = out_c[2]; pack0.b[3] = out_c[3];
    pack1.b[0] = out_c[4]; pack1.b[1] = out_c[5]; pack1.b[2] = out_c[6]; pack1.b[3] = out_c[7];
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out_idx + global_warp_id * 256);
    out_ptr[lane_id] = pack0.v; out_ptr[lane_id + 32] = pack1.v;
}

__global__ void ultra_fused_reconstruct_kernel(
    const uint8_t* __restrict__ in_idx,
    const float* __restrict__ vec_norms,
    const float* __restrict__ centroids,
    const float* __restrict__ d,
    float* __restrict__ out_kmse,
    int D, int N)
{
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= N) return;
    int lane_id = threadIdx.x & 31;
    
    const uint32_t* in_ptr = reinterpret_cast<const uint32_t*>(in_idx + global_warp_id * 256);
    uint32_t pack0_v = in_ptr[lane_id];
    uint32_t pack1_v = in_ptr[lane_id + 32];
    union { uint8_t b[4]; uint32_t v; } pack0, pack1;
    pack0.v = pack0_v; pack1.v = pack1_v;
    
    float r[8] = {centroids[pack0.b[0]], centroids[pack0.b[1]], centroids[pack0.b[2]], centroids[pack0.b[3]],
                  centroids[pack1.b[0]], centroids[pack1.b[1]], centroids[pack1.b[2]], centroids[pack1.b[3]]};

    #pragma unroll
    for (int step = 1; step <= 2; step <<= 1) {
        #pragma unroll
        for (int i = 0; i <= 4; i += 4) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if ((j & step) == 0) {
                    float a = r[i + j]; float b = r[i + (j | step)];
                    r[i + j] = a + b; r[i + (j | step)] = a - b;
                }
            }
        }
    }
    #pragma unroll
    for (int t_step = 1; t_step <= 16; t_step <<= 1) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float other = __shfl_xor_sync(0xffffffff, r[k], t_step);
            if ((lane_id & t_step) == 0) r[k] = r[k] + other;
            else r[k] = other - r[k];
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float a = r[j]; float b = r[j + 4];
        r[j] = a + b; r[j + 4] = a - b;
    }
    
    float scale = 1.0f / 16.0f;
    float norm = vec_norms[global_warp_id];
    const float4* v_d_vec = reinterpret_cast<const float4*>(d);
    float4 v_d0 = v_d_vec[lane_id];
    float4 v_d1 = v_d_vec[lane_id + 32];
    float m0 = scale * norm * v_d0.x; float m1 = scale * norm * v_d0.y;
    float m2 = scale * norm * v_d0.z; float m3 = scale * norm * v_d0.w;
    float m4 = scale * norm * v_d1.x; float m5 = scale * norm * v_d1.y;
    float m6 = scale * norm * v_d1.z; float m7 = scale * norm * v_d1.w;
    
    float4 out0 = {r[0]*m0, r[1]*m1, r[2]*m2, r[3]*m3};
    float4 out1 = {r[4]*m4, r[5]*m5, r[6]*m6, r[7]*m7};
    reinterpret_cast<float4*>(out_kmse + global_warp_id * 256)[lane_id] = out0;
    reinterpret_cast<float4*>(out_kmse + global_warp_id * 256)[lane_id + 32] = out1;
}

std::vector<torch::Tensor> ultra_fused_compress_cuda(torch::Tensor x, torch::Tensor d, torch::Tensor centroids) {
    const int N = x.size(0); const int D = x.size(1);
    at::cuda::CUDAGuard device_guard(x.device());
    torch::Tensor out_idx = torch::empty({N, D}, torch::device(x.device()).dtype(torch::kUInt8));
    torch::Tensor out_norms = torch::empty({N, 1}, torch::device(x.device()).dtype(torch::kFloat32));
    ultra_fused_compress_kernel<<<(N + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), d.data_ptr<float>(), centroids.data_ptr<float>(), out_idx.data_ptr<uint8_t>(), out_norms.data_ptr<float>(), D, N, centroids.size(0));
    return {out_idx, out_norms};
}

torch::Tensor ultra_fused_reconstruct_cuda(torch::Tensor indices, torch::Tensor vec_norms, torch::Tensor centroids, torch::Tensor d) {
    const int N = indices.size(0); const int D = indices.size(1);
    at::cuda::CUDAGuard device_guard(indices.device());
    torch::Tensor out_kmse = torch::empty({N, D}, torch::device(indices.device()).dtype(torch::kFloat32));
    ultra_fused_reconstruct_kernel<<<(N + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        indices.data_ptr<uint8_t>(), vec_norms.data_ptr<float>(), centroids.data_ptr<float>(), d.data_ptr<float>(), out_kmse.data_ptr<float>(), D, N);
    return out_kmse;
}

void fwht_cuda_forward(torch::Tensor x) {
    const int N = x.size(0); const int D = x.size(1);
    at::cuda::CUDAGuard device_guard(x.device());
    fwht_kernel_warp<<<(N + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), D, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwht_cuda_forward, "KudaHitam Warp-Only FWHT");
    m.def("ultra_fused_compress", &ultra_fused_compress_cuda, "KudaHitam Ultra-Fused Compression");
    m.def("ultra_fused_reconstruct", &ultra_fused_reconstruct_cuda, "KudaHitam Ultra-Fused Reconstruction");
}
