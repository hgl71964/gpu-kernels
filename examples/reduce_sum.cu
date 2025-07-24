#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

int cdiv(int len, int block_size) {
        return (len + block_size -1)/block_size;
}

template <int BLOCK_SIZE>
__global__ void native(float* in, float* out, int M) {
    float local_sum = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < M; i+=gridDim.x*BLOCK_SIZE) {
        local_sum += in[i];
    }
    atomicAdd(out, local_sum);
}

template <int BLOCK_SIZE>
__global__ void reduce_sum(float* in, float* out, int M) {
    __shared__ float sdata[BLOCK_SIZE];

    if (threadIdx.x + blockIdx.x * blockDim.x >= M) return;
    sdata[threadIdx.x] = in[threadIdx.x + blockIdx.x * blockDim.x];
    __syncthreads();

    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdata[0];
}

template <int BLOCK_SIZE, int VEC_SIZE>
__global__ void reduce_sum_vec(float* in, float* out, int M) {
    __shared__ float sdata[BLOCK_SIZE];

    if ((threadIdx.x + blockIdx.x * blockDim.x) * VEC_SIZE >= M) return;

    float4 data = reinterpret_cast<float4*>(in)[(threadIdx.x + blockIdx.x * blockDim.x)];

    sdata[threadIdx.x] = data.x + data.y + data.z + data.w;
    __syncthreads();

    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdata[0];
}

// This helper function performs a reduction on a float value within a single warp.
// The result is broadcast to all threads in the warp, but we only care about the
// value in lane 0.
// It assumes all threads in the warp participate.
__device__ inline float warp_reduce_sum(float val) {
    // The mask 0xffffffff indicates all 32 threads in the warp are participating.
    // __shfl_down_sync(mask, var, delta) gets the value of `var` from the thread
    // `delta` lanes down from the current thread.
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    // After these operations, the thread at lane 0 of the warp holds the sum.
    // The result is then broadcast back to all threads in the warp using __shfl_sync.
    return __shfl_sync(0xffffffff, val, 0);
}

template <int BLOCK_SIZE>
__global__ void reduce_sum_warp(float* in, float* out, int M) {
    // For this implementation, BLOCK_SIZE should be a multiple of 32 (warpSize).
    // Statically assert this for safety.
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32.");

    // Shared memory to store the partial sum of each warp.
    // We only need as many slots as there are warps in the block.
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    __shared__ float sdata[WARPS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    
    // 1. Load initial value from global memory.
    // Each thread starts with its own value or 0 if it's out of bounds.
    float val = (gid < M) ? in[gid] : 0.0f;

    // 2. Perform the intra-warp reduction.
    // After this, `val` in every thread of a warp will contain the sum of values
    // for that entire warp.
    val = warp_reduce_sum(val);

    // 3. Store the partial sum for each warp into shared memory.
    // Only the first thread of each warp (the "warp leader") needs to do this.
    unsigned int lane_id = tid % 32;
    unsigned int warp_id = tid / 32;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }

    // Synchronize to ensure all warp sums are written to shared memory
    // before the final reduction step. This is the only __syncthreads() needed!
    __syncthreads();

    // 4. Final reduction across warps.
    // The first warp (warp_id == 0) is responsible for reducing the partial sums
    // stored in sdata.
    // We reload `val` with the partial sums.
    val = (tid < WARPS_PER_BLOCK) ? sdata[lane_id] : 0.0f;

    if (warp_id == 0) {
        // The first warp performs the final reduction on the partial sums.
        val = warp_reduce_sum(val);
    }
    
    // 5. The first thread (thread 0 of the block) writes the final result.
    if (tid == 0) {
        out[blockIdx.x] = val;
    }
}


int main() {
        const int M = 256;
        const int BM = 32;
        const int NUM_BLOCKS = cdiv(M, BM);

        float *hc =  (float *) malloc(sizeof(float)*M);
        float *out =  (float *) malloc(sizeof(float)*NUM_BLOCKS);

        for (int i = 0 ; i < M ; ++i) {
            hc[i] = i;
        }

        float accu = 0.0f;
        for (int i = 0; i < M; ++i) {
            accu += hc[i];
        }
        float gold = accu;

        float *dc;
        float *dout;
        cudaMalloc((void**)&dc, sizeof(float)*M);
        cudaMalloc((void**)&dout, sizeof(float)*NUM_BLOCKS);
        cudaMemcpy(dc, hc, sizeof(float)*M,cudaMemcpyHostToDevice);
        cudaMemcpy(dout, out, sizeof(float)*NUM_BLOCKS,cudaMemcpyHostToDevice);

        dim3 block(BM, 1, 1);
        dim3 grid(NUM_BLOCKS, 1, 1);
        std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        std::cout << "block: " << block.x << " " << block.y << " " << block.z << std::endl;
        // native<BM><<<grid, block>>>(dc, dout, M);
        // reduce_sum<BM><<<grid, block>>>(dc, dout, M);
        // reduce_sum_vec<BM, 4><<<grid, block>>>(dc, dout, M); // will skip if out of bound
        reduce_sum_warp<BM><<<grid, block>>>(dc, dout, M); // will skip if out of bound

        cudaMemcpy(out, dout, sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // final reduce
        accu = 0.0f;
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            std::cout << out[i] << std::endl;
            accu += out[i];
        }
        std::cout << "Gold: " << gold << std::endl;
        std::cout << "Accu: " << accu << std::endl;
}
