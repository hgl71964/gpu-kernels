#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

// example for 64bit write; uint2 = 2 * uint32_t = 64bit
// 1 word = 4 bytes = 32bit; there're 32 banks; so 
// the ith word -> ()i%32)th bank
__global__ void smem_1(uint32_t *a) {
    // write some data
    __shared__ uint32_t smem[64];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 2; i++)
        smem[i * 32 + tid] = tid;
    __syncthreads();

    // vector write; e.g. tid0 -> data0, data1
    // TODO: notice whether ncu report bank conflict?
    reinterpret_cast<uint2 *>(a)[tid] = reinterpret_cast<const uint2 *>(smem)[tid];
}

// example for 128bit write 
// essentially, if you want to see whether there's share memory access bank conflict
// change the thread access pattern and check ncu's report
__global__ void smem_2(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++)
        smem[i * 32 + tid] = tid;
    __syncthreads();

    // write
    if (tid == 15 || tid == 16)
        reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(smem)[4];
}

int main() {
    // 1 warp
    dim3 grid;
    grid.x = 1;
    dim3 block;
    block.x = 32;

    uint32_t *da, *a;
    a = (uint32_t *)malloc(sizeof(uint32_t)*64);
    cudaMalloc(&da, 64 * sizeof(uint32_t));
    smem_1<<<grid, block>>>(da);
    cudaDeviceSynchronize();

    cudaMemcpy(a, da, sizeof(uint32_t)*64, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < 64; i++) {
        printf("%d\n", a[i]);
    }
    
    return 0;
    
}