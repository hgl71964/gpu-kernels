
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void cuda_idx()
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x; // thread在对应block内的行id
    const int ty = threadIdx.y; // thread在对应block内的列id
    const int tid = ty * blockDim.x + tx; // thread在对应block中的全局id（从左到右，从上到下，从0开始逐一标）


    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    // 当前thread负责加载的第一个数在s_a中的col
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    
    // 当前thread负责把B中的相关数据从global memory加载到SMEM，
    // 这里在计算该thread负责加载的第一个数在s_b中的row
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    // 当前thread负责加载的第一个数在s_b中的col
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    printf("bx: %d by: %d tx: %d ty: %d tid: %d load_a_smem_m: %d load_a_smem_k: %d load_b_smem_k: %d load_b_smem_n: %d \n", bx , by, tx, ty, tid, load_a_smem_m, load_a_smem_k, load_b_smem_k, load_b_smem_n);

    printf("tid & 1: %d \n", tid & 1);
}

// .cu file can be directly compiled by cmake's CUDA
int main()
{
    dim3 block;
    block.x = 2;
    block.y = 2;

    cuda_idx<<<1, block>>>();
    cudaDeviceSynchronize();

    int driver_version;
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    cout << "driver: " << driver_version << endl;
    cout << "runtime: " << runtime_version << endl;

    return 0;
}
