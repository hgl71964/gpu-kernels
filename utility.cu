
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

using namespace std;

#define CHECK_CUDA(func, ...)                                                               \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << func << std::endl;           \
      return e;                                                                             \
    }                                                                                       \
  }

__global__ void hello()
{
    // __shared__ char smem;
    extern __shared__ uint8_t smem[];  // dynamic shared mem

    // when running insides docker container, device cannot print anything
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from device!\n");
    printf("device idx: %d \n", idx);
}

// .cu file can be directly compiled by cmake's CUDA
int main()
{
    dim3 block;
    block.x = 2;
    // block.y = 1;

    auto kernel = hello;

    uint32_t smem_size = 4;

    const uint32_t num_warps = 4;
    const uint32_t warp_size = 32;

    int dev_id = 0;

    constexpr uint32_t num_threads = num_warps * warp_size;

    int max_smem_per_sm = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    cout << "max_smem_per_sm: " << max_smem_per_sm << endl;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);

    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_threads, smem_size);

    cout << "num_sm: " << num_sm << " num_blocks_per_sm: " << num_blocks_per_sm << endl;


    hello<<<1, block>>>();
    cudaDeviceSynchronize();

    int driver_version;
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    cout << "driver: " << driver_version << endl;
    cout << "runtime: " << runtime_version << endl;

    return 0;
}
