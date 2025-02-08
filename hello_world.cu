#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <iostream>
//#include <stdio.h>

using namespace std;

__global__ void cuda_hello()
{
    // when running insides docker container, device cannot print anything
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from device!\n");
    printf("device idx: %d \n", idx);
}

// .cu file can be directly compiled by cmake's CUDA
int main()
{
    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("Hello World from Host!\n");

    int driver_version;
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    cout << "driver: " << driver_version << endl;
    cout << "runtime: " << runtime_version << endl;

    return 0;
}
