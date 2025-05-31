#include <cutlass/gemm/device/gemm.h>

nvcc -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include cute_gemm/hello_world.cu -o out

__global__ void hello_cutlass() {
    printf("CUTLASS is available!\n");
}

int main() {
        hello_cutlass<<<1,1>>>();
    cudaDeviceSynchronize();
}

