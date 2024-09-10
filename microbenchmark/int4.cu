#include <cuda_runtime.h>
#include <stdio.h>

__global__ void copyInt4Kernel(const int4* src, int4* dst, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < numElements) {
        dst[idx] = src[idx];
    }
}

int main() {
    // Number of int4 elements (each int4 contains 4 integers)
    int numElements = 2;

    // Allocate memory on the host
    int *h_src = (int*)malloc(numElements * sizeof(int4));
    int *h_dst = (int*)malloc(numElements * sizeof(int4));

    // Initialize the source array with some values
    for (int i = 0; i < numElements * 4; i++) {
        h_src[i] = i;
    }

    // Allocate memory on the device
    int4 *d_src = nullptr;
    int4 *d_dst = nullptr;
    cudaMalloc((void**)&d_src, numElements * sizeof(int4));
    cudaMalloc((void**)&d_dst, numElements * sizeof(int4));

    // Copy the source array to device memory
    cudaMemcpy(d_src, h_src, numElements * sizeof(int4), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 2;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("thread count: %d; block count: %d\n", threadsPerBlock, blocksPerGrid);

    // Launch the kernel
    copyInt4Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, numElements);

    // Copy the result back to the host
    cudaMemcpy(h_dst, d_dst, numElements * sizeof(int4), cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < numElements * 4; i++) {
          printf("idx: %i, src: %d, dst: %d \t", i, h_src[i], h_dst[i]);
        if (h_src[i] != h_dst[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Copy successful!\n");
    } else {
        printf("Copy failed!\n");
    }

    // Free memory
    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}

