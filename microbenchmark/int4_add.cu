#include <cuda_runtime.h>
#include <stdio.h>

__global__ void addInt4Kernel( int4* a,  int4* b, int4* result) {
    // Since we are using only 1 block and 1 thread, no need for indexing
    //result[0].x = a[0].x + b[0].x;
    //result[0].y = a[0].y + b[0].y;
    //result[0].z = a[0].z + b[0].z;
    //result[0].w = a[0].w + b[0].w;
    *result = *a + *b;
}

int main() {
    // Declare and initialize host-side int4 variables
    int4 h_a = {1, 2, 3, 4};
    int4 h_b = {5, 6, 7, 8};
    int4 h_result;

    // Declare device-side int4 pointers
    int4 *d_a, *d_b, *d_result;

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, sizeof(int4));
    cudaMalloc((void**)&d_b, sizeof(int4));
    cudaMalloc((void**)&d_result, sizeof(int4));

    // Copy input data from host to device
    cudaMemcpy(d_a, &h_a, sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int4), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 1 thread
    addInt4Kernel<<<1, 1>>>(d_a, d_b, d_result);

    // Copy result back from device to host
    cudaMemcpy(&h_result, d_result, sizeof(int4), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result: (%d, %d, %d, %d)\n", h_result.x, h_result.y, h_result.z, h_result.w);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}

