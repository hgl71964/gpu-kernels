#include <stdio.h>

#include <cstdlib>  // for rand
#include <ctime> // for time()

// compile: nvcc thread_coarsening.cu

__global__ void vectorAddCoarsened(float *A, float *B, float *C, int N, int coarseningFactor) {
    int idx = blockIdx.x * blockDim.x * coarseningFactor + threadIdx.x;
    for(int i = 0; i < coarseningFactor; i++) {
        if (idx + i * blockDim.x < N) {
            C[idx + i * blockDim.x] = A[idx + i * blockDim.x] + B[idx + i * blockDim.x];
        }
    }
}

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024; // Size of vectors
    int coarseningFactor = 2; // Example coarsening factor
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C; // Host vectors
    float *d_A, *d_B, *d_C; // Device vectors
    float *h_ref, *ref; 

    // Allocate memory on host
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    h_ref = (float *)malloc(size);

	float min = 0.0f;
	float max = 1.0f;
    for(int i = 0; i < N; i++) {
        h_A[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
        h_B[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }

    // Allocate memory on device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMalloc(&ref, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // coarsened version
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + coarseningFactor * threadsPerBlock - 1) / (coarseningFactor * threadsPerBlock);
    printf("N: %d, coarseningFactor: %d\n", N, coarseningFactor);
    printf("grid: %d, block: %d\n", blocksPerGrid, threadsPerBlock);
    vectorAddCoarsened<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, coarseningFactor);
    cudaDeviceSynchronize();

    // non-coarsened version
    dim3 block(256);
    dim3 grid(0);
    grid.x = (N + block.x - 1) / block.x;
    printf("grid: %d, block: %d\n", grid.x, block.x);
    vectorAdd<<<grid, block>>>(d_A, d_B, ref, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, ref, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double epsilon = 1.0E-8; // Error tolerance
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - h_ref[i]) > epsilon) {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", h_C[i], h_ref[i], i);
        }
    }

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}

