#include <stdio.h>

// compile: nvcc block_launch_order.cu

__global__ void AtomicPrint(unsigned int *addr) {

    unsigned int old = atomicInc(addr, 10000);


    if (threadIdx.x == 0) {
        printf("blockIdx.x == %d, old == %d\n", blockIdx.x, old);
    }
}

int main() {

    unsigned int *h_addr, *d_addr;

    h_addr = (unsigned int *)malloc(sizeof(unsigned int));
    h_addr[0] = 0;

    cudaMalloc((void **)&d_addr, sizeof(unsigned int));
    cudaMemcpy(d_addr, h_addr, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    AtomicPrint<<<1024, 1>>>(d_addr);

    cudaDeviceSynchronize();
}


