//
// 1.cu
// to ptx: nvcc -ptx -arch 80 1.cu -o output.ptx
// to run: nvcc -arch 80 1.cu 
// to ptx -> sass: ptxas -v -o 1.cubin --gpu-name sm_89 1.ptx
//

#include <cuda.h>
#include <stdio.h>

__global__ void add(int* a, int*b, int*c, long long int * clock_prof){
    int aa = *a;
    int bb = *b;
    long long int before = clock64();
    int cc = aa + bb;
    long long int after = clock64();
    *c = cc;
    *clock_prof = (after - before);
}

int main(){
    // alloc
    int *ha = (int *)malloc(sizeof(int)*1);
    int *hb = (int *)malloc(sizeof(int)*1);
    int *hc = (int *)malloc(sizeof(int)*1);
    long long int  *clock = (long long int  *)malloc(sizeof(long long int )*1);
    int *da;
    int *db;
    int *dc;
    long long int  *dclock;
    cudaMalloc((void**)&db, sizeof(int)*1);
    cudaMalloc((void **)&da, sizeof(int)*1);
    cudaMalloc((void**)&dc, sizeof(int)*1);
    cudaMalloc((void**)&dclock, sizeof(long long int )*1);
    *ha = 0;
    *hb = 1;
    *hc = 2;
    cudaMemcpy(da, ha, sizeof(int)*1,cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(int)*1, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, sizeof(int)*1, cudaMemcpyHostToDevice);

    //
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    add<<<grid, block>>>(da, db, dc, dclock);
    cudaDeviceSynchronize();

    cudaMemcpy(hc, dc, sizeof(int)*1,cudaMemcpyDeviceToHost);
    cudaMemcpy(clock, dclock, sizeof(long long int )*1,cudaMemcpyDeviceToHost);

    if (*hc == 1) {
        printf("OK\n");
    } else {
        printf("expect 1, got %d\n", *hc);
    }

    printf("clock: %llu\n", *clock);

    return 0;
}
