//
// 1.cu
// to ptx: nvcc -ptx 1.cu -o 1.ptx
// to run: nvcc -arch sm_80 1.cu
// to ptx -> sass: ptxas -v -o 1.cubin --gpu-name sm_80 1.ptx
//

#include <cuda.h>
#include <stdio.h>

using lint = long int;

__global__ void add(lint* a, lint*b, lint*c, long long int * clock_prof){
    lint a1 = a[0];
    lint b1 = b[0];
    lint a2 = a[1];
    lint b2 = b[1];
    lint a3 = a[2];
    lint b3 = b[2];

    long long int before = clock64();
    lint c1 = a1 + b1;
    lint c2 = a2 + b2;
    lint c3 = a3 + b3;
    long long int after = clock64();

    c[0] = c1;
    c[1] = c2;
    c[2] = c3;

    *clock_prof = (after - before);
}

int main(){
    // alloc
    lint *ha = (lint *)malloc(sizeof(lint)*100);
    lint *hb = (lint *)malloc(sizeof(lint)*100);
    lint *hc = (lint *)malloc(sizeof(lint)*100);
    long long int  *clock = (long long int  *)malloc(sizeof(long long int )*1);
    lint *da;
    lint *db;
    lint *dc;
    long long int  *dclock;
    cudaMalloc((void**)&db, sizeof(lint)*100);
    cudaMalloc((void **)&da, sizeof(lint)*100);
    cudaMalloc((void**)&dc, sizeof(lint)*100);
    cudaMalloc((void**)&dclock, sizeof(long long int )*1);
    for (int i =0;i<100;++i) {
        ha[i] = 1;
        hb[i] = 1;
        hc[i] = 0;
    }
    cudaMemcpy(da, ha, sizeof(lint)*100,cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(lint)*100, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, sizeof(lint)*100, cudaMemcpyHostToDevice);

    //
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);

    add<<<grid, block>>>(da, db, dc, dclock);
    cudaDeviceSynchronize();

    cudaMemcpy(hc, dc, sizeof(lint)*100,cudaMemcpyDeviceToHost);
    cudaMemcpy(clock, dclock, sizeof(long long int )*1,cudaMemcpyDeviceToHost);

    int cnt = 0 ;
    for (int i =0;i<100;++i) {
        //printf("i: %d - hc: %d \t", i, hc[i]);
        if (hc[i]==2)
            cnt++;
    }
    printf("\n");

    printf("got %d hc\n", cnt);
    printf("clock: %llu\n", *clock);

    // NOTE: this is incorrect; because within clock64 there can be different SASS
    float avg = static_cast<float>(*clock) / static_cast<float>(cnt);
    printf("average: %f\n", avg);

    return 0;
}
