// 1.cu
// to compile: nvcc -ptx 1.cu -o output.ptx
#include <cuda.h>

__global__ void add(int* a,bool* cond){
    int tid = threadIdx.x;
    if(cond[tid])
        a[tid] = 2;
    }

int main(){
    return 0;
}
