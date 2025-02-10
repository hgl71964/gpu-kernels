#include <iostream>
#include "cuda_runtime.h"

#include <cstdlib>  // for rand
#include <ctime> // for time()



#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }
void cudaErrCheck_(cudaError_t stat, const char* file, int line)
{
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

int cdiv(const int length, const int block_size) {
    return (length + block_size - 1)/block_size;
}

__global__ void tile_mm(float *A, float *B, float *C, const int M, const int N, const int K) {

    // extern __shared__ float shared_storage[];
    // float *SA = reinterpret_cast<float *>(shared_storage);
    __shared__ float sa[4][4];
    __shared__ float sb[4][4];
    int block_k = 4;

    int row_start = blockIdx.x * blockDim.x;
    int col_start = blockIdx.y * blockDim.y;

    float accu = 0.0f;
    for (int k = 0; k < K; k+=block_k) {
        // load (assume divisibility)
        sa[threadIdx.x][threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + k];
        sb[threadIdx.x][threadIdx.y] = B[(threadIdx.x+k)*N+col_start+threadIdx.y];
        __syncthreads();

        // accu
        for (int kk = 0; kk < block_k; ++kk) {
            accu += sa[threadIdx.x][kk] * sb[kk][threadIdx.y];
        }
        __syncthreads();
    }

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        C[row*N+col] = accu;
    }
}


int main() {
    srand(time(0));

    const int M = 64;
    const int N = 64;
    const int K = 64;

    const int block_m = 4;
    const int block_n = 4;

    float *ha = (float *)malloc(sizeof(float)*M*K);
    float *hb = (float *)malloc(sizeof(float)*N*K);
    float *hc = (float *)malloc(sizeof(float)*M*N);
    float *ref_c = (float *)malloc(sizeof(float)*M*N);

    float *da, *db, *dc;
    cudaMalloc((void**)&db, sizeof(float)*N*K);
    cudaMalloc((void **)&da, sizeof(float)*M*K);
    cudaMalloc((void**)&dc, sizeof(float)*M*N);

	float min = 0.0f;
	float max = 1.0f;
    for (int i =0;i<M*K;++i) {
        ha[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }
    for (int i =0;i<N*K;++i) {
        hb[i] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
    }
    for (int i = 0;i<M*N;++i) {
        hc[i] = 1.0f;
    }
    for (int i =0;i<M;++i) {
        for (int j =0;j<N;++j){
            float accu = 0.0f;

            //assume both row-major
            for (int k = 0; k < K; ++k) {
                accu += ha[i*K+k] * hb[k*N+j];
            }
            ref_c[i*N+j] = accu;
        }
    }
    cudaMemcpy(da, ha, sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(float)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, sizeof(float)*M*N, cudaMemcpyHostToDevice);


    dim3 block(block_m, block_n, 1);
    dim3 grid(cdiv(M, block_m), cdiv(N, block_n), 1);

        std::cout << "GEMM: " << M << "; " << N << "; " << K << std::endl;
        std::cout << "grid: " << grid.x << "; " << grid.y << std::endl;
        std::cout << "block: " << block.x << "; " << block.y << std::endl;

    // a shared GPU's kernel can silently fail for not sufficient launch resources????
    tile_mm<<<grid, block, 0, nullptr>>>(da, db, dc, M, N, K);

    cudaMemcpy(hc, dc, sizeof(float)*N*M,cudaMemcpyDeviceToHost);
    cudaErrCheck(cudaDeviceSynchronize());


        int err=0;
        for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
			if (std::abs(hc[i*N+j] - ref_c[i*N+j]) > 1e-4) {
			    std::cout << i << "." << j << ": " << hc[i*N+j] << " " << ref_c[i*N+j] << std::endl;
                            err++;
			}
                        if (err > 10)
                                break;
                }
                if (err > 10)
                        break;
        }

        if (err > 0) 
            printf("MM check has %d errors!\n", err);
        else
            printf("OK\n");

}