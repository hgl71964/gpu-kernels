#include <iostream>
#include "cuda_runtime.h"
#include <cmath>


#define BLOCK_DIM 8


int cdiv(int length, int block_size) {
        return (length + block_size - 1)/block_size;
}

__global__ void native_mm(float* da, float* db, float* dc, int block_m, int block_n, int block_k, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
                for (int k = 0; k < K; ++k) {
                        float a = da[row*M + k];
                        float b = db[k*N + col];
                        dc[row*M + col] += a*b;
                }
        }
}

__global__ void tiled_mm(float* da, float* db, float* dc, int block_m, int block_n, int block_k, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	 // __shared__ float tileA[block_m][block_k];
	 // __shared__ float tileB[block_k][block_n];
	 __shared__ float tileA[BLOCK_DIM][BLOCK_DIM];
	 __shared__ float tileB[BLOCK_DIM][BLOCK_DIM];

         int tile_row = threadIdx.y;
         int tile_col = threadIdx.x;

         float res = 0; // thread local

        for (int k = 0; k < K; k+=block_k) {
                // fetch to shared memory
                float a=0;
                float b=0;
                if (M*row+k < M*K) {
                        a = da[row*M+k];
                }
                if (k*N + col < N*K) {
                        b = db[k*N + col];
                }

		tileA[tile_row][tile_col] = a;
		tileB[tile_row][tile_col] = b;

                __syncthreads();

                // accum
                for (int kk = 0; kk < block_k; ++kk) {
                        float a = tileA[tile_row][kk];
                        float b = tileB[kk][tile_col];
                        res += a*b;
                }
                __syncthreads();
        }

        if (row < M && col < N) {
                dc[row*M + col] = res;
        }
}

int main() {
        int m = 1024;
        int n = 1024;
        int k = 1024;

        // allocate input/output buffer
        float *ha = (float *)malloc(sizeof(float) * m * k);
        float *hb = (float *)malloc(sizeof(float)*n*k);
        float *hc = (float *)malloc(sizeof(float)*m*n);

        float *da;
        float *db;
        float *dc;
        cudaMalloc((void**)&db, sizeof(float)*n*k);
        cudaMalloc((void **)&da, sizeof(float)*m*k);
        cudaMalloc((void**)&dc, sizeof(float)*m*n);

        // init and transfer memory
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                        ha[i*m + j] = 1.0;
                }
        }
        for (int i = 0; i < n; ++i) {
                for (int j = 0; j < k; ++j) {
                        hb[i*n + j] = 2.0;
                }
        }
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        hc[i*m + j] = 0.0;
                }
        }
        cudaMemcpy(da, ha, sizeof(float)*m*k,cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, sizeof(float)*n*k, cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc, sizeof(float)*n*m, cudaMemcpyHostToDevice);

        // kernel param
        int block_m = BLOCK_DIM;
        int block_n = BLOCK_DIM;
        int block_k = BLOCK_DIM;

        int grid_x = cdiv(m, block_m);
        int grid_y = cdiv(n, block_n);

        dim3 grid(grid_x, grid_y, 1);
        dim3 block(block_m, block_n, 1);

        // launch
        std::cout << "grid: " << grid_x << "; " << grid_y << std::endl;
        std::cout << "block: " << block_m << "; " << block_n << std::endl;

        // native_mm<<<grid, block>>>(da, db, dc, block_m, block_n, block_k, m, n, k);
        tiled_mm<<<grid, block>>>(da, db, dc, block_m, block_n, block_k, m, n, k);
        cudaDeviceSynchronize();

        // test output
        cudaMemcpy(hc, dc, sizeof(float)*n*m,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        float ref = (1.0 * 2.0)*k;
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
			if (std::abs(hc[i*m + j] - ref) > 1e-4) {
			    std::cout << i << "." << j << ": " << hc[i*m+j] << ";    ";
			}

                }
        }
        std::cout << std::endl;
}
