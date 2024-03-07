#include <iostream>
#include "cuda_runtime.h"
#include <cmath>

#include <cstdlib>  // for rand
#include <ctime> // for time()


// compile: nvcc basic.cu

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

	 __shared__ float tileA[BLOCK_DIM][BLOCK_DIM];
	 __shared__ float tileB[BLOCK_DIM][BLOCK_DIM];

         int tile_row = threadIdx.y;
         int tile_col = threadIdx.x;

         float res = 0; // thread local

        for (int k = 0; k < K; k+=block_k) {
                // fetch to shared memory
                float a=0;
                float b=0;

                int y = row*M + k + tile_col;
                int x = (k+tile_row)*N + col;
                if (y < M*K) {
                        a = da[y];
                }
                if (x < N*K) {
                        b = db[x];
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


__global__ void tiled_mm_double_buffer(float* da, float* db, float* dc, int block_m, int block_n, int block_k, int M, int N, int K) {
        // FIXME!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tileA[2][BLOCK_DIM][BLOCK_DIM];
    __shared__ float tileB[2][BLOCK_DIM][BLOCK_DIM];

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    float res = 0; // thread local

    int n = K / block_k; // Assuming K is perfectly divisible by block_k for simplicity
    for (int m = 0; m < n; m++) {
        int k = m * block_k;
        int buff_idx = m % 2; // Determine current buffer index (0 or 1)

        // Calculate global indices to load into shared memory
        int y = row * M + k + tile_col;
        int x = (k + tile_row) * N + col;

        if (y < M * K) {
            tileA[buff_idx][tile_row][tile_col] = da[y];
        }
        if (x < N * K) {
            tileB[buff_idx][tile_row][tile_col] = db[x];
        }

        __syncthreads(); // Make sure all data is loaded into shared memory

        // Use the other buffer for computation to overlap loading and computation
        int compute_idx = (m - 1) % 2;
        if (m > 0) { // Ensure we don't compute in the first iteration
            for (int kk = 0; kk < block_k; ++kk) {
                float a = tileA[compute_idx][tile_row][kk];
                float b = tileB[compute_idx][kk][tile_col];
                res += a * b;
            }
        }
        __syncthreads(); // Ensure computation is done before next load
    }

    // Perform computation for the last set of tiles
    for (int kk = 0; kk < block_k; ++kk) {
        float a = tileA[n % 2][tile_row][kk];
        float b = tileB[n % 2][kk][tile_col];
        res += a * b;
    }

    if (row < M && col < N) {
        dc[row * M + col] = res;
    }
}


int main(int argc, char *argv[]) {
	int arg = 0;
	if (argc > 1) {
		arg = std::atoi(argv[1]);
	}

        srand(time(0));

        int m = 1024;
        int n = 1024;
        int k = 1024;

        // allocate input/output buffer
        float *ha = (float *)malloc(sizeof(float) * m * k);
        float *hb = (float *)malloc(sizeof(float)*n*k);
        float *hc = (float *)malloc(sizeof(float)*m*n);
        float *ref_c = (float *)malloc(sizeof(float)*m*n);

        float *da;
        float *db;
        float *dc;
        cudaMalloc((void**)&db, sizeof(float)*n*k);
        cudaMalloc((void **)&da, sizeof(float)*m*k);
        cudaMalloc((void**)&dc, sizeof(float)*m*n);

        // init and transfer memory (assume row-major)
	float min = 0.0f;
	float max = 1.0f;
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                        ha[i*m + j] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                }
        }
        for (int i = 0; i < n; ++i) {
                for (int j = 0; j < k; ++j) {
                        hb[i*n + j] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                }
        }
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        hc[i*m + j] = 0.0;
                }
        }
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        float c = 0;
			for (int kk = 0; kk < k; ++kk) {
                                c += ha[i*m+kk] * hb[kk*n+j];
			}
	                ref_c[i*m + j] = c;
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
        std::cout << "arg: " << arg <<  std::endl;

        if (arg == 0)
                native_mm<<<grid, block>>>(da, db, dc, block_m, block_n, block_k, m, n, k);
        else if (arg == 1)
                tiled_mm<<<grid, block>>>(da, db, dc, block_m, block_n, block_k, m, n, k);
        else if (arg == 2)
                tiled_mm_double_buffer<<<grid, block>>>(da, db, dc, block_m, block_n, block_k, m, n, k);
        cudaDeviceSynchronize();

        // test output
        cudaMemcpy(hc, dc, sizeof(float)*n*m,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
			if (std::abs(hc[i*m + j] - ref_c[i*m+j]) > 1e-4) {
			    std::cout << i << "." << j << ": " << hc[i*m+j] << " " << ref_c[i*m+j] << ";    ";
			}

                }
        }
        std::cout << std::endl;
}
