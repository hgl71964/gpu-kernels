#include <iostream>
#include "cuda_runtime.h"
#include <cmath>

#include <cstdlib>  // for rand
#include <ctime> // for time()


// compile: nvcc basic.cu

#define BLOCK_DIM 8


#define cudaAssert(condition) \
	if (!(condition)){ printf("Assertion %s failed!\n", #condition); asm("trap;"); }


int cdiv(int length, int block_size) {
        return (length + block_size - 1)/block_size;
}

__global__ void native_mm(float* da, float* db, float* dc, int block_m, int block_n, int block_k, int M, int N, int K) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < M && col < N) {
                float res = 0;
                for (int k = 0; k < K; ++k) {
                        float a = da[row*K + k];
                        float b = db[k*N + col];
                        res += a*b;
                }
                dc[row*N + col] = res;
        }
}

__global__ void tiled_mm(float* da, float* db, float* dc, int block_m, int block_n, int block_k, int M, int N, int K) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

         // must match block_m, block_n, block_k
	 __shared__ float tileA[BLOCK_DIM][BLOCK_DIM];
	 __shared__ float tileB[BLOCK_DIM][BLOCK_DIM];

         int tile_row = threadIdx.x;
         int tile_col = threadIdx.y;

         float res = 0; // thread local

        for (int k = 0; k < K; k+=block_k) {
                // fetch to shared memory
                float a=0;
                float b=0;

                int x = row*K + (k + tile_col);
                int y = (k + tile_row)*N + col;
                if (x < M*K) {
                        a = da[x];
                }
                if (y < N*K) {
                        b = db[y];
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
                dc[row*N + col] = res;
        }
}


__global__ void tiled_mm_double_buffer(float* da, float* db, float* dc, int block_m, int block_n, int block_k, int M, int N, int K) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// Double buffering: two sets of tiles for A and B
	__shared__ float tileA[2][BLOCK_DIM][BLOCK_DIM];
	__shared__ float tileB[2][BLOCK_DIM][BLOCK_DIM];

	int tile_row = threadIdx.x;
	int tile_col = threadIdx.y;

	float res = 0;

	int n_iter = K / block_k; // Assuming K is divisible by block_k for simplicity
	cudaAssert(n_iter%2==0);
	for (int t = 0; t < n_iter; t++) {
	    int k = t * block_k;
	    int buff_idx = t % 2; // Current buffer index
	    int next_buff_idx = (t + 1) % 2; // Next buffer index

	    // Load current tile
	    int x = row * K + (k + tile_col);
	    int y = (k + tile_row) * N + col;
	    if (x < M * K) {
		tileA[buff_idx][tile_row][tile_col] = da[x];
	    } else {
		tileA[buff_idx][tile_row][tile_col] = 0;
	    }
	    if (y < N * K) {
		tileB[buff_idx][tile_row][tile_col] = db[y];
	    } else {
		tileB[buff_idx][tile_row][tile_col] = 0;
	    }

	    // Load next tile in advance if within bounds
	    if (t < n_iter - 1) {
		x = row * K + ((t + 1) * block_k + tile_col);
		y = ((t + 1) * block_k + tile_row) * N + col;
		if (x < M * K) {
		    tileA[next_buff_idx][tile_row][tile_col] = da[x];
		} else {
		    tileA[next_buff_idx][tile_row][tile_col] = 0;
		}
		if (y < N * K) {
		    tileB[next_buff_idx][tile_row][tile_col] = db[y];
		} else {
		    tileB[next_buff_idx][tile_row][tile_col] = 0;
		}
	    }

	    __syncthreads();

	    // Compute using current tile
	    for (int kk = 0; kk < block_k; ++kk) {
		float a = tileA[buff_idx][tile_row][kk];
		float b = tileB[buff_idx][kk][tile_col];
		res += a * b;
	    }

	    __syncthreads();
	}

	if (row < M && col < N) {
	    dc[row * N + col] = res;
	}
}



int main(int argc, char *argv[]) {
	int arg = 0;
	if (argc > 1) {
		arg = std::atoi(argv[1]);
	}

        srand(time(0));

        int m = 1024;
        int n = 512;
        int k = 128;

        // allocate input/output buffer
        float *ha = (float *)malloc(sizeof(float)*m*k);
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
                        ha[i*k + j] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                }
        }
        for (int i = 0; i < k; ++i) {
                for (int j = 0; j < n; ++j) {
                        hb[i*n + j] = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                }
        }
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        hc[i*n + j] = 0.0;
                }
        }
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        float c = 0;
			for (int kk = 0; kk < k; ++kk) {
                                c += ha[i*k+kk] * hb[kk*n+j];
			}
	                ref_c[i*n + j] = c;
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
        std::cout << "GEMM: " << m << "; " << n << "; " << k << std::endl;
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

        int err=0;
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
			if (std::abs(hc[i*n+j] - ref_c[i*n+j]) > 1e-4) {
			    std::cout << i << "." << j << ": " << hc[i*n+j] << " " << ref_c[i*n+j] << std::endl;
                            err++;
			}
                        if (err > 10)
                                break;
                }
                if (err > 10)
                        break;
        }
}
