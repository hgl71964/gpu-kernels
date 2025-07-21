
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

int cdiv(int len, int block_size) {
        return (len + block_size -1)/block_size;
}


__global__ void transpose(float *in, float *out, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    out[col*M + row] = in[row*N + col];
}

template <int BM, int BN, int TM_PER_THR>
__global__ void transpose_thr_blocking(float *in, float *out, int M, int N) {
    // assign each thr more work to interleave with idx cost
    int row_start = blockIdx.x * BM;
    int col_start = blockIdx.y * BN;

    int row = threadIdx.x/(BN);
    int col = threadIdx.x%(BN);
    for (int i = 0; i < TM_PER_THR; ++i) {
        if (row_start + row + i*BM/TM_PER_THR < M && col_start + col < N) 
                out[(col_start + col)*M + row_start + row + i*BM/TM_PER_THR] = in[(row_start + row + i*BM/TM_PER_THR)*N + col_start + col];
    }
}

template <int BLOCK_SIZE, int TM_PER_THR>
__global__ void transpose_shmem(float *in, float *out, int M, int N) {

        __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE];

    // assign each thr more work to interleave with idx cost
    int row_start = blockIdx.x * BLOCK_SIZE;
    int col_start = blockIdx.y * BLOCK_SIZE;

    int row = threadIdx.x/(BLOCK_SIZE);
    int col = threadIdx.x%(BLOCK_SIZE);
    for (int i = 0; i < TM_PER_THR; ++i) {
        if (row_start + row + i*BLOCK_SIZE/TM_PER_THR < M && col_start + col < N) 
                smem[row + i*BLOCK_SIZE/TM_PER_THR][col] = in[(row_start + row + i*BLOCK_SIZE/TM_PER_THR)*N + col_start + col];
    }
    __syncthreads();

    // normal copy
//     for (int i = 0; i < TM_PER_THR; ++i) {
//         if (row_start + row+i*BLOCK_SIZE/TM_PER_THR < M && col_start + col < N) {
//                 out[(col_start+col)*M + row_start+row+i*BLOCK_SIZE/TM_PER_THR] = smem[row+i*BLOCK_SIZE/TM_PER_THR][col];
//         }
//     }

    for (int i = 0; i < TM_PER_THR; ++i) {
        int trow = threadIdx.x%BLOCK_SIZE;
        int tcol = threadIdx.x/BLOCK_SIZE;
        if (col_start + row + i*BLOCK_SIZE/TM_PER_THR < N && row_start + col < M) {
                out[(col_start + row + i*BLOCK_SIZE/TM_PER_THR)*M+ row_start + col ] = smem[trow][tcol+i*BLOCK_SIZE/TM_PER_THR];
        }
    }
}


int main() {
        const int M = 128;
        const int N = 64;

        float *hc =  (float *) malloc(sizeof(float)*M*N);
        float *out =  (float *) malloc(sizeof(float)*M*N);
        float *gold =  (float *) malloc(sizeof(float)*M*N);

        for (int k = 0 ; k < M * N; ++k) {
            hc[k] = k;
            out[k] = 0;
        }

        // row major
        for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    gold[j*M+i] = hc[i*N+j];
                }
        }

        // for (int i = 0; i < M; ++i) {
        //         for (int j = 0; j < N; ++j) {
        //             std::cout << hc[i*N+j] << ", " << gold[i*N+j] << std::endl;
        //         }
        // }

        float *dout;
        float *dc;
        cudaMalloc((void**)&dout, sizeof(float)*M*N);
        cudaMalloc((void**)&dc, sizeof(float)*M*N);
        cudaMemcpy(dout, out, sizeof(float)*M*N,cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc, sizeof(float)*M*N,cudaMemcpyHostToDevice);

        {
        // const int BM = 16;
        // const int BN = 32;
        // dim3 block(BM, BN, 1);
        // dim3 grid(cdiv(M, BM), cdiv(N, BN), 1);
        // std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        // std::cout << "block: " << block.x << " " << block.y << " " << block.z << std::endl;
        // transpose<<<grid, block>>>(dc, dout, M, N);
        }


        {
        // const int BM = 64;
        // const int BN = 32;
        // const int TM_PER_THR = 8;
        // dim3 grid(cdiv(M, BM), cdiv(N, BN), 1);
        // dim3 block(BM/TM_PER_THR * BN, 1);
        // std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        // std::cout << "block: " << block.x << " " << block.y << " " << block.z << std::endl;
        // transpose_thr_blocking<BM, BN, TM_PER_THR><<<grid, block>>>(dc, dout, M, N);
        }


        {
        const int BLOCK = 32;
        const int TM_PER_THR = 4;
        dim3 grid(cdiv(M, BLOCK), cdiv(N, BLOCK), 1);
        dim3 block(BLOCK/TM_PER_THR * BLOCK, 1);
        std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        std::cout << "block: " << block.x << " " << block.y << " " << block.z << std::endl;
        transpose_shmem<BLOCK, TM_PER_THR><<<grid, block>>>(dc, dout, M, N);
        }


        cudaMemcpy(out, dout, sizeof(float)*M*N,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        int err = 0;
        for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                        if (std::abs(gold[i*N + j] - out[i*N + j]) > 1e-4 ) {
                                err++;
                        }
                }
        }
        std::cout << "Err: " << err << std::endl;
}
