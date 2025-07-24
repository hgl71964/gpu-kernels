#include <iostream>
#include <cuda_runtime.h>

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

int cdiv(int len, int block_size) {
        return (len + block_size -1)/block_size;
}


template<int BM, int BN, int BK>
__global__ void gemm(float *A, float *B, float *C, int M, int N, int K) {

        int row = blockIdx.x * BM + threadIdx.x/BN;
        int col = blockIdx.y * BN + threadIdx.x%BN;

        // int row = blockIdx.x * BM + threadIdx.x%BM;
        // int col = blockIdx.y * BN + threadIdx.x/BM;

        float accu = 0;
        if (row<M&&col<N) {
                for (int k = 0; k < K ; ++k) {
                        accu += A[row*K + k] * B[k*N + col];
                }
                C[row*N + col] = accu;
        }
}

template<int BM, int BN, int BK>
__global__ void gemm_tile(float *A, float *B, float *C, int M, int N, int K) {
        __shared__ float SA[BM][BK];
        __shared__ float SB[BK][BN];

        // NOTE: must be enough thr to load!!!
        static_assert(blockDim.x >= BM * BK);
        static_assert(blockDim.x >= BK * BN);

        float accu = 0;
        for (int k = 0; k < K; k+=BK) {
                if (threadIdx.x < BM * BK) {
                        int row = threadIdx.x / BK;
                        int col = threadIdx.x % BK;
                        if (row+blockIdx.x*BM < M && k+col < K)
                                SA[row][col] = A[(blockIdx.x * BM + row)*K + k + col];
                        else
                                SA[row][col] = 0;
                } 
                if (threadIdx.x < BK * BN) {
                        int row = threadIdx.x/BN; 
                        int col = threadIdx.x%BN;

                        if (col + blockIdx.y*BN < N && k+row < K)
                                SB[row][col] = B[(k+row)*N + col + blockIdx.y*BN];
                        else
                                SB[row][col] = 0;
                }
                __syncthreads();

                for (int kk = 0 ; kk < BK; ++kk) {
                        int row = threadIdx.x/BN;
                        int col = threadIdx.x%BN;
                        accu += SA[row][kk] * SB[kk][col];
                }
                __syncthreads();
        }

        int row = blockIdx.x * BM + threadIdx.x/BN;
        int col = blockIdx.y * BN + threadIdx.x%BN;
        if (row<M&&col<N) {
                C[row*N + col] = accu;
        }
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_thread_blocking(float *A, float *B, float *C, int M, int N, int K) {
        // here TM and TN means works per thread along M and N
        __shared__ float SA[BM][BK];
        __shared__ float SB[BK][BN];

        float RA[TM] = {0};
        float RB[TN] = {0};
        float RC[TM][TN] = {0};

        // NOTE: must be enough thr to load!!!
        static_assert(BM/TM * BN/TN >= (BM/TM) * BK);
        static_assert(BM/TM * BN/TN >= BK * BN/TN);

        for (int k = 0 ; k < K; k+=BK) {
                if (threadIdx.x < (BM/TM) * BK) {
                        for (int ti = 0 ; ti < TM; ++ti) {
                                int row = threadIdx.x / BK + ti * BM/TM;
                                int col = threadIdx.x % BK ;
                                if (row + blockIdx.x * BM < M && k + col < K)
                                        SA[row][col] = A[(blockIdx.x * BM + row)*K + k + col];
                                else
                                        SA[row][col] = 0;
                        }
                }
                if (threadIdx.x < BK * (BN/TN)) {
                        for (int tj = 0 ; tj < TN; ++tj) {
                                int row = threadIdx.x / (BN/TN);
                                int col = threadIdx.x % (BN/TN) + tj * BN/TN;
                                if (k + row < K && col + blockIdx.y * BN < N)
                                        SB[row][col] = B[(k+row)*N + col + blockIdx.y * BN];
                                else
                                        SB[row][col] = 0;
                        }
                }
                __syncthreads();

                for (int kk = 0 ; kk < BK; ++kk) {
                        int row = threadIdx.x / (BN/TN);
                        int col = threadIdx.x % (BN/TN);

                        #pragma unroll
                        for (int ti = 0 ; ti < TM; ++ti) {
                                RA[ti] = SA[row + ti * BM/TM][kk];
                        }
                        #pragma unroll
                        for (int tj = 0; tj < TN; ++tj) {
                                RB[tj] = SB[kk][col+tj*BN/TN];
                        }

                        #pragma unroll
                        for (int ti = 0 ; ti < TM; ++ti) {
                                #pragma unroll
                                for (int tj = 0 ; tj < TN; ++tj) {
                                        RC[ti][tj] += RA[ti] * RB[tj];
                                }
                        }
                }
                __syncthreads();
        }

        int row = threadIdx.x / (BN/TN);
        int col = threadIdx.x % (BN/TN);
        for (int ti = 0 ; ti < TM; ++ti) {
                for (int tj = 0 ; tj < TN; ++tj) {
                        if (row + ti * BM/TM + blockIdx.x * BM < M && col + tj * BN/TN + blockIdx.y * BN < N)
                                C[(blockIdx.x * BM + row + ti * BM/TM)*N + blockIdx.y * BN + col + tj * BN/TN] = RC[ti][tj];
                }
        }
}

template<int BM, int BN, int BK, int TM, int TN, int STAGE=2>
__global__ void gemm_thread_blocking_double_buffer(float *A, float *B, float *C, int M, int N, int K) {
        // here TM and TN means works per thread along M and N
        __shared__ float SA[STAGE][BM][BK];
        __shared__ float SB[STAGE][BK][BN];

        float RA[TM] = {0};
        float RB[TN] = {0};
        float RC[TM][TN] = {0};
        static_assert(STAGE == 2);

        // NOTE: must be enough thr to load!!!
        static_assert(BM/TM * BN/TN >= (BM/TM) * BK);
        static_assert(BM/TM * BN/TN >= BK * BN/TN);

        // prologue
        int k = 0;
        int stage = 0;
        {
                if (threadIdx.x < (BM/TM) * BK) {
                        for (int ti = 0 ; ti < TM; ++ti) {
                                int row = threadIdx.x / BK + ti * BM/TM;
                                int col = threadIdx.x % BK ;
                                if (row + blockIdx.x * BM < M && k + col < K)
                                        SA[stage][row][col] = A[(blockIdx.x * BM + row)*K + k + col];
                                else
                                        SA[stage][row][col] = 0;
                        }
                }
                if (threadIdx.x < BK * (BN/TN)) {
                        for (int tj = 0 ; tj < TN; ++tj) {
                                int row = threadIdx.x / (BN/TN);
                                int col = threadIdx.x % (BN/TN) + tj * BN/TN;
                                if (k + row < K && col + blockIdx.y * BN < N)
                                        SB[stage][row][col] = B[(k+row)*N + col + blockIdx.y * BN];
                                else
                                        SB[stage][row][col] = 0;
                        }
                }
        }
        k += BK;
        stage++;
        stage %= STAGE;
        int comp_stage = stage-1;

        for (; k < K; k+=BK) {
                __syncthreads();
                if (threadIdx.x < (BM/TM) * BK) {
                        for (int ti = 0 ; ti < TM; ++ti) {
                                int row = threadIdx.x / BK + ti * BM/TM;
                                int col = threadIdx.x % BK ;
                                if (row + blockIdx.x * BM < M && k + col < K)
                                        SA[stage][row][col] = A[(blockIdx.x * BM + row)*K + k + col];
                                else
                                        SA[stage][row][col] = 0;
                        }
                }
                if (threadIdx.x < BK * (BN/TN)) {
                        for (int tj = 0 ; tj < TN; ++tj) {
                                int row = threadIdx.x / (BN/TN);
                                int col = threadIdx.x % (BN/TN) + tj * BN/TN;
                                if (k + row < K && col + blockIdx.y * BN < N)
                                        SB[stage][row][col] = B[(k+row)*N + col + blockIdx.y * BN];
                                else
                                        SB[stage][row][col] = 0;
                        }
                }

                for (int kk = 0 ; kk < BK; ++kk) {
                        int row = threadIdx.x / (BN/TN);
                        int col = threadIdx.x % (BN/TN);

                        #pragma unroll
                        for (int ti = 0 ; ti < TM; ++ti) {
                                RA[ti] = SA[comp_stage][row + ti * BM/TM][kk];
                        }
                        #pragma unroll
                        for (int tj = 0; tj < TN; ++tj) {
                                RB[tj] = SB[comp_stage][kk][col+tj*BN/TN];
                        }

                        #pragma unroll
                        for (int ti = 0 ; ti < TM; ++ti) {
                                #pragma unroll
                                for (int tj = 0 ; tj < TN; ++tj) {
                                        RC[ti][tj] += RA[ti] * RB[tj];
                                }
                        }
                }
                stage++;
                stage %= STAGE;
                comp_stage++;
                comp_stage %= STAGE;
        }

        // epilogue
        {
                __syncthreads();
                for (int kk = 0 ; kk < BK; ++kk) {
                        int row = threadIdx.x / (BN/TN);
                        int col = threadIdx.x % (BN/TN);

                        #pragma unroll
                        for (int ti = 0 ; ti < TM; ++ti) {
                                RA[ti] = SA[comp_stage][row + ti * BM/TM][kk];
                        }
                        #pragma unroll
                        for (int tj = 0; tj < TN; ++tj) {
                                RB[tj] = SB[comp_stage][kk][col+tj*BN/TN];
                        }

                        #pragma unroll
                        for (int ti = 0 ; ti < TM; ++ti) {
                                #pragma unroll
                                for (int tj = 0 ; tj < TN; ++tj) {
                                        RC[ti][tj] += RA[ti] * RB[tj];
                                }
                        }
                }
        }

        int row = threadIdx.x / (BN/TN);
        int col = threadIdx.x % (BN/TN);
        for (int ti = 0 ; ti < TM; ++ti) {
                for (int tj = 0 ; tj < TN; ++tj) {
                        if (row + ti * BM/TM + blockIdx.x * BM < M && col + tj * BN/TN + blockIdx.y * BN < N)
                                C[(blockIdx.x * BM + row + ti * BM/TM)*N + blockIdx.y * BN + col + tj * BN/TN] = RC[ti][tj];
                }
        }
}

int main() {
        const int M = 512;
        const int N = 256;
        const int K = 128;

        float *ha =  (float *) malloc(sizeof(float)*M*K);
        float *hb =  (float *) malloc(sizeof(float)*N*K);
        float *hc =  (float *) malloc(sizeof(float)*M*N);
        float *gold =  (float *) malloc(sizeof(float)*M*N);

        for (int i = 0; i < M*K; ++i){
                ha[i] = 1;
        }
        for (int i = 0; i < N*K; ++i){
                hb[i] = 2;
        }
        for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                        float accu = 0;
                        for (int k = 0; k < K; ++k) {
                                // row major
                                accu += ha[i * K + k] * hb[k*N + j];
                        }
                        gold[i*N + j] = accu;
                        //gold[i*N + j] = 0;
                }
        }

        float *da;
        float *db;
        float *dc;
        cudaMalloc((void**)&da, sizeof(float)*M*K);
        cudaMalloc((void**)&db, sizeof(float)*N*K);
        cudaMalloc((void**)&dc, sizeof(float)*M*N);

        cudaMemcpy(da, ha, sizeof(float)*M*K,cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, sizeof(float)*N*K,cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc, sizeof(float)*M*N,cudaMemcpyHostToDevice);

        // const int BM = 32;
        // const int BN = 16;
        // const int BK = 16;

        // dim3 grid(cdiv(M,BM), cdiv(N,BN), 1);
        // dim3 block(BM * BN, 1);
        // std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        // std::cout << "block: " << block.x << " " << block.y << " " << block.z << std::endl;
        // // gemm<BM,BN,BK><<<grid, block>>>(da, db, dc, M, N, K);
        // gemm_tile<BM,BN,BK><<<grid, block>>>(da, db, dc, M, N, K);


        const int TM = 2;  // this means each thread compute 2 along M dim
        const int TN = 2;
        const int BM = 64;
        const int BN = 32;
        const int BK = 16;

        dim3 grid(cdiv(M,BM), cdiv(N,BN), 1);
        dim3 block((BM/TM) * (BN/TN), 1);
        std::cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        std::cout << "block: " << block.x << " " << block.y << " " << block.z << std::endl;
        // gemm_thread_blocking<BM, BN, BK, TM, TN><<<grid, block>>>(da, db, dc, M, N, K);
        gemm_thread_blocking_double_buffer<BM, BN, BK, TM, TN><<<grid, block>>>(da, db, dc, M, N, K);



        cudaMemcpy(hc, dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);
        cudaErrCheck(cudaDeviceSynchronize());

        int err = 0;
        for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                        if (fabs(gold[i*N + j] - hc[i*N + j]) > 1e-4 ) {
                                err++;
                        }
                }
        }
        std::cout << "Err: " << err << std::endl;
}
