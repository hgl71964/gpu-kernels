#include <iostream>
#include "cuda_runtime.h"

#include <cstdlib>  // for rand
#include <ctime> // for time()


// c++ interface for async global -> shared
// ptx: cp.async
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>




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



__global__ void tile_mm_sync(float *A, float *B, float *C, const int M, const int N, const int K) {
    extern __shared__ float shared_storage[];
    float *sa0 = reinterpret_cast<float *>(shared_storage);
    const int block_k = 8;
    float *sa1 = sa0 + 8*8;
    float *sa2 = sa1 + 8*8;
    float *sa3 = sa2 + 8*8;
    float *sb0 = sa3 + 8*8;
    float *sb1 = sb0 + 8*8;
    float *sb2 = sb1 + 8*8;
    float *sb3 = sb2 + 8*8;

    int row_start = blockIdx.x * blockDim.x;
    int col_start = blockIdx.y * blockDim.y;


    // prologue
    int n_pipeline = 4;

    //stage 0
    sa0[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + 0];
    sb0[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+0)*N+col_start+threadIdx.y];

    //stage 1
    sa1[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + (1)*block_k];
    sb1[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+(1)*block_k)*N+col_start+threadIdx.y];

    //stage 2
    sa2[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + (2)*block_k];
    sb2[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+(2)*block_k)*N+col_start+threadIdx.y];

    float accu = 0.0f;
    int n_iter = K/block_k; 
    // static_assert(K % block_k==0); // why const does not work 
    assert(K%block_k==0); 

    for (int iter=0; iter<n_iter; iter+=n_pipeline) {

        // stage 3
        if (iter+3<n_iter) {
            sa3[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + (iter+3)*block_k];
            sb3[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+(iter+3)*block_k)*N+col_start+threadIdx.y];
        }
        // accu stage 0
        for (int kk = 0; kk < block_k; ++kk) {
            accu += sa0[threadIdx.x*8 + kk] * sb0[kk*8 + threadIdx.y];
        }
        __syncthreads(); 


        // stage 0
        if (iter+4<n_iter) {
            sa0[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + (iter+4)*block_k];
            sb0[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+(iter+4)*block_k)*N+col_start+threadIdx.y];
        }
        // accu stage 1
        for (int kk = 0; kk < block_k; ++kk) {
            accu += sa1[threadIdx.x*8 + kk] * sb1[kk*8 + threadIdx.y];
        }
        __syncthreads();

        // stage 1
        if (iter+5<n_iter) {
            sa1[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + (iter+5)*block_k];
            sb1[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+(iter+5)*block_k)*N+col_start+threadIdx.y];
        }
        // accu stage 2
        for (int kk = 0; kk < block_k; ++kk) {
            accu += sa2[threadIdx.x*8 + kk] * sb2[kk*8 + threadIdx.y];
        }
        __syncthreads();

        // stage 2
        if (iter+6<n_iter) {
            sa2[threadIdx.x*8 + threadIdx.y] = A[(row_start+threadIdx.x)*K+threadIdx.y + (iter+6)*block_k];
            sb2[threadIdx.x*8 + threadIdx.y] = B[(threadIdx.x+(iter+6)*block_k)*N+col_start+threadIdx.y];
        }
        // accu stage 3
        for (int kk = 0; kk < block_k; ++kk) {
            accu += sa3[threadIdx.x*8 + kk] * sb3[kk*8 + threadIdx.y];
        }
        __syncthreads();
    }

    // epilogue; in this shape, nothing left for epilogue

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        C[row*N+col] = accu;
    }
}






///////
//////// ASYNC
///////

__device__ void loadSmemA_async(cooperative_groups::thread_block TB, float* sa, float* A, int row_start, const int K, const int stage_offset) {
    cooperative_groups::memcpy_async(TB, sa, A+row_start*K + stage_offset , sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+8, A+(row_start+1)*K + stage_offset, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+16, A+(row_start+2)*K + stage_offset, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+24, A+(row_start+3)*K + stage_offset, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+32, A+(row_start+4)*K + stage_offset, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+40, A+(row_start+5)*K + stage_offset, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+48, A+(row_start+6)*K + stage_offset, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sa+56, A+(row_start+7)*K + stage_offset, sizeof(float)*8);
}

__device__ void loadSmemB_async(cooperative_groups::thread_block TB, float* sb, float* B, int col_start, const int N, const int stage_offset) {
    cooperative_groups::memcpy_async(TB, sb, B+(0+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+8, B+(1+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+16, B+(2+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+24, B+(3+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+32, B+(4+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+40, B+(5+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+48, B+(6+stage_offset)*N+col_start, sizeof(float)*8);
    cooperative_groups::memcpy_async(TB, sb+56, B+(7+stage_offset)*N+col_start, sizeof(float)*8);
}

__global__ void tile_mm_async(float *A, float *B, float *C, const int M, const int N, const int K) {
    extern __shared__ float shared_storage[];
    float *sa0 = reinterpret_cast<float *>(shared_storage);
    const int block_k = 8;
    float *sa1 = sa0 + 8*8;
    float *sa2 = sa1 + 8*8;
    float *sa3 = sa2 + 8*8;
    float *sb0 = sa3 + 8*8;
    float *sb1 = sb0 + 8*8;
    float *sb2 = sb1 + 8*8;
    float *sb3 = sb2 + 8*8;

    int row_start = blockIdx.x * blockDim.x;
    int col_start = blockIdx.y * blockDim.y;
    auto TB = cooperative_groups::this_thread_block();

    // prologue
    int n_pipeline=4;

    //stage 0
    {
        cooperative_groups::memcpy_async(TB, sa0, A+row_start*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+8, A+(row_start+1)*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+16, A+(row_start+2)*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+24, A+(row_start+3)*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+32, A+(row_start+4)*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+40, A+(row_start+5)*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+48, A+(row_start+6)*K, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sa0+56, A+(row_start+7)*K, sizeof(float)*8);

        cooperative_groups::memcpy_async(TB, sb0, B+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+8, B+(1)*N+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+16, B+(2)*N+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+24, B+(3)*N+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+32, B+(4)*N+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+40, B+(5)*N+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+48, B+(6)*N+col_start, sizeof(float)*8);
        cooperative_groups::memcpy_async(TB, sb0+56, B+(7)*N+col_start, sizeof(float)*8);

    }

    //stage 1
    loadSmemA_async(TB, sa1, A, row_start, K, block_k);
    loadSmemB_async(TB, sb1, B, col_start, N, block_k);

    //stage 2
    loadSmemA_async(TB, sa2, A, row_start, K, 2*block_k);
    loadSmemB_async(TB, sb2, B, col_start, N, 2*block_k);


    float accu = 0.0f;
    int n_iter = K/block_k; 
    // static_assert(K % block_k==0); // why const does not work 
    assert(K%block_k==0); 

    for (int iter = 0; iter < n_iter; iter+=n_pipeline) {

        // sync all for now
        //cooperative_groups::wait(TB); 
        cooperative_groups::wait_prior<2>(TB); // sth may be subtly wrong, should maintain a on-fly async copy and wait

        // stage 3
        if (iter+3<n_iter) {
            loadSmemA_async(TB, sa3, A, row_start, K, (iter+3)*block_k);
            loadSmemB_async(TB, sb3, B, col_start, N, (iter+3)*block_k);
        }

        // accu stage 0
        for (int kk = 0; kk < block_k; ++kk) {
            //accu += sa[threadIdx.x][kk] * sb[kk][threadIdx.y];
            accu += sa0[threadIdx.x*8 + kk] * sb0[kk*8 + threadIdx.y];
        }
        __syncthreads();


        // stage 0
        //cooperative_groups::wait(TB); 
        cooperative_groups::wait_prior<2>(TB);
        if (iter+4<n_iter){
            loadSmemA_async(TB, sa0, A, row_start, K, (iter+4)*block_k);
            loadSmemB_async(TB, sb0, B, col_start, N, (iter+4)*block_k);
        }
        //accu stage 1
        for (int kk = 0; kk < block_k; ++kk)
            accu += sa1[threadIdx.x*8 + kk] * sb1[kk*8 + threadIdx.y];
        __syncthreads();

        // stage 1
        //cooperative_groups::wait(TB); 
        cooperative_groups::wait_prior<2>(TB);
        if (iter+5<n_iter) {
            loadSmemA_async(TB, sa1, A, row_start, K, (iter+5)*block_k);
            loadSmemB_async(TB, sb1, B, col_start, N, (iter+5)*block_k);
        }
        //accu stage 2
        for (int kk = 0; kk < block_k; ++kk)
            accu += sa2[threadIdx.x*8 + kk] * sb2[kk*8 + threadIdx.y];
        __syncthreads();

        // stage 2
        //cooperative_groups::wait(TB); 
        cooperative_groups::wait_prior<2>(TB);
        if (iter+6<n_iter) {
            loadSmemA_async(TB, sa2, A, row_start, K, (iter+6)*block_k);
            loadSmemB_async(TB, sb2, B, col_start, N, (iter+6)*block_k);
        }
        //accu stage 3
        for (int kk = 0; kk < block_k; ++kk)
            accu += sa3[threadIdx.x*8 + kk] * sb3[kk*8 + threadIdx.y];
        __syncthreads();
    }

    // epilogue
    {
        // in this shape, nothing left for epilogue?
    }

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        C[row*N+col] = accu;
    }
}


//////////
////////// MAIN
//////////

int main() {
    srand(time(0));

    const int M = 128;
    const int N = 128;
    const int K = 128;

    const int block_m = 8;
    const int block_n = 8;

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

    size_t smem = 4*2*block_m*block_n*4; // 4 pipeline stages * 2 smem * element * f32 = num of bytes
     //tile_mm_sync<<<grid, block, smem, nullptr>>>(da, db, dc, M, N, K);
     tile_mm_async<<<grid, block, smem, nullptr>>>(da, db, dc, M, N, K);

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