
#include <iostream>

#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <cublas_v2.h>  // for reference

// compile: nvcc -arch sm_80  cp_async.cu -lcublas

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
#define cublasErrCheck(stat)                         \
    {                                                \
        cublasErrCheck_((stat), __FILE__, __LINE__); \
    }
void cublasErrCheck_(cublasStatus_t stat, const char* file, int line)
{
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define cudaAssert(condition) \
	if (!(condition)){ printf("Assertion %s failed!\n", #condition); asm("trap;"); }





const int MI = 16;
//const int NI = 16;
const int KI = 16;
const int MII = 16;
const int NII = 16;
//const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__device__ void loadSmemA(half *smem, half *A, int M, int K)
{
    int tid = threadIdx.x;
    int row = tid%2;
    int col = tid/2;

    // shared memory layout
    // assume A is col_major and is 16x16, half-precision
    // we have 32 threads, each thread loads 16x16x16/32 = 128 bits = 16 bytes 
    //
    // t0, t1 -> col 0 ; t2, t3 -> col 1, ...
    // 
    void *ptr = (void *)(smem + 8 * row + 16 * col);
    uint32_t smem_ptr;

    asm(
        "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    // LDGSTS
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                    "l"(&A[row * 8 + 16 * col]),
                    "n"(16));

}

__device__ void loadSmemB(half *smem, half *B, int N, int K)
{
    int tid = threadIdx.x;
    int row = tid%2;
    int col = tid/2;

    // shared memory layout
    void *ptr = (void *)(smem + 8 * row + 16 * col);
    uint32_t smem_ptr;

    asm(
        "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    // LDGSTS
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                    "l"(&B[row * 8 + 16 * col]),
                    "n"(16));
}

__device__ void storeSmemC(float *C, float *smem, int M, int N)
{
    int tid = threadIdx.x;
    // we have 16x16=256 elements and 32 threads
    for (int i = 0; i < 8; ++i) {
        int lane_id = tid % 32;  
        int off = i * 32 + lane_id;
        C[off] = smem[off];
    }

}

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> *frag, half *smem, int ki)
{
    // int tid = threadIdx.x;
    // int row = tid%2;
    // int col = tid/2;
    //nvcuda::wmma::load_matrix_sync(frag[0], smem + row * 8 + 16 * col , 16);
    nvcuda::wmma::load_matrix_sync(frag[0], smem, 16);
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> *frag, half *smem, int ki)
{
    //int tid = threadIdx.x;
    //int row = tid%2;
    //int col = tid/2;
    //nvcuda::wmma::load_matrix_sync(frag[0], smem + row * 8 + 16 * col , 16);
    nvcuda::wmma::load_matrix_sync(frag[0], smem, 16);
}

__device__ void storeAccum(float *ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> *frag)
{
    // int tid = threadIdx.x;
    // int row = tid%2;
    // int col = tid/2;
    //nvcuda::wmma::store_matrix_sync(ptr + row * 8 + 16 * col,
    //                            frag[0],
    //                            16,
    //                            nvcuda::wmma::mem_col_major);
    nvcuda::wmma::store_matrix_sync(ptr,
                                frag[0],
                                16,
                                nvcuda::wmma::mem_col_major);
}

__global__ void cp_async_wmma(half *A, half *B, float *C, int M, int N, int K)
{
    // assume A, B col_major

    extern __shared__ uint8_t shared_storage[];
    half *SA1 = reinterpret_cast<half *>(shared_storage);
    half *SB1 = SA1 + MI * KI;

    float *SC = reinterpret_cast<float *>(shared_storage);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragA[MII / wmmaM];  // 1

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII / wmmaN];  // 1

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];

    nvcuda::wmma::fill_fragment(Accum[0], 0.0);

    loadSmemA(SA1, A, M, K);
    loadSmemB(SB1, B, N, K);
    asm volatile("cp.async.commit_group;\n" ::);

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();

    loadFragA(FragA, SA1, 0);
    loadFragB(FragB, SB1, 0);

    // 16x16x16 for each wmma
    nvcuda::wmma::mma_sync(Accum[0], FragA[0], FragB[0], Accum[0]);

    //// multiple steps
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);

    //// directly to global
    //nvcuda::wmma::store_matrix_sync(C, Accum[0], 16, nvcuda::wmma::mem_col_major);
}


/////
/////
/////


int main(int argc, char *argv[]) {
	//int arg = 0;
	//if (argc > 1) {
	//	arg = std::atoi(argv[1]);
	//}

        srand(time(0));

        int m = 16;
        int n = 16;
        int k = 16;

        // allocate input/output buffer
        half *ha = (half *)malloc(sizeof(half)*m*k);
        half *hb = (half *)malloc(sizeof(half)*n*k);
        float *hc = (float *)malloc(sizeof(float)*m*n);
        float *ref_c = (float *)malloc(sizeof(float)*m*n);

        half *da;
        half *db;
        float *dc;
        float *cublas_dc;
        cudaMalloc((void**)&db, sizeof(half)*n*k);
        cudaMalloc((void **)&da, sizeof(half)*m*k);
        cudaMalloc((void**)&dc, sizeof(float)*m*n);
        cudaMalloc((void**)&cublas_dc, sizeof(float)*m*n);

        // init and transfer memory (assume row-major)
	float min = 0.0f;
	float max = 10.0f;
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                        float v = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                        ha[i*k + j] = __float2half_ru(v);
                        //printf("%f %f %f;;  ", v, ha[i*k+j], __half2float(ha[i*k+j]));
                        //printf("%f %f ;;  ", v,  __half2float(ha[i*k+j]));
                }
        }
        for (int i = 0; i < k; ++i) {
                for (int j = 0; j < n; ++j) {
                        float v = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                        hb[i*n + j] = __float2half_ru(v);
                }
        }
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        hc[i*n + j] = 0.0;
                        ref_c[i*n + j] = 0.0;
                }
        }
        cudaErrCheck(cudaMemcpy(da, ha, sizeof(half)*m*k,cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(db, hb, sizeof(half)*n*k, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(dc, hc, sizeof(float)*n*m, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(cublas_dc, ref_c, sizeof(float)*n*m, cudaMemcpyHostToDevice));

        // kernel param
        // int block_m = 1;
        // int block_n = 1;
        // int block_k = 1;

        // int grid_x = cdiv(m, block_m);
        // int grid_y = cdiv(n, block_n);

        // dim3 grid(grid_x, grid_y, 1);
        // dim3 block(block_m, block_n, 1);
        dim3 grid(1, 1, 1);
        dim3 block(32, 1, 1);  // wmma must have at least 32 threads!

        // launch
        std::cout << "GEMM: " << m << "; " << n << "; " << k << std::endl;
        // std::cout << "grid: " << grid_x << "; " << grid_y << std::endl;
        // std::cout << "block: " << block_m << "; " << block_n << std::endl;
        // std::cout << "arg: " << arg <<  std::endl;

        size_t smem = 2*16*16*16/8;  // num of bytes

        cp_async_wmma<<<grid, block, smem, nullptr>>>(da, db, dc, m, n, k);
        cudaErrCheck(cudaDeviceSynchronize());
        cudaErrCheck(cudaMemcpy(hc, dc, sizeof(float)*n*m,cudaMemcpyDeviceToHost));

        // cublas for ref
        cublasHandle_t handle;
        cublasErrCheck(cublasCreate(&handle));
        cublasErrCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); // use tensor core
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasErrCheck(cublasGemmEx(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,
             //CUBLAS_OP_T, CUBLAS_OP_T,
             m, n, k,
             &alpha,
             da, CUDA_R_16F, m,
             db, CUDA_R_16F, k,
             &beta,
             cublas_dc, CUDA_R_32F, m,
             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaErrCheck(cudaMemcpy(ref_c, cublas_dc, sizeof(float)*m*n, cudaMemcpyDeviceToHost));

        int errors = 0;
        printf("checking :\n\n");
        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        float v1 = hc[i*n+j];
                        float v2 = ref_c[i*n+j];
                        float diff = fabs(v1 - v2);
                        float relative_err = diff / v2;
                        float eps = 1e-4;
                        if ((relative_err >= eps)) {
                                errors++;
                                if (errors < 10)
                                        printf("%f %f\n", v1, v2);
                        }
                        // printf("%f %f\n", v1, v2);
                }
        }
        if (errors > 0) 
            printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
        else
            printf("OK\n");
}
