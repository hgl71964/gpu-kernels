#include <iostream>
#include "cuda_runtime.h"
#include <cmath>

#include <cstdlib>  // for rand
#include <ctime> // for time()

#include <mma.h>
#include <cuda_fp16.h>  // half precision utility
#include <cublas_v2.h>  // for reference

// compile: nvcc -arch sm_80  wmma.cu -lcublas

// compile -> ptx: nvcc -arch sm_80  -ptx wmma.cu -o wmma.ptx
// (-arch is needed to specify as not all architecture can do mma?)
//


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


__global__ void wmma_kernel(half *a, half *b, float *c, int m, int n, int k) {
    namespace wmma = nvcuda::wmma;  // alias namespace

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   //wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
   //wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   //wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_col_major);
}

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
        dim3 block(32, 1, 1);  // wmma at least 32 threads!

        // launch
        std::cout << "GEMM: " << m << "; " << n << "; " << k << std::endl;
        // std::cout << "grid: " << grid_x << "; " << grid_y << std::endl;
        // std::cout << "block: " << block_m << "; " << block_n << std::endl;
        // std::cout << "arg: " << arg <<  std::endl;

        wmma_kernel<<<grid, block>>>(da, db, dc, m, n, k);
        cudaErrCheck(cudaDeviceSynchronize());

        // test output
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
}
