
#include "helpers.h"
#include <cublasLt.h>

// compile: nvcc -arch sm_89 LtFp8Matmul.cu  -lcublasLt



/// Sample wrapper executing fp8 matmul with cublasLtMatmul, with addition of per-tensor scaling, amax calculations, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtFp8Matmul(cublasLtHandle_t ltHandle,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    const float *a_scale, /* device pointer */
    const __nv_fp8_e4m3 *A,
    int lda,
    const float *b_scale, /* device pointer */
    const __nv_fp8_e4m3 *B,
    int ldb,
    const float *c_scale, /* device pointer */
    __nv_fp8_e4m3 *D,
    int ldc,
    const float *d_scale, /* device pointer */
    float *amax_d, /* device pointer */
    void *workspace,
    size_t workspaceSize) {

cublasLtMatmulDesc_t operationDesc = NULL;
cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
cublasLtMatmulPreference_t preference = NULL;

cublasOperation_t transa = CUBLAS_OP_T;
cublasOperation_t transb = CUBLAS_OP_N;
float beta = 0.0; // Can be non-zero starting from 12.0

int returnedResults                             = 0;
cublasLtMatmulHeuristicResult_t heuristicResult = {};

// create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
// set the transforms for A and B
checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

// set scaling factors
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amax_d, sizeof(amax_d)));

// create matrix descriptors, we are good with the details here so no need to set any extra attributes
// table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_8F_E4M3, m, n, ldc));

// create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
// will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
// directly come from cudaMalloc)
checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

// we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
// is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

if (returnedResults == 0) {
checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
}

checkCublasStatus(cublasLtMatmul(ltHandle,
                        operationDesc,
                        alpha,
                        A,
                        Adesc,
                        B,
                        Bdesc,
                        &beta,
                        nullptr,
                        Cdesc,
                        D,
                        Ddesc,
                        &heuristicResult.algo,
                        workspace,
                        workspaceSize,
                        0));

// descriptors are no longer needed as all GPU work was already enqueued
if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

}

int main() {
    TestBench<__nv_fp8_e4m3, __nv_fp8_e4m3, float> props(64, 128, 256, 2.0f, 0.0f /* ignored */, 32ULL * 1024 * 1024);

    props.run([&props] {
        LtFp8Matmul(props.ltHandle,
                    props.m,
                    props.n,
                    props.k,
                    &props.alpha,
                    props.AscaleDev,
                    props.Adev,
                    props.k,
                    props.BscaleDev,
                    props.Bdev,
                    props.k,
                    props.CscaleDev,
                    props.Cdev,
                    props.m,
                    props.DscaleDev,
                    props.DamaxDev,
                    props.workspace,
                    props.workspaceSize);
    });

    return 0;
}