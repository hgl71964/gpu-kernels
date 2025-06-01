
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"


// nvcc -I../third_party/cutlass/include -I../third_party/cutlass/tools/util/include layout_vis.cu

using T = int;
//using T = float;

__global__ void kernel1(T *d, int n){
    using namespace cute;
    printf("grid: %d, block: %d\n\n", blockIdx.x, threadIdx.x);

    Layout la = make_layout(make_shape (2,4),
                            make_stride(4, 1));

    Tensor tA = make_tensor(make_gmem_ptr(d), la);
    printf("tA: "); print_tensor(tA); 
    printf("iter: ");
    for (int i = 0; i < n; ++i) {
        printf("tA[%d]: %d\t", i, tA(i));
    }
    printf("\n");
    printf("\n");

    Layout lb = make_layout(make_shape (4, 2),
                            make_stride(1, 4));
    Tensor tB = make_tensor(make_gmem_ptr(d), lb);
    printf("tB: "); print_tensor(tB); 
    printf("iter: ");
    for (int i = 0; i < n; ++i) {
        printf("tB[%d]: %d\t", i, tB(i));
    }
    printf("\n");
    printf("\n");

    printf("hierarchical layout:\n");

    auto lc = make_layout(make_shape (make_shape(2, 2), 2),
                          make_stride(make_stride(4, 1), 2));
    Tensor tC = make_tensor(make_gmem_ptr(d), lc);
    printf("tC: "); print_tensor(tC); 
    printf("iter: ");
    for (int i = 0; i < n; ++i) {
        printf("tC[%d]: %d\t", i, tC(i));
    }
    printf("\n");
    printf("\n");

    auto ld = make_layout(make_shape (make_shape(2, 2), make_shape(2, 1)),
                          make_stride(make_stride(4, 1), make_stride(2, 1)));
    Tensor tD = make_tensor(make_gmem_ptr(d), ld);
    printf("tD: "); print_tensor(tD);

    // iter
    printf("iter: ");
    for (int i = 0; i < n; ++i) {
        printf("tD[%d]: %d\t", i, tD(i));
    }
    printf("\n");
    printf("\n");

    // auto le = make_layout(make_shape (make_shape(2, 2), make_shape(2, 1)),
    //                       make_stride(make_stride(2,1), 4));
    // Tensor tE = make_tensor(make_gmem_ptr(d), le);
    // printf("tE: "); print_tensor(tE); printf("\n");

    auto l5 = make_layout(make_shape (make_shape(2, 2), make_shape(1,2)),
                          make_stride(make_stride(4, 1), make_stride(1,2)));
    Tensor t5 = make_tensor(make_gmem_ptr(d), l5);
    printf("t5: "); print_tensor(t5); printf("\n");

    // iter
    printf("iter: ");
    for (int i = 0; i < n; ++i) {
        printf("t5[%d]: %d\t", i, t5(i));
    }
    printf("\n");
    printf("\n");

    auto l6 = make_layout(make_shape (make_shape(2, 2), 2),
                          make_stride(make_stride(1,2),4));
    Tensor t6 = make_tensor(make_gmem_ptr(d), l6);
    printf("t6: "); print_tensor(t6); printf("\n");
}

__global__ void kernel2(T *d, int n){
    using namespace cute;
    printf("grid: %d, block: %d\n\n", blockIdx.x, threadIdx.x);

    auto l1 = make_layout(make_shape (make_shape(2, 2), make_shape(1,2)),
                          make_stride(make_stride(4, 1), make_stride(1,2)));
}

int main(int argc, char** argv)
{
  cute::device_init(0);
  const int n = 2*4;

  // init
  thrust::host_vector<T> h(n);
  for (int j = 0; j < n; ++j) 
    h[j] = j;
  thrust::device_vector<T> d = h;

  dim3 grid(1);
  dim3 block(1);
  kernel1<<<grid, block>>>(d.data().get(), n);
  CUTE_CHECK_LAST(); // cu device sync
}

