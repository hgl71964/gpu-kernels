
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"


// nvcc -I../third_party/cutlass/include -I../third_party/cutlass/tools/util/include xx.cu

using T = int;
//using T = float;

__global__ void kernel(T *d, int n){
    using namespace cute;
    printf("grid: %d, block: %d\n\n", blockIdx.x, threadIdx.x);

    
  // Tensor A = make_tensor(make_gmem_ptr(d), make_shape(4, 8), make_stride(8, 1));
  Tensor A = make_tensor(make_gmem_ptr(d), make_shape(4, 8), make_stride(1, 4));
  printf("A: "); print_tensor(A); printf("\n");

  auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                          Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

  // Compose A with the tv_layout to transform its shape and order
  Tensor tv = composition(A, tv_layout);                           
  printf("compose tv: "); print_tensor(tv); printf("\n");

  // Slice so each thread has 4 values in the shape and order that the tv_layout prescribes
  Tensor  v0 = tv(0, _);                                  // (4)
  printf("v0: "); print_tensor(v0); printf("\n");

  Tensor  v1 = tv(1, _);                                  // (4)
  printf("v1: "); print_tensor(v1); printf("\n");
}


int main(int argc, char** argv)
{
  cute::device_init(0);
  const int n = 32;

  // init
  thrust::host_vector<T> h(n);
  for (int j = 0; j < n; ++j) 
    h[j] = j;
  thrust::device_vector<T> d = h;

  dim3 grid(1);
  dim3 block(1);
  kernel<<<grid, block>>>(d.data().get(), n);
  CUTE_CHECK_LAST(); // cu device sync
}

