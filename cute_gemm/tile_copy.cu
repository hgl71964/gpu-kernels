
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

    //
    // A's view
    // 
    
    Tensor gA = make_tensor(make_gmem_ptr(d), make_shape(8 ,32), make_stride(32, 1)); 
    printf("A: "); print_tensor(gA); printf("\n"); // (8,24)

    //
    // tile copy
    //

  // TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
  //                                   Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
  //                                   Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copy_a = make_tiled_copy(Copy_Atom<UniversalCopy<T>, T>{},
                                    Layout<Shape<_2,_8>>{}, 
                                    // Layout<Shape< _1,_1>>{});              
                                    Layout<Shape< _2,_1>>{});              
  ThrCopy thr_copy_1 = copy_a.get_slice(0);
  Tensor tAgA_1 = thr_copy_1.partition_S(gA);                          
  printf("tAgA: "); print_tensor(tAgA_1); printf("\n");

  ThrCopy thr_copy_2 = copy_a.get_slice(1);
  Tensor tAgA_2 = thr_copy_2.partition_S(gA);                           
  printf("tAgA_2: "); print_tensor(tAgA_2); printf("\n");

  // ThrCopy thr_copy_3 = copy_a.get_slice(63);
  // Tensor tAgA_3 = thr_copy_3.partition_S(gA);                           
  // printf("tAgA_3: "); print_tensor(tAgA_3); printf("\n");


}


int main(int argc, char** argv)
{
  cute::device_init(0);
  const int n = 256;

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

