
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"

// https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/

// nvcc -I../third_party/cutlass/include -I../third_party/cutlass/tools/util/include xx.cu

using T = int;
//using T = float;
const int n = 16;

__global__ void kernel(T *d){
    using namespace cute;
    printf("grid: %d, block: %d\n\n", blockIdx.x, threadIdx.x);

    //
    // A's view
    // 
    
    Tensor gA = make_tensor(make_gmem_ptr(d), make_shape(n , n), make_stride(n,1)); 
    printf("gA: "); print_tensor(gA); printf("\n");


    //
    // tiled divide
    //
  using b = Int<4>;
  auto block_shape = make_shape(b{}, b{});       // (b, b)
  Tensor tiled_tensor_S  = tiled_divide(gA, block_shape); // ([b,b], m/b, n/b)
  printf("tiled_tensor_S: "); print(tiled_tensor_S); printf("\n"); // print_tensor of a divide does not often make sense

  // Tensor slice1 = tiled_tensor_S(make_coord(_, _), 0);
  // printf("slice1: "); print_tensor(slice1); printf("\n");
  Tensor slice2 = tiled_tensor_S(make_coord(_, _), 0,0);
  printf("slice2: "); print_tensor(slice2); printf("\n");

  Tensor slice3 = tiled_tensor_S(make_coord(_, _), 0,1);
  printf("slice3: "); print_tensor(slice3); printf("\n");

  Tensor slice4 = tiled_tensor_S(make_coord(_, _), 1,0);
  printf("slice4: "); print_tensor(slice4); printf("\n");
}


int main(int argc, char** argv)
{
  cute::device_init(0);

  // init
  thrust::host_vector<T> h(n*n);
  for (int j = 0; j < n*n; ++j) 
    h[j] = j;
  thrust::device_vector<T> d = h;

  dim3 grid(1);
  dim3 block(1);
  kernel<<<grid, block>>>(d.data().get());
  CUTE_CHECK_LAST(); // cu device sync
}

