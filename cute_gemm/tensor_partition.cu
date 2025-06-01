
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
    
    Tensor A = make_tensor(make_gmem_ptr(d), make_shape(8,24), make_stride(24,1)); 
    printf("A: "); print_tensor(A); printf("\n"); // (8,24)

    //
    // tiling 1 (inner partition), i.e. local_tile
    //

    auto tiler = Shape<_4,_8>{};                    // (_4,_8)
    Tensor tiled_a = zipped_divide(A, tiler);       // ((_4,_8),(2,3))
    printf("tile_a (zipped_divide): "); printf("\n");
    print_tensor(tiled_a); printf("\n");

    // auto slice1 = tiled_a(0, _);
    // printf("slice1: "); print_tensor(slice1); printf("\n");

    Tensor cta_0_0 = tiled_a(make_coord(_,_), make_coord(0, 0));
    printf("cta_0_0: "); print_tensor(cta_0_0); printf("\n");

    Tensor cta_0_1 = tiled_a(make_coord(_,_), make_coord(0, 1));
    printf("cta_0_1: "); print_tensor(cta_0_1); printf("\n");

    //
    // outer partition, i.e. local_partition
    //
    Tensor thr0 = tiled_a(0, make_coord(_,_)); // (2,3)
    printf("thr0: "); print_tensor(thr0); printf("\n");

    Tensor thr1 = tiled_a(1, make_coord(_,_));
    printf("thr1: "); print_tensor(thr1); printf("\n");

    //
    // A's view
    // 
    printf("A: "); print_tensor(A); printf("\n"); // (8,24)

    //
    // per-mode tiling
    //
    auto tiler2 = make_tile(Layout<_2,_4>{},  // Apply 3:4 to mode-0
                           Layout<_8,_2>{}); // Apply 8:2 to mode-1
    Tensor per_mode_tile = composition(A, tiler2);
    printf("per_mode_tile: "); print_tensor(per_mode_tile); printf("\n");

}


int main(int argc, char** argv)
{
  cute::device_init(0);
  const int n = 192;

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

