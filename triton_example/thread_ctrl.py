import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def tl_kernel(output_ptr,  # *Pointer* to output vector.
              BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
            ):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    off_ptr = output_ptr + block_start
    val = 11
    # scalar pointer, scalar value
    tl.store(off_ptr, val)

@triton.jit
def tl2(output_ptr,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_ptr = output_ptr + block_start
    val = 11
    # off_ptr is a vector of pointers
    tl.store(off_ptr, val)

@triton.jit
def get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )

@triton.jit
def get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )

@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def tl3(output_ptr,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  

    # each thread in the threadblock gets its unique thread id
    tid = get_flat_tid()

    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_ptr = output_ptr + block_start
    tl.store(off_ptr, tid)

@triton.jit
def tl4(output_ptr,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE + tid
    off_ptr = output_ptr + block_start
    # tid is scalar, so off_ptr is a scalar pointer
    # compiler will control only 1 thread to do the store in the threadblock
    tl.store(off_ptr, tid)

@triton.jit
def tl5(output_ptr,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE + tid
    off_ptr = output_ptr + block_start
    # we can bypass the compiler by inline asm
    tl.inline_asm_elementwise(
        """
        st.global.b32 [$1], { $2 };
        """,
        "=r,l,r",
        [off_ptr, tid],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


def main():
    length = 256
    block_size = 32
    out = torch.empty((length, ), device=DEVICE, dtype=torch.int32)
    # k = tl_kernel[(2, )](out, block_size)
    # k = tl2[(2, )](out, block_size)
    # k = tl3[(2, )](out, block_size)
    # k = tl3[(2, )](out, block_size, num_warps=1)
    # k = tl4[(2, )](out, block_size, num_warps=1)
    k = tl5[(2, )](out, block_size, num_warps=1)
    # with open(f'tl3.ptx', 'w') as f:
    #     f.write(k.asm['ptx'])

    # NOTE: this is unsafe, because we can have only 32 threads in a threadblock,
    # but we set block size to 64, the compiler will map each thread to 2 data elements
    # which can product incorrect results

    for i in range(length):
        print(i, out[i].item(), end=', ')
        if (i+1)%32 == 0:
            print()
    print()


if __name__ == "__main__":
    main()