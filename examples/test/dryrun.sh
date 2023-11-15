#$ _NVVM_BRANCH_=nvvm
#$ _SPACE_= 
#$ _CUDART_=cudart
#$ _HERE_=/usr/local/cuda/bin
#$ _THERE_=/usr/local/cuda/bin
#$ _TARGET_SIZE_=
#$ _TARGET_DIR_=
#$ _TARGET_DIR_=targets/x86_64-linux
#$ TOP=/usr/local/cuda/bin/..
#$ NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
#$ LD_LIBRARY_PATH=/usr/local/cuda/bin/../lib:
#$ PATH=/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/home/gh512/.local/bin:/home/gh512/.local/bin:/home/gh512/.cargo/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda/bin:/snap/bin
#$ INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"  
#$ LIBRARIES=  "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"
#$ CUDAFE_FLAGS=
#$ PTXAS_FLAGS=
#$ gcc -D__CUDA_ARCH_LIST__=800 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=52 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "kernel.cu" -o "/tmp/tmpxft_002f43ef_00000000-5_kernel.cpp4.ii" 
#$ cudafe++ --c++17 --gnu_version=110400 --display_error_number --orig_src_file_name "kernel.cu" --orig_src_path_name "/home/gh512/workspace/gpu-kernel/examples/test/kernel.cu" --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_002f43ef_00000000-6_kernel.cudafe1.cpp" --stub_file_name "tmpxft_002f43ef_00000000-6_kernel.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_002f43ef_00000000-4_kernel.module_id" "/tmp/tmpxft_002f43ef_00000000-5_kernel.cpp4.ii" 
#$ gcc -D__CUDA_ARCH__=800 -D__CUDA_ARCH_LIST__=800 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=52 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "kernel.cu" -o "/tmp/tmpxft_002f43ef_00000000-9_kernel.cpp1.ii" 
#$ cicc --c++17 --gnu_version=110400 --display_error_number --orig_src_file_name "kernel.cu" --orig_src_path_name "/home/gh512/workspace/gpu-kernel/examples/test/kernel.cu" --allow_managed   -arch compute_80 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_002f43ef_00000000-3_kernel.fatbin.c" -tused --module_id_file_name "/tmp/tmpxft_002f43ef_00000000-4_kernel.module_id" --gen_c_file_name "/tmp/tmpxft_002f43ef_00000000-6_kernel.cudafe1.c" --stub_file_name "/tmp/tmpxft_002f43ef_00000000-6_kernel.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_002f43ef_00000000-6_kernel.cudafe1.gpu"  "/tmp/tmpxft_002f43ef_00000000-9_kernel.cpp1.ii" -o "/tmp/tmpxft_002f43ef_00000000-6_kernel.ptx"
#$ ptxas -arch=sm_80 -m64  "/tmp/tmpxft_002f43ef_00000000-6_kernel.ptx"  -o "/tmp/tmpxft_002f43ef_00000000-10_kernel.sm_80.cubin" 
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=80,file=/tmp/tmpxft_002f43ef_00000000-10_kernel.sm_80.cubin" "--image3=kind=ptx,sm=80,file=/tmp/tmpxft_002f43ef_00000000-6_kernel.ptx" --embedded-fatbin="/tmp/tmpxft_002f43ef_00000000-3_kernel.fatbin.c" 
#$ rm /tmp/tmpxft_002f43ef_00000000-3_kernel.fatbin
#$ gcc -D__CUDA_ARCH__=800 -D__CUDA_ARCH_LIST__=800 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_002f43ef_00000000-6_kernel.cudafe1.cpp" -o "/tmp/tmpxft_002f43ef_00000000-11_kernel.o" 
#$ nvlink -m64 --arch=sm_80 --register-link-binaries="/tmp/tmpxft_002f43ef_00000000-7_kernel_dlink.reg.c"    "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_002f43ef_00000000-11_kernel.o"  -lcudadevrt  -o "/tmp/tmpxft_002f43ef_00000000-12_kernel_dlink.sm_80.cubin" --host-ccbin "gcc"
#$ fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=80,file=/tmp/tmpxft_002f43ef_00000000-12_kernel_dlink.sm_80.cubin" --embedded-fatbin="/tmp/tmpxft_002f43ef_00000000-8_kernel_dlink.fatbin.c" 
#$ rm /tmp/tmpxft_002f43ef_00000000-8_kernel_dlink.fatbin
#$ gcc -D__CUDA_ARCH_LIST__=800 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_002f43ef_00000000-8_kernel_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_002f43ef_00000000-7_kernel_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=52 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda/bin/crt/link.stub" -o "/tmp/tmpxft_002f43ef_00000000-13_kernel_dlink.o" 
#$ g++ -D__CUDA_ARCH_LIST__=800 -m64 -Wl,--start-group "/tmp/tmpxft_002f43ef_00000000-13_kernel_dlink.o" "/tmp/tmpxft_002f43ef_00000000-11_kernel.o"   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "kernel" 
