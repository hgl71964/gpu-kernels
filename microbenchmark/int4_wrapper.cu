#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include <cuda.h>

// compile: nvcc int4_wrapper.cu  -lcuda -lcudart -o wrapper

// Function to read the contents of a file into a string
std::string read_ptx(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// std::string rf(const char* filename) {
//     // Step 2: Load PTX from file
//     std::ifstream ptxFile(filename, std::ios::in);
//     if (!ptxFile) {
//         std::cerr << "Error opening PTX file." << std::endl;
//         exit(EXIT_FAILURE);
//     }
//     std::string ptx((std::istreambuf_iterator<char>(ptxFile)), std::istreambuf_iterator<char>());
//     ptxFile.close();
//     return ptx;
// }

size_t read_cubin(const char *filename, unsigned char **buffer) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 0;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    *buffer = new unsigned char[fileSize];
    file.read((char *)*buffer, fileSize);
    file.close();

    return fileSize;
}


int main() {
    const char* kernel_name = "_Z14copyInt4KernelPK4int4PS_i";

    const char* filename = "cubin";
    CUresult result;
    CUfunction fun;
    CUmodule mod;

    // device and context
    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to initialize CUDA." << std::endl;
    }
    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to get device." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to create context." << std::endl;
        exit(EXIT_FAILURE);
    }

    // PTX
    // std::string ptxCode = read_ptx(filename);
    // std::string ptxCode = rf(ptxFile);

    // std::cout << ptxCode;
    // std::cout << std::endl;
    // const char* ptx = ptxCode.c_str();

    // result = cuModuleLoadData(&mod, ptx);

    // if (result != CUDA_SUCCESS) {
    // std::cerr << "Failed to load PTX/Cubin module." << std::endl;
    //     return EXIT_FAILURE;
    // }


    // CUBIN
    unsigned char *cubinBuffer;
    size_t cubinBufferSize = read_cubin(filename, &cubinBuffer);
    if (cubinBufferSize == 0) {
        std::cerr << "Failed to read CUBIN file." << std::endl;
        return 1; // Error reading file
    }
    result = cuModuleLoadData(&mod, cubinBuffer);
    if (result != CUDA_SUCCESS) {
    std::cerr << "Failed to load PTX/Cubin module." << std::endl;
        return EXIT_FAILURE;
    }


    // GET KERNEL
    result = cuModuleGetFunction(&fun, mod, kernel_name);
    if (result != CUDA_SUCCESS) {
        std::cerr << "Failed to get kernel function." << std::endl;
        return EXIT_FAILURE;
    }


    // alloc
    int numElements = 2;

    // Allocate memory on the host
    int *h_src = (int*)malloc(numElements * sizeof(int4));
    int *h_dst = (int*)malloc(numElements * sizeof(int4));

    // Initialize the source array with some values
    for (int i = 0; i < numElements * 4; i++) {
        h_src[i] = i;
    }

    // Allocate memory on the device
    int4 *d_src = nullptr;
    int4 *d_dst = nullptr;
    cudaMalloc((void**)&d_src, numElements * sizeof(int4));
    cudaMalloc((void**)&d_dst, numElements * sizeof(int4));

    // Copy the source array to device memory
    cudaMemcpy(d_src, h_src, numElements * sizeof(int4), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 2;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("thread count: %d; block count: %d\n", threadsPerBlock, blocksPerGrid);

    // launch
    dim3 grid(blocksPerGrid, 1, 1);
    dim3 block(threadsPerBlock, 1, 1);
    void *args[] = {&d_src, &d_dst, &numElements};

    auto err = cuLaunchKernel(fun,
                        grid.x, grid.y, grid.z,
                        block.x, block.y, block.z,
                        0, // Shared memory
                        0, // Stream
                        args,
                        0);
    cudaDeviceSynchronize();
    if (err != CUDA_SUCCESS) {
        printf("Failed to launch kernel.\n");
        return;
    }

    // Copy the result back to the host
    cudaMemcpy(h_dst, d_dst, numElements * sizeof(int4), cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < numElements * 4; i++) {
          printf("idx: %i, src: %d, dst: %d \t", i, h_src[i], h_dst[i]);
        if (h_src[i] != h_dst[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Copy successful!\n");
    } else {
        printf("Copy failed!\n");
    }

}
