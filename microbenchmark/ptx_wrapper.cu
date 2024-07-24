#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include <cuda.h>

// compile: nvcc ptx_wrapper.cu  -lcuda -lcudart -o wrapper

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
    const char* kernel_name = "_Z3addPiS_S_Px";
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
    int *ha = (int *)malloc(sizeof(int)*100);
    int *hb = (int *)malloc(sizeof(int)*100);
    int *hc = (int *)malloc(sizeof(int)*100);
    long long int  *clock = (long long int  *)malloc(sizeof(long long int )*1);
    int *da;
    int *db;
    int *dc;
    long long int  *dclock;
    cudaMalloc((void**)&db, sizeof(int)*100);
    cudaMalloc((void **)&da, sizeof(int)*100);
    cudaMalloc((void**)&dc, sizeof(int)*100);
    cudaMalloc((void**)&dclock, sizeof(long long int )*1);
    for (int i =0;i<100;++i) {
        ha[i] = 1;
        hb[i] = 1;
        hc[i] = 0;
    }
    cudaMemcpy(da, ha, sizeof(int)*100,cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(int)*100, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, sizeof(int)*100, cudaMemcpyHostToDevice);

    // launch
    dim3 grid(1, 1, 1);
    dim3 block(1, 1, 1);
    void *args[] = {&da, &db, &dc, &dclock};

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

    cudaMemcpy(hc, dc, sizeof(int)*100,cudaMemcpyDeviceToHost);
    cudaMemcpy(clock, dclock, sizeof(long long int )*1,cudaMemcpyDeviceToHost);

    int cnt = 0 ;
    for (int i =0;i<100;++i) {
        printf("i: %d - hc: %d \t", i, hc[i]);
        if (hc[i]==2)
            cnt++;
    }
    printf("\n");

    printf("got %d hc\n", cnt);
    printf("clock: %llu\n", *clock);

    // ENSURE: there's only a sequence of the same instruction within clock64 block
    float avg = static_cast<float>(*clock) / static_cast<float>(cnt);
    printf("average: %f\n", avg);

}