#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    cudaError_t error;

    // Get number of devices
    int deviceCount;
    error = cudaGetDeviceCount(&deviceCount);
    checkCudaError(error, "Failed to get CUDA device count");

    printf("Found %d CUDA device(s)\n\n", deviceCount);

    // Iterate through devices and get properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, i);
        checkCudaError(error, "Failed to get device properties");

        printf("Device %d: %s\n", i, deviceProp.name);
        printf("  CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
               deviceProp.major, deviceProp.minor,
               CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

        printf("  CUDA Capability Major/Minor version number: %d.%d\n",
               deviceProp.major, deviceProp.minor);

        printf("  Total Global Memory: %.2f GB\n",
               static_cast<float>(deviceProp.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));

        printf("  GPU Clock rate: %.0f MHz\n",
               deviceProp.clockRate * 1e-3f);

        printf("  Memory Clock rate: %.0f MHz\n",
               deviceProp.memoryClockRate * 1e-3f);

        printf("  Memory Bus Width: %d-bit\n",
               deviceProp.memoryBusWidth);

        printf("  Number of multiprocessors: %d\n",
               deviceProp.multiProcessorCount);

        printf("\n");
    }

    // Get CUDA version
    int driverVersion = 0;
    int runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    return 0;
}
