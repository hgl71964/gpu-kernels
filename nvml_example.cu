#include <nvml.h>
#include <stdio.h>


// nvcc -o nvml nvml_example.cu -lnvidia-ml
int main() {
  nvmlInit();
  unsigned int deviceCount = 0;
  nvmlDeviceGetCount(&deviceCount);
  printf("Number of GPUs: %d\n", deviceCount);
  nvmlShutdown();
  return 0;
}
