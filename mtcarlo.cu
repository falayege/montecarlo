#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp properties;

    // Get device ID of the first CUDA device
    cudaGetDevice(&device);

    // Get properties of the device
    cudaGetDeviceProperties(&properties, device);

    std::cout << "Device Name: " << properties.name << std::endl;
    std::cout << "Compute capability: " << properties.major << "." << properties.minor << std::endl;
    std::cout << "Max Threads per Block: " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "Number of SMs: " << properties.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << properties.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Shared Memory per Block: " << properties.sharedMemPerBlock << " bytes" << std::endl;

    return 0;
}
