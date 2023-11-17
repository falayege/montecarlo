#include <iostream>
#include <cmath>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

// Kernel for simulating asset paths and computing option payoff
__global__ void simulateOptionPaths(float *payoffs, float initialPrice, float strikePrice, float maturity, float volatility, float riskFreeRate, int paths) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < paths) {
        curandState_t state;
        curand_init(1234, idx, 0, &state);
        float dt = maturity / 100.0f; // Assuming 100 time steps
        float S = initialPrice;

        for (int i = 0; i < 100; ++i) {
            float gauss_bm = curand_normal(&state);
            S *= exp((riskFreeRate - 0.5f * volatility * volatility) * dt + volatility * sqrtf(dt) * gauss_bm);
        }

        payoffs[idx] = exp(-riskFreeRate * maturity) * max(S - strikePrice, 0.0f);
    }
}

int main() {
    int paths = 1000000;
    float *d_payoffs;

    // Option parameters
    float initialPrice = 100.0f;
    float strikePrice = 100.0f;
    float maturity = 1.0f;
    float volatility = 0.2f;
    float riskFreeRate = 0.05f;

    // Allocate array on GPU
    cudaMalloc((void**)&d_payoffs, paths * sizeof(float));

    // Define workspace topology
    int blockSize = 256;
    int gridSize = (paths + blockSize - 1) / blockSize;

    // Execute kernel
    simulateOptionPaths<<<gridSize, blockSize>>>(d_payoffs, initialPrice, strikePrice, maturity, volatility, riskFreeRate, paths);

    // Copy result back to host
    float *h_payoffs = new float[paths];
    cudaMemcpy(h_payoffs, d_payoffs, paths * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-processing: Calculate average payoff
    float sum = 0.0f;
    for (int i = 0; i < paths; ++i) {
        sum += h_payoffs[i];
    }
    float averagePayoff = sum / paths;

    std::cout << "Average Option Price: " << averagePayoff << std::endl;

    // Clean up
    cudaFree(d_payoffs);
    delete[] h_payoffs;

    return 0;
}
