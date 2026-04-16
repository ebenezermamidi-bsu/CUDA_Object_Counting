#include "gpu_pipeline.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <sstream>

namespace {

inline bool checkCuda(cudaError_t status, std::string& error, const char* step) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << step << " failed: " << cudaGetErrorString(status);
        error = oss.str();
        return false;
    }
    return true;
}

__global__ void thresholdKernel(const unsigned char* src, unsigned char* dst, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    dst[idx] = (src[idx] > threshold) ? 255 : 0;
}

__global__ void erode3x3Kernel(const unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char minVal = 255;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = min(max(x + kx, 0), width - 1);
            int yy = min(max(y + ky, 0), height - 1);
            unsigned char v = src[yy * width + xx];
            if (v < minVal) minVal = v;
        }
    }
    dst[y * width + x] = minVal;
}

__global__ void dilate3x3Kernel(const unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char maxVal = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int xx = min(max(x + kx, 0), width - 1);
            int yy = min(max(y + ky, 0), height - 1);
            unsigned char v = src[yy * width + xx];
            if (v > maxVal) maxVal = v;
        }
    }
    dst[y * width + x] = maxVal;
}

bool applyMorphIterations(
    unsigned char* d_a,
    unsigned char* d_b,
    int width,
    int height,
    int iterations,
    bool erode,
    std::string& error
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    unsigned char* src = d_a;
    unsigned char* dst = d_b;

    for (int i = 0; i < iterations; ++i) {
        if (erode) {
            erode3x3Kernel<<<grid, block>>>(src, dst, width, height);
        } else {
            dilate3x3Kernel<<<grid, block>>>(src, dst, width, height);
        }
        if (!checkCuda(cudaGetLastError(), error, erode ? "erode kernel launch" : "dilate kernel launch")) {
            return false;
        }
        if (!checkCuda(cudaDeviceSynchronize(), error, erode ? "erode synchronize" : "dilate synchronize")) {
            return false;
        }
        std::swap(src, dst);
    }

    if (src != d_a) {
        if (!checkCuda(cudaMemcpy(d_a, src, static_cast<size_t>(width) * height, cudaMemcpyDeviceToDevice), error, "device copy")) {
            return false;
        }
    }
    return true;
}

}

bool runGpuPipeline(
    const GrayImage& input,
    const PipelineConfig& config,
    PipelineResult& result,
    std::string& error
) {
    if (input.width <= 0 || input.height <= 0 || input.pixels.empty()) {
        error = "Input image is empty.";
        return false;
    }

    const size_t bytes = static_cast<size_t>(input.width) * static_cast<size_t>(input.height);

    unsigned char* d_src = nullptr;
    unsigned char* d_mask = nullptr;
    unsigned char* d_tmp = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    if (!checkCuda(cudaMalloc(&d_src, bytes), error, "cudaMalloc d_src")) return false;
    if (!checkCuda(cudaMalloc(&d_mask, bytes), error, "cudaMalloc d_mask")) {
        cudaFree(d_src);
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_tmp, bytes), error, "cudaMalloc d_tmp")) {
        cudaFree(d_src);
        cudaFree(d_mask);
        return false;
    }
    if (!checkCuda(cudaEventCreate(&startEvent), error, "cudaEventCreate start")) {
        cudaFree(d_src);
        cudaFree(d_mask);
        cudaFree(d_tmp);
        return false;
    }
    if (!checkCuda(cudaEventCreate(&stopEvent), error, "cudaEventCreate stop")) {
        cudaFree(d_src);
        cudaFree(d_mask);
        cudaFree(d_tmp);
        cudaEventDestroy(startEvent);
        return false;
    }

    auto cleanup = [&]() {
        if (d_src) cudaFree(d_src);
        if (d_mask) cudaFree(d_mask);
        if (d_tmp) cudaFree(d_tmp);
        if (startEvent) cudaEventDestroy(startEvent);
        if (stopEvent) cudaEventDestroy(stopEvent);
    };

    if (!checkCuda(cudaMemcpy(d_src, input.pixels.data(), bytes, cudaMemcpyHostToDevice), error, "copy input to device")) {
        cleanup();
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((input.width + block.x - 1) / block.x, (input.height + block.y - 1) / block.y);

    if (!checkCuda(cudaEventRecord(startEvent), error, "cudaEventRecord start")) {
        cleanup();
        return false;
    }

    thresholdKernel<<<grid, block>>>(d_src, d_mask, input.width, input.height, config.threshold);
    if (!checkCuda(cudaGetLastError(), error, "threshold kernel launch")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaDeviceSynchronize(), error, "threshold synchronize")) {
        cleanup();
        return false;
    }

    result.mask.width = input.width;
    result.mask.height = input.height;
    result.mask.pixels.resize(bytes);

    if (!checkCuda(cudaMemcpy(result.mask.pixels.data(), d_mask, bytes, cudaMemcpyDeviceToHost), error, "copy mask to host")) {
        cleanup();
        return false;
    }

    if (config.erodeIterations > 0) {
        if (!applyMorphIterations(d_mask, d_tmp, input.width, input.height, config.erodeIterations, true, error)) {
            cleanup();
            return false;
        }
    }
    if (config.dilateIterations > 0) {
        if (!applyMorphIterations(d_mask, d_tmp, input.width, input.height, config.dilateIterations, false, error)) {
            cleanup();
            return false;
        }
    }
    if (config.dilateIterations > 0) {
        if (!applyMorphIterations(d_mask, d_tmp, input.width, input.height, config.dilateIterations, false, error)) {
            cleanup();
            return false;
        }
    }
    if (config.erodeIterations > 0) {
        if (!applyMorphIterations(d_mask, d_tmp, input.width, input.height, config.erodeIterations, true, error)) {
            cleanup();
            return false;
        }
    }

    if (!checkCuda(cudaEventRecord(stopEvent), error, "cudaEventRecord stop")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaEventSynchronize(stopEvent), error, "cudaEventSynchronize stop")) {
        cleanup();
        return false;
    }

    float elapsedMs = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), error, "cudaEventElapsedTime")) {
        cleanup();
        return false;
    }
    result.gpuTimeMs = static_cast<double>(elapsedMs);

    result.cleaned.width = input.width;
    result.cleaned.height = input.height;
    result.cleaned.pixels.resize(bytes);

    if (!checkCuda(cudaMemcpy(result.cleaned.pixels.data(), d_mask, bytes, cudaMemcpyDeviceToHost), error, "copy cleaned image to host")) {
        cleanup();
        return false;
    }

    cleanup();
    return true;
}
