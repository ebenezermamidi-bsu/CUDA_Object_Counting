#pragma once

#include "image_io.h"
#include <string>

struct PipelineConfig {
    int threshold = 140;
    int erodeIterations = 1;
    int dilateIterations = 1;
};

struct PipelineResult {
    GrayImage mask;
    GrayImage cleaned;
    double gpuTimeMs = 0.0;
};

bool runGpuPipeline(
    const GrayImage& input,
    const PipelineConfig& config,
    PipelineResult& result,
    std::string& error
);
