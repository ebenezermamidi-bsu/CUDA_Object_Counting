#pragma once

#include "image_io.h"
#include <string>
#include <vector>

struct RegionStat {
    int label = 0;
    int area = 0;
    int minX = 0;
    int minY = 0;
    int maxX = 0;
    int maxY = 0;
};

struct RegionAnalysisResult {
    int objectCount = 0;
    std::vector<RegionStat> regions;
    GrayImage labeledPreview;
};

bool analyzeRegions(
    const GrayImage& binaryImage,
    int minArea,
    RegionAnalysisResult& result,
    std::string& error
);
