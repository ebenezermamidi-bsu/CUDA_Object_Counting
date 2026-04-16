#include "region_analysis.h"

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

bool analyzeRegions(
    const GrayImage& binaryImage,
    int minArea,
    RegionAnalysisResult& result,
    std::string& error
) {
    if (binaryImage.width <= 0 || binaryImage.height <= 0 || binaryImage.pixels.empty()) {
        error = "Binary image is empty.";
        return false;
    }

    const int width = binaryImage.width;
    const int height = binaryImage.height;
    const int total = width * height;

    std::vector<int> labels(total, 0);
    std::vector<RegionStat> allRegions;
    int nextLabel = 1;

    auto inside = [&](int x, int y) {
        return x >= 0 && x < width && y >= 0 && y < height;
    };

    const int dirs[8][2] = {
        {-1,-1}, {0,-1}, {1,-1},
        {-1, 0},          {1, 0},
        {-1, 1}, {0, 1}, {1, 1}
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (binaryImage.pixels[idx] == 0 || labels[idx] != 0) {
                continue;
            }

            RegionStat stat;
            stat.label = nextLabel;
            stat.area = 0;
            stat.minX = stat.maxX = x;
            stat.minY = stat.maxY = y;

            std::queue<int> q;
            q.push(idx);
            labels[idx] = nextLabel;

            while (!q.empty()) {
                int current = q.front();
                q.pop();

                int cx = current % width;
                int cy = current / width;

                stat.area++;
                stat.minX = std::min(stat.minX, cx);
                stat.minY = std::min(stat.minY, cy);
                stat.maxX = std::max(stat.maxX, cx);
                stat.maxY = std::max(stat.maxY, cy);

                for (const auto& d : dirs) {
                    int nx = cx + d[0];
                    int ny = cy + d[1];
                    if (!inside(nx, ny)) continue;
                    int nidx = ny * width + nx;
                    if (binaryImage.pixels[nidx] == 0) continue;
                    if (labels[nidx] != 0) continue;

                    labels[nidx] = nextLabel;
                    q.push(nidx);
                }
            }

            allRegions.push_back(stat);
            nextLabel++;
        }
    }

    result.regions.clear();
    result.objectCount = 0;
    result.labeledPreview.width = width;
    result.labeledPreview.height = height;
    result.labeledPreview.pixels.assign(total, 0);

    std::vector<uint8_t> labelToValue(nextLabel + 1, 0);
    int displayValue = 40;

    for (const auto& stat : allRegions) {
        if (stat.area < minArea) continue;
        result.regions.push_back(stat);
        result.objectCount++;

        labelToValue[stat.label] = static_cast<uint8_t>(displayValue);
        displayValue += 35;
        if (displayValue > 240) displayValue = 60;
    }

    for (int i = 0; i < total; ++i) {
        int label = labels[i];
        if (label >= 0 && label < static_cast<int>(labelToValue.size())) {
            result.labeledPreview.pixels[i] = labelToValue[label];
        }
    }

    for (const auto& stat : result.regions) {
        for (int x = stat.minX; x <= stat.maxX; ++x) {
            result.labeledPreview.pixels[stat.minY * width + x] = 255;
            result.labeledPreview.pixels[stat.maxY * width + x] = 255;
        }
        for (int y = stat.minY; y <= stat.maxY; ++y) {
            result.labeledPreview.pixels[y * width + stat.minX] = 255;
            result.labeledPreview.pixels[y * width + stat.maxX] = 255;
        }
    }

    return true;
}
