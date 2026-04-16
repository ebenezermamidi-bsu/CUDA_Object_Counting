#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

bool loadPGM(const std::string& path, GrayImage& image, std::string& error);
bool savePGM(const std::string& path, const GrayImage& image, std::string& error);
