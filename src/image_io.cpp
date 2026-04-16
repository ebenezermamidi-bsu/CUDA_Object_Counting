#include "image_io.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace {
bool readToken(std::istream& in, std::string& token) {
    token.clear();
    char c = '\0';

    while (in.get(c)) {
        if (c == '#') {
            std::string dummy;
            std::getline(in, dummy);
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(c))) {
            token.push_back(c);
            break;
        }
    }

    if (token.empty()) {
        return false;
    }

    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            break;
        }
        token.push_back(c);
    }
    return true;
}
}

bool loadPGM(const std::string& path, GrayImage& image, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Unable to open file: " + path;
        return false;
    }

    std::string magic, widthStr, heightStr, maxValStr;
    if (!readToken(in, magic) || magic != "P5") {
        error = "Only binary PGM (P5) is supported: " + path;
        return false;
    }
    if (!readToken(in, widthStr) || !readToken(in, heightStr) || !readToken(in, maxValStr)) {
        error = "Invalid PGM header: " + path;
        return false;
    }

    int width = std::stoi(widthStr);
    int height = std::stoi(heightStr);
    int maxVal = std::stoi(maxValStr);

    if (width <= 0 || height <= 0 || maxVal != 255) {
        error = "Unsupported PGM dimensions or max value in: " + path;
        return false;
    }

    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    in.read(reinterpret_cast<char*>(image.pixels.data()), image.pixels.size());
    if (!in) {
        error = "Failed to read pixel data: " + path;
        return false;
    }

    return true;
}

bool savePGM(const std::string& path, const GrayImage& image, std::string& error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "Unable to write file: " + path;
        return false;
    }

    out << "P5\n" << image.width << " " << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels.data()), image.pixels.size());

    if (!out) {
        error = "Failed while writing file: " + path;
        return false;
    }

    return true;
}
