#include "utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

std::vector<std::string> listFilesWithExtension(const std::string& directory, const std::string& extension) {
    std::vector<std::string> files;
    std::filesystem::path dir(directory);
    if (!std::filesystem::exists(dir)) {
        return files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == extension) {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

bool ensureDirectory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::create_directories(path, ec) || std::filesystem::exists(path);
}

std::string stemFromPath(const std::string& path) {
    return std::filesystem::path(path).stem().string();
}

bool writeTextFile(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out) return false;
    out << content;
    return static_cast<bool>(out);
}
