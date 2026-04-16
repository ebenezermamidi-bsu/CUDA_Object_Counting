#pragma once

#include <string>
#include <vector>

std::vector<std::string> listFilesWithExtension(const std::string& directory, const std::string& extension);
bool ensureDirectory(const std::string& path);
std::string stemFromPath(const std::string& path);
bool writeTextFile(const std::string& path, const std::string& content);
