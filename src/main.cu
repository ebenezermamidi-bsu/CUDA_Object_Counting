#include "gpu_pipeline.h"
#include "image_io.h"
#include "region_analysis.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Args {
    std::string inputDir = "data/coco/prepared";
    std::string outputDir = "output";
    int threshold = 140;
    int minArea = 50;
    int erodeIterations = 1;
    int dilateIterations = 1;
    int maxImages = 5000;
    int saveSamples = 7;
};

static void printUsage() {
    std::cout
        << "Usage:\n"
        << "  ./bin/coco_object_counter "
        << "--input-dir <dir> --max-images <int> --output-dir <dir> "
        << "--threshold <int> --min-area <int> "
        << "--erode-iterations <int> --dilate-iterations <int> "
        << "--save-samples <int>\n";
}

static bool parseArgs(int argc, char** argv, Args& args, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needValue = [&](const std::string& option) -> bool {
            if (i + 1 >= argc) {
                error = "Missing value for " + option;
                return false;
            }
            return true;
        };

        if (key == "--input-dir") {
            if (!needValue(key)) return false;
            args.inputDir = argv[++i];
        } else if (key == "--output-dir") {
            if (!needValue(key)) return false;
            args.outputDir = argv[++i];
        } else if (key == "--threshold") {
            if (!needValue(key)) return false;
            args.threshold = std::stoi(argv[++i]);
        } else if (key == "--min-area") {
            if (!needValue(key)) return false;
            args.minArea = std::stoi(argv[++i]);
        } else if (key == "--erode-iterations") {
            if (!needValue(key)) return false;
            args.erodeIterations = std::stoi(argv[++i]);
        } else if (key == "--dilate-iterations") {
            if (!needValue(key)) return false;
            args.dilateIterations = std::stoi(argv[++i]);
        } else if (key == "--max-images") {
            if (!needValue(key)) return false;
            args.maxImages = std::stoi(argv[++i]);
        } else if (key == "--save-samples") {
            if (!needValue(key)) return false;
            args.saveSamples = std::stoi(argv[++i]);
        } else if (key == "--help" || key == "-h") {
            printUsage();
            std::exit(0);
        } else {
            error = "Unknown argument: " + key;
            return false;
        }
    }

    if (args.maxImages < 1) args.maxImages = 1;
    if (args.saveSamples < 0) args.saveSamples = 0;
    return true;
}

static std::string extractSourceKey(const std::string& patchStem) {
    size_t lastUnderscore = patchStem.rfind('_');
    if (lastUnderscore == std::string::npos) return patchStem;

    size_t secondLastUnderscore = patchStem.rfind('_', lastUnderscore - 1);
    if (secondLastUnderscore == std::string::npos) return patchStem;

    return patchStem.substr(0, secondLastUnderscore);
}

int main(int argc, char** argv) {
    Args args;
    std::string error;
    if (!parseArgs(argc, argv, args, error)) {
        std::cerr << error << "\n";
        printUsage();
        return 1;
    }

    ensureDirectory(args.outputDir);
    ensureDirectory(args.outputDir + "/masks");
    ensureDirectory(args.outputDir + "/cleaned");
    ensureDirectory(args.outputDir + "/labeled");
    ensureDirectory(args.outputDir + "/stats");

    auto files = listFilesWithExtension(args.inputDir, ".pgm");
    if (files.empty()) {
        std::cerr << "No .pgm files found in: " << args.inputDir << "\n";
        std::cerr << "Run dataset preparation first.\n";
        return 1;
    }

    if (static_cast<int>(files.size()) > args.maxImages) {
        files.resize(args.maxImages);
    }

    const int totalImages = static_cast<int>(files.size());
    const int sampleInterval = (args.saveSamples > 0)
        ? std::max(1, totalImages / args.saveSamples)
        : totalImages;

    std::cout << "Loaded " << files.size() << " prepared images from " << args.inputDir << "\n";

    std::ofstream summaryCsv(args.outputDir + "/stats/processing_summary.csv");
    if (!summaryCsv) {
        std::cerr << "Unable to create processing summary CSV.\n";
        return 1;
    }
    summaryCsv << "job_id,image_name,width,height,object_count,gpu_time_ms,total_time_ms,status\n";

    std::ofstream objectCsv(args.outputDir + "/stats/object_stats.csv");
    if (!objectCsv) {
        std::cerr << "Unable to create object stats CSV.\n";
        return 1;
    }
    objectCsv << "job_id,image_name,object_count,label,area,min_x,min_y,max_x,max_y\n";

    std::ostringstream runLog;
    runLog << "COCO Object Counter Run Log\n";
    runLog << "Input dir: " << args.inputDir << "\n";
    runLog << "Output dir: " << args.outputDir << "\n";
    runLog << "Threshold: " << args.threshold << "\n";
    runLog << "Min area: " << args.minArea << "\n";
    runLog << "Requested max images: " << args.maxImages << "\n";
    runLog << "Actual images processed: " << files.size() << "\n";
    runLog << "Save samples: " << args.saveSamples << "\n";
    runLog << "Sample interval: " << sampleInterval << "\n\n";

    int successfulJobs = 0;
    int failedJobs = 0;
    int savedSamples = 0;
    double totalGpuMs = 0.0;
    double totalEndToEndMs = 0.0;
    std::set<std::string> sampledSources;

    auto globalStart = std::chrono::steady_clock::now();

    for (size_t i = 0; i < files.size(); ++i) {
        int jobId = static_cast<int>(i) + 1;
        const std::string& path = files[i];
        std::string baseName = stemFromPath(path);
        std::string sourceKey = extractSourceKey(baseName);

        GrayImage input;
        if (!loadPGM(path, input, error)) {
            failedJobs++;
            summaryCsv << jobId << "," << baseName << ",0,0,0,0,0,load_failed\n";
            runLog << "FAILED load: " << path << " => " << error << "\n";
            continue;
        }

        PipelineConfig config;
        config.threshold = args.threshold;
        config.erodeIterations = args.erodeIterations;
        config.dilateIterations = args.dilateIterations;

        auto jobStart = std::chrono::steady_clock::now();

        PipelineResult pipelineResult;
        if (!runGpuPipeline(input, config, pipelineResult, error)) {
            failedJobs++;
            summaryCsv << jobId << "," << baseName << "," << input.width << "," << input.height << ",0,0,0,gpu_failed\n";
            runLog << "FAILED gpu: " << path << " job " << jobId << " => " << error << "\n";
            continue;
        }

        RegionAnalysisResult analysis;
        if (!analyzeRegions(pipelineResult.cleaned, args.minArea, analysis, error)) {
            failedJobs++;
            summaryCsv << jobId << "," << baseName << "," << input.width << "," << input.height << ",0,"
                       << std::fixed << std::setprecision(3) << pipelineResult.gpuTimeMs << ",0,analysis_failed\n";
            runLog << "FAILED analysis: " << path << " job " << jobId << " => " << error << "\n";
            continue;
        }

        auto jobEnd = std::chrono::steady_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(jobEnd - jobStart).count();

        successfulJobs++;
        totalGpuMs += pipelineResult.gpuTimeMs;
        totalEndToEndMs += totalMs;

        summaryCsv << jobId << "," << baseName << "," << input.width << "," << input.height << ","
                   << analysis.objectCount << ","
                   << std::fixed << std::setprecision(3) << pipelineResult.gpuTimeMs << ","
                   << totalMs << ",success\n";

        if (analysis.regions.empty()) {
            objectCsv << jobId << "," << baseName << ",0,0,0,0,0,0,0\n";
        } else {
            for (const auto& region : analysis.regions) {
                objectCsv << jobId << "," << baseName << ","
                          << analysis.objectCount << ","
                          << region.label << ","
                          << region.area << ","
                          << region.minX << ","
                          << region.minY << ","
                          << region.maxX << ","
                          << region.maxY << "\n";
            }
        }

        bool shouldSample =
            (args.saveSamples > 0) &&
            (savedSamples < args.saveSamples) &&
            (sampledSources.count(sourceKey) == 0) &&
            (static_cast<int>(i) >= savedSamples * sampleInterval);

        if (shouldSample) {
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "sample_%03d", savedSamples + 1);

            std::string saveError;
            savePGM(args.outputDir + "/masks/" + std::string(prefix) + "_mask.pgm", pipelineResult.mask, saveError);
            savePGM(args.outputDir + "/cleaned/" + std::string(prefix) + "_cleaned.pgm", pipelineResult.cleaned, saveError);
            savePGM(args.outputDir + "/labeled/" + std::string(prefix) + "_labeled.pgm", analysis.labeledPreview, saveError);

            sampledSources.insert(sourceKey);
            savedSamples++;
        }

        if (jobId <= 10 || jobId % 500 == 0) {
            std::cout << "Processed job " << jobId
                      << " (" << baseName << ")"
                      << " -> objects: " << analysis.objectCount
                      << ", gpu ms: " << std::fixed << std::setprecision(3) << pipelineResult.gpuTimeMs
                      << "\n";
        }
    }

    auto globalEnd = std::chrono::steady_clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(globalEnd - globalStart).count();
    double avgGpuMs = successfulJobs > 0 ? totalGpuMs / successfulJobs : 0.0;
    double avgTotalMs = successfulJobs > 0 ? totalEndToEndMs / successfulJobs : 0.0;
    double throughput = wallMs > 0.0 ? (1000.0 * successfulJobs / wallMs) : 0.0;

    runLog << "\nSummary\n";
    runLog << "Successful jobs: " << successfulJobs << "\n";
    runLog << "Failed jobs: " << failedJobs << "\n";
    runLog << "Saved sample outputs: " << savedSamples << "\n";
    runLog << std::fixed << std::setprecision(3);
    runLog << "Average GPU time per successful job (ms): " << avgGpuMs << "\n";
    runLog << "Average end-to-end time per successful job (ms): " << avgTotalMs << "\n";
    runLog << "Total wall time (ms): " << wallMs << "\n";
    runLog << "Throughput (images/sec): " << throughput << "\n";

    writeTextFile(args.outputDir + "/run_log.txt", runLog.str());

    std::cout << "Processing complete.\n";
    std::cout << "Artifacts written to: " << args.outputDir << "\n";

    return successfulJobs > 0 ? 0 : 1;
}