#pragma once
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>
#include <filesystem>
#include <vector>
#include <iostream>

struct StreamContext {
    cudaStream_t stream{};
    Npp8u* d_src{};
    Npp8u* d_dst{};
    size_t srcPitch{};
    size_t dstPitch{};
};

namespace fs = std::filesystem;

void checkNppStatus(NppStatus status, const std::string& msg);
std::vector<fs::path> load_images(const std::string& inputDir);
std::vector<StreamContext> initialize_streams(int numStreams);
void process_image(const fs::path& imgPath, const std::string& outputDir, StreamContext& c);
void cleanup_streams(std::vector<StreamContext>& ctx);
void process_directory(const std::string& inputDir, const std::string& outputDir, int numStreams);

void process_directory(
    const std::string& inputDir,
    const std::string& outputDir,
    int numStreams = 4
);
