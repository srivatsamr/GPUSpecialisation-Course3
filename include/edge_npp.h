#pragma once
#include <string>

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
