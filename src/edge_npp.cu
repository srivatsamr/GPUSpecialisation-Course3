#include "CImg.h"
#include "edge_npp.h"

using namespace cimg_library;

void checkNppStatus(NppStatus status, const std::string& msg) {
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error (" << status << ") : " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to load and filter image paths
std::vector<fs::path> load_images(const std::string& inputDir) {
    std::vector<fs::path> images;
    for (auto& p : fs::directory_iterator(inputDir)) {
        auto ext = p.path().extension().string();
        images.push_back(p.path());
    }
    return images;
}

// Function to initialize CUDA streams and contexts
std::vector<StreamContext> initialize_streams(int numStreams) {
    std::vector<StreamContext> ctx(numStreams);
    for (auto& c : ctx)
        cudaStreamCreate(&c.stream);
    return ctx;
}

// Function to process a single image
void process_image(const fs::path& imgPath, const std::string& outputDir, StreamContext& c) {
    CImg<unsigned char> img(imgPath.string().c_str());
    CImg<unsigned char> gray = img.get_RGBtoYCbCr().get_channel(0);

    int w = gray.width();
    int h = gray.height();

    if (!c.d_src) {
        cudaMallocPitch(&c.d_src, &c.srcPitch, w * sizeof(Npp8u), h);
        cudaMallocPitch(&c.d_dst, &c.dstPitch, w * sizeof(Npp8u), h);
    }

    cudaMemcpy2DAsync(c.d_src, c.srcPitch, gray.data(), w * sizeof(Npp8u), w * sizeof(Npp8u), h, cudaMemcpyHostToDevice, c.stream);

    NppStreamContext nppCtx = {};
    nppCtx.hStream = c.stream;

    NppiSize srcSize{w, h};
    NppiPoint srcOffset{0, 0};
    NppiSize roi{w, h};

    int nBufferSize = 0;
    checkNppStatus(nppiFilterCannyBorderGetBufferSize(roi, &nBufferSize), "Failed to get Canny buffer size");

    Npp8u* pScratchBuffer = nullptr;
    cudaMalloc(&pScratchBuffer, nBufferSize);

    Npp16s lowThresh = 72;
    Npp16s highThresh = 256;

    checkNppStatus(
        nppiFilterCannyBorder_8u_C1R_Ctx(
            c.d_src, static_cast<int>(c.srcPitch),
            srcSize, srcOffset,
            c.d_dst, static_cast<int>(c.dstPitch),
            roi,
            NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
            lowThresh, highThresh,
            nppiNormL2, NPP_BORDER_REPLICATE,
            pScratchBuffer, nppCtx),
        "Canny filter failed"
    );

    cudaFree(pScratchBuffer);

    CImg<unsigned char> out(w, h, 1, 1);
    cudaMemcpy2DAsync(out.data(), w * sizeof(Npp8u), c.d_dst, c.dstPitch, w * sizeof(Npp8u), h, cudaMemcpyDeviceToHost, c.stream);
    cudaStreamSynchronize(c.stream);

    fs::path outPath = fs::path(outputDir) / imgPath.filename();
    out.save_jpeg(outPath.string().c_str());
}

// Function to clean up CUDA streams and memory
void cleanup_streams(std::vector<StreamContext>& ctx) {
    for (auto& c : ctx) {
        cudaStreamDestroy(c.stream);
        cudaFree(c.d_src);
        cudaFree(c.d_dst);
    }
}

// Updated process_directory function
void process_directory(const std::string& inputDir, const std::string& outputDir, int numStreams) {
    fs::create_directories(outputDir);

    auto images = load_images(inputDir);
    if (images.empty()) {
        std::cerr << "No images found in " << inputDir << std::endl;
        return;
    }

    auto ctx = initialize_streams(numStreams);
    int streamIdx = 0;

    for (const auto& imgPath : images) {
        process_image(imgPath, outputDir, ctx[streamIdx]);
        streamIdx = (streamIdx + 1) % numStreams;
    }

    cleanup_streams(ctx);
}

// Simple main
int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: ./npp_edge <input_dir> <output_dir>\n";
        return 1;
    }

    process_directory(argv[1], argv[2], 4);
    return 0;
}
