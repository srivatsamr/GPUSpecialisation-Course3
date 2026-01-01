# Canny Edge Detection using CUDA NPP

The aim of this project was to get hands on experience with NVIDIA's NPP
for image processing. I chose a simple canny edge detectioni example from cuda samples
and adapted it to run on multiple input images using cuda streams. I decided to use
[CImg](https://cimg.eu/) as it is header only dependency for reading and saving images (instead of FreeImage used in samples).

I use the [BSDS](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) dataset as it is already
in JPG format. 

# Setup and building the program

In order to setup the development environment, I use [pixi](https://pixi.prefix.dev/latest/installation/).
Use `pixi install`and `pixi shell` to activate the environment in terminal before building the project.
Note: I use direnv to activate the environment.

To build the project, modify the jpeg library paths in CMakeLists and then:

```
mkdir -p build
cd build
cmake ..`
cmake --build . -j$(nproc)
```

And run the code using:
`./npp_edge </path/to/input/images> </path/to/output/images>`

# Lessons Learnt

- As I used CImg instead of helper functions from nvidia sample code, I had to learn about the [pitched](https://www.cis.upenn.edu/~devietti/classes/cis601-spring2017/slides/cuda-2d-profiling.pdf)
memory and copying 2D images for coalesced memory access. 
- NPP API with stream context can be used to run the same function (kernel) on different streams for
embarrasingly parallel work on GPU as streams allow ovelapping data transfers and execution on independent data

