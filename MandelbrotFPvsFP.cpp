#include "MandelbrotCPU.h"
#include "MandelbrotGPU.h"

int main()
{
    const Params params = {
        480 * 4,                // width
        320 * 4,                // height
        256,                // maxIterations
        { -2.5f, 1.5f },    // limitsH
        { -1.5f, 1.5f},     // limitsV,
        true                // useDiscreteGPU
    };

    const char* cpuNativeFPFileName = ".\\cpunative-fp.ppm";
    const char* cpuFixedPointFileName = ".\\cpufixedpoint.ppm";
    const char* cpuNativeSIMDFileName = ".\\cpunativesimd-fp.ppm";
    const char* cpuFixedSIMDFileName = ".\\cpufixedpointsimd.ppm";
    const char* gpuNativeFPFileName = ".\\gpunative-fp.ppm";
    const char* gpuFixedPointFileName = ".\\gpufixedpoint.ppm";

    bool renderResults = false;
    const size_t N = 25;

    //std::cout << "CPU SIMD native floating-point\n";
    //for (size_t i = 0; i < N; i++)
    //{
    //    // CPU single-threaded scalar native floating-point implementation
    //    cpu::RenderMandelbrotSIMD<__m256>(params, cpuNativeSIMDFileName, renderResults);
    //}

    std::cout << "CPU SIMD fixed floating-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // CPU single-threaded scalar native floating-point implementation
        cpu::RenderMandelbrotSIMD<__m256i>(params, cpuFixedSIMDFileName, renderResults);
    }

    //std::cout << "CPU scalar native floating-point\n";
    //for (size_t i = 0; i < N; i++)
    //{
    //    // CPU single-threaded scalar native floating-point implementation
    //    cpu::RenderMandelbrotScalar<float>(params, cpuNativeFPFileName, renderResults);
    //}

    //std::cout << "GPU native floating-point\n";
    //for (size_t i = 0; i < N; i++)
    //{
    //    // GPU native floating-point implementation
    //    gpu::RenderMandelbrot(params, "gpgpu-floating-point_comp.spv", gpuNativeFPFileName, renderResults);
    //}

    //std::cout << "CPU scalar fixed-point\n";
    //for (size_t i = 0; i < N; i++)
    //{
    //    // CPU single-threaded scalar fixed-point implementation
    //    cpu::RenderMandelbrotScalar<fixed>(params, cpuFixedPointFileName, renderResults);
    //}

    //std::cout << "GPU fixed-point\n";
    //for (size_t i = 0; i < N; i++)
    //{
    //    // GPU native fixed-point implementation
    //    gpu::RenderMandelbrot(params, "gpgpu-fixed-point_comp.spv", gpuFixedPointFileName, renderResults);
    //}
}