#include "MandelbrotCPU.h"
#include "MandelbrotGPU.h"

int main()
{
    const Params params = {
        1920,                // width
        1280,                // height
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

    bool renderResults = true;
    const size_t N = 1;
#ifdef CPU_MULTITHREADED
    constexpr bool isMultithreaded = true;
#else
    constexpr bool isMultithreaded = false;
#endif

    std::cout << "CPU " << (isMultithreaded ? "MT " : " ST ") << "scalar native floating-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // CPU ST/MT scalar native floating-point implementation
        cpu::RenderMandelbrotScalar<float>(params, cpuNativeFPFileName, renderResults);
    }

    std::cout << (params.m_UseDiscreteGPU ? "dGPU " : "iGPU ") << "native floating-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // GPU native floating-point implementation
        gpu::RenderMandelbrot(params, "gpgpu-floating-point_comp.spv", gpuNativeFPFileName, renderResults);
    }

    std::cout << "CPU " << (isMultithreaded ? "MT " : " ST ") << "scalar fixed-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // CPU ST/MT scalar fixed-point implementation
        cpu::RenderMandelbrotScalar<fixed>(params, cpuFixedPointFileName, renderResults);
    }

    std::cout << "CPU " << (isMultithreaded ? "MT " : " ST ") << "SIMD native floating-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // CPU ST/MT scalar native floating-point implementation
        cpu::RenderMandelbrotSIMD<__m256>(params, cpuNativeSIMDFileName, renderResults);
    }

    std::cout << "CPU " << (isMultithreaded ? "MT " : " ST ") << " SIMD fixed-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // CPU ST/MT SIMD fixed-point implementation
        cpu::RenderMandelbrotSIMD<__m256i>(params, cpuFixedSIMDFileName, renderResults);
    }

    std::cout << (params.m_UseDiscreteGPU ? "dGPU " : "iGPU ") << "fixed-point\n";
    for (size_t i = 0; i < N; i++)
    {
        // GPU native fixed-point implementation
        gpu::RenderMandelbrot(params, "gpgpu-fixed-point_comp.spv", gpuFixedPointFileName, renderResults);
    }
}