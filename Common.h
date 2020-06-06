#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <cstdint>
#include <chrono>
//AVX2
#include <immintrin.h>

// Escape limit
static constexpr float      g_scFloatBoundary = 4.f;

// Arbitrary fixed-point precision
static constexpr uint32_t   g_scFixedPrecision = 16;

// Undef to not use OpenMP multithreaded execution
#define CPU_MULTITHREADED

// fixed-point typedefs
using fixed = int32_t;
using fixeddw = int64_t;

struct Params
{
    uint32_t    m_Width;
    uint32_t    m_Height;
    uint32_t    m_MaxIterations;
    float       m_LimitsH[2];    // [h0, h1]
    float       m_LimitsV[2];    // [v0, v1]
    bool        m_UseDiscreteGPU;
};

void OutputFrame(uint32_t* pFrameBuffer, const char* pFileName, const Params& params)
{
    FILE* pOutFile = fopen(pFileName, "w");
    if (pOutFile == nullptr)
    {
        printf("Error opening file: %s\n", strerror(errno));
        throw;
    }
    else
    {
        fprintf(pOutFile, "P3\n%d %d\n%d\n ", params.m_Width, params.m_Height, 255);
        for (auto i = 0u; i < params.m_Width * params.m_Height; ++i)
        {
            float pixel = pFrameBuffer[i] / 255.f;
            uint32_t color = pow(pixel, 1 / 2.2f) * 255.f;
            fprintf(pOutFile, "%d %d %d ", color, color, color);
        }
        fclose(pOutFile);
    }
}