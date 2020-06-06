#pragma once

#include "Common.h"

namespace cpu
{
    template<typename T>
    T init_uint(uint32_t val);

    template<typename T>
    T init_float(float f);

    template<typename T>
    T init_row_offsets();

    template<typename T>
    T reinterpret_vector(__m256i a);

    uint32_t min(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    // CPU native floating-point math routines
    float add(float a, float b)
    {
        return a + b;
    }

    float sub(float a, float b)
    {
        return a - b;
    }

    float mul(float a, float b)
    {
        return a * b;
    }

    float div(float a, float b)
    {
        return a / b;
    }

    template<>
    float init_uint(uint32_t val)
    {
        return static_cast<float>(val);
    }

    template<>
    float init_float(float f)
    {
        return f;
    }

    bool less_than(float a, float b)
    {
        return a < b;
    }

    // CPU fixed-point math routines
    fixed add(fixed a, fixed b)
    {
        return a + b;
    }

    fixed sub(fixed a, fixed b)
    {
        return a - b;
    }

    fixed mul(fixed a, fixed b)
    {
        return static_cast<fixed>((static_cast<fixeddw>(a) * static_cast<fixeddw>(b)) >> g_scFixedPrecision);
    }

    fixed div(fixed a, fixed b)
    {
        return static_cast<fixed>((static_cast<fixeddw>(a) << g_scFixedPrecision) / static_cast<fixeddw>(b));
    }

    template<>
    fixed init_float(float f)
    {
        return static_cast<fixed>(f * (1 << g_scFixedPrecision));
    }

    template<>
    fixed init_uint(uint32_t val)
    {
        return static_cast<fixed>(val * (1 << g_scFixedPrecision));
    }

    bool less_than(fixed a, fixed b)
    {
        return a < b;
    }

    // CPU AVX2 floating-point math routines
    __m256 add(__m256 a, __m256 b)
    {
        return _mm256_add_ps(a, b);
    }

    __m256 sub(__m256 a, __m256 b)
    {
        return _mm256_sub_ps(a, b);
    }

    __m256 mul(__m256 a, __m256 b)
    {
        return _mm256_mul_ps(a, b);
    }

    __m256 div(__m256 a, __m256 b)
    {
        return _mm256_div_ps(a, b);
    }

    template<>
    __m256 init_uint(uint32_t val)
    {
        return _mm256_set1_ps(static_cast<float>(val));
    }

    template<>
    __m256 init_float(float f)
    {
        return _mm256_set1_ps(f);
    }

    template<>
    __m256 init_row_offsets()
    {
        return _mm256_setr_ps(
            init_float<float>(0.f),
            init_float<float>(1.f),
            init_float<float>(2.f),
            init_float<float>(3.f),
            init_float<float>(4.f),
            init_float<float>(5.f),
            init_float<float>(6.f),
            init_float<float>(7.f));
    }

    template<>
    __m256 reinterpret_vector(__m256i a)
    {
        return _mm256_castsi256_ps(a);
    }

    __m256 less_than(__m256 a, __m256 b)
    {
        return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
    }

    __m256 compute_active_mask(__m256i maskIterations, __m256 maskBoundary)
    {
        // maskIterations is of type integer, so each of its elements is either 0 or 1.
        // By subtracting 1, we'll end up with either FFFFFFFF or 0 per element, which is then NOT'ed and AND'ed with maskBoundary
        return _mm256_andnot_ps(reinterpret_vector<__m256>(_mm256_sub_epi32(maskIterations, _mm256_set1_epi32(1))), maskBoundary);
    }

    // Return masked-incremented 'iterations' value based on active lanes mask
    __m256i increment_masked(const __m256i& iterations, const __m256& activeMask)
    {
        // Use vblendps
        return _mm256_castps_si256(_mm256_blendv_ps(reinterpret_vector<__m256>(iterations),
            reinterpret_vector<__m256>(_mm256_add_epi32(iterations,
                _mm256_set1_epi32(1))), activeMask));
    }

    // CPU AVX2 fixed-point math routines
    __m256i add(__m256i a, __m256i b)
    {
        return _mm256_add_epi32(a, b);
    }

    __m256i sub(__m256i a, __m256i b)
    {
        return _mm256_sub_epi32(a, b);
    }

    __m256i mul(__m256i a, __m256i b)
    {
        // Extract LH of a & b and pack into 64-bit integers prior to multiplication
        __m256i lha = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(a, 0));
        __m256i lhb = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(b, 0));

        __m256i mulLH = _mm256_mul_epi32(lha, lhb);
        mulLH = _mm256_srli_epi64(mulLH, g_scFixedPrecision);

        // Extract UH of a & b and pack into 64-bit integers prior to multiplication
        __m256i uha = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(a, 1));
        __m256i uhb = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(b, 1));

        __m256i mulUH = _mm256_mul_epi32(uha, uhb);
        mulUH = _mm256_srli_epi64(mulUH, g_scFixedPrecision);

        // There is no way to truncate signed 64-bit integers on AVX2, i.e. as with _mm256_cvtepi64_epi32 in AVX-512,
        // hence, the ugly hack :(
        return _mm256_setr_epi32(
            mulLH.m256i_i64[0],
            mulLH.m256i_i64[1],
            mulLH.m256i_i64[2],
            mulLH.m256i_i64[3],
            mulUH.m256i_i64[0],
            mulUH.m256i_i64[1],
            mulUH.m256i_i64[2],
            mulUH.m256i_i64[3]);
    }

    __m256i div(__m256i a, __m256i b)
    {
        // Extract LH of a & b and pack into 64-bit integers prior to division
        __m256i lha = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(a, 0));
        __m256i lhb = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(b, 0));
        __m256i divLH = _mm256_div_epi64(_mm256_slli_epi64(lha, g_scFixedPrecision), lhb);

        // Extract UH of a & b and pack into 64-bit integers prior to division
        __m256i uha = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(a, 1));
        __m256i uhb = _mm256_cvtepi32_epi64(_mm256_extractf128_si256(b, 1));
        __m256i divUH = _mm256_div_epi64(_mm256_slli_epi64(uha, g_scFixedPrecision), uhb);

        // There is no way to truncate signed 64-bit integers on AVX2, i.e. as with _mm256_cvtepi64_epi32 in AVX-512,
        // hence, the ugly hack :(
        return _mm256_setr_epi32(
            divLH.m256i_i64[0],
            divLH.m256i_i64[1],
            divLH.m256i_i64[2],
            divLH.m256i_i64[3],
            divUH.m256i_i64[0],
            divUH.m256i_i64[1],
            divUH.m256i_i64[2],
            divUH.m256i_i64[3]);
    }

    template<>
    __m256i init_uint(uint32_t val)
    {
        return _mm256_set1_epi32(init_uint<fixed>(val));
    }

    template<>
    __m256i init_float(float f)
    {
        return _mm256_set1_epi32(init_float<fixed>(f));
    }

    template<>
    __m256i init_row_offsets()
    {
        return _mm256_setr_epi32(
            init_uint<fixed>(0),
            init_uint<fixed>(1),
            init_uint<fixed>(2),
            init_uint<fixed>(3),
            init_uint<fixed>(4),
            init_uint<fixed>(5),
            init_uint<fixed>(6),
            init_uint<fixed>(7));
    }

    template<>
    __m256i reinterpret_vector(__m256i a)
    {
        return a;
    }

    __m256i less_than(__m256i a, __m256i b)
    {
        // (a < b) == !(a == b) && !(a > b) -> !((a == b) || (a > b))
        __m256i maskGT = _mm256_cmpgt_epi32(a, b);
        __m256i maskEQ = _mm256_cmpeq_epi32(a, b);

        return _mm256_andnot_si256(_mm256_or_si256(maskGT, maskEQ), _mm256_set1_epi32(1));
    }

    __m256 compute_active_mask(__m256i maskIterations, __m256i maskBoundary)
    {
        // In contrast to above code for __m256, both maskIterations and maskBoundary are of type integer,
        // so each of their elements is either 0 or 1. By subtracting 1 again, we'll end up with either FFFFFFFF or 0 per element,
        // which is then OR'ed together and the result is similarly NOT'ed and AND'ed with all 0xFFFFFFFF to arrive at correct mask.
        __m256 maskIterationsPS = reinterpret_vector<__m256>(_mm256_sub_epi32(maskIterations, _mm256_set1_epi32(1)));
        __m256 maskBoundaryPS = reinterpret_vector<__m256>(_mm256_sub_epi32(maskBoundary, _mm256_set1_epi32(1)));

        return _mm256_andnot_ps(_mm256_or_ps(maskIterationsPS, maskBoundaryPS), reinterpret_vector<__m256>(_mm256_set1_epi32(0xFFFFFFFF)));
    }

    // z_k+1 = z_k^2 + c
    // z_0 = 0 |  c = x + i * y
    // |z_k| <= N ? Inside : Outside

    // Calculate Mandelbrot set for each pixel using native floating-point or fixed-point math implementation
    template<typename Real>
    void RenderMandelbrotScalar(const Params& params, const char* renderFileName, bool renderResults)
    {
        // Frame buffer for storing rendered Mandelbrot contents
        uint32_t* pFrameBuffer = new uint32_t[params.m_Width * params.m_Height];

        // Start timing execution
        auto begin = std::chrono::high_resolution_clock::now();

        // xstep = (h1 - h0) / (float)width
        Real xStep = div(sub(init_float<Real>(params.m_LimitsH[1]), init_float<Real>(params.m_LimitsH[0])), init_uint<Real>(params.m_Width));
        // ystep = (v1 - v0) / (float)height
        Real yStep = div(sub(init_float<Real>(params.m_LimitsV[1]), init_float<Real>(params.m_LimitsV[0])), init_uint<Real>(params.m_Height));

#ifdef CPU_MULTITHREADED
#pragma omp parallel for schedule(dynamic)
#endif
        for (int32_t col = 0; col < params.m_Height; col++)
        {
            for (int32_t row = 0; row < params.m_Width; row++)
            {
                // (row * x_step) + x0
                // (col * y_step) + y0
                Real x0 = add(mul(xStep, init_uint<Real>(row)), init_float<Real>(params.m_LimitsH[0]));
                Real y0 = add(mul(yStep, init_uint<Real>(col)), init_float<Real>(params.m_LimitsV[0]));

                uint32_t iterations = 0;
                Real x = init_float<Real>(0.f);
                Real y = init_float<Real>(0.f);

                // Escape loop
                while (true)
                {
                    // x^2 = x * x
                    // y^2 = y * y
                    Real xSquared = mul(x, x);
                    Real ySquared = mul(y, y);

                    if (// x^2 + y^2 < BOUNDARY
                        !(less_than(add(xSquared, ySquared), init_float<Real>(g_scFloatBoundary))) ||
                        // maxIterations limit
                        ((iterations + 1) >= params.m_MaxIterations))
                    {
                        break;
                    }

                    // y = 2 * x * y + y0
                    // x = x^2 - y^2 + x0
                    y = add(mul(y, mul(init_uint<Real>(2), x)), y0);
                    x = add(sub(xSquared, ySquared), x0);

                    ++iterations;
                }

                pFrameBuffer[row + col * params.m_Width] = min(255, iterations);
            }
        }

        // End timing
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> diff = end - begin;
        std::cout << "\tRuntime: " << diff.count() << " ms" << std::endl;

        // Operation completed, render results if needed
        if (renderResults)
        {
            OutputFrame(pFrameBuffer, renderFileName, params);
        }

        delete[] pFrameBuffer;
    }

    // Calculate Mandelbrot set with AVX2 8 pixels at a time using native floating-point or fixed-point math implementation
    template<typename Real>
    void RenderMandelbrotSIMD(const Params& params, const char* renderFileName, bool renderResults)
    {
        // Frame buffer for storing rendered Mandelbrot contents
        uint32_t* pFrameBuffer = reinterpret_cast<uint32_t*>(_mm_malloc(params.m_Width * params.m_Height * sizeof(uint32_t), 32));

        // Start timing execution
        auto begin = std::chrono::high_resolution_clock::now();

        // xstep = (h1 - h0) / (float)width
        Real xStep = div(sub(init_float<Real>(params.m_LimitsH[1]), init_float<Real>(params.m_LimitsH[0])), init_uint<Real>(params.m_Width));
        // ystep = (v1 - v0) / (float)height
        Real yStep = div(sub(init_float<Real>(params.m_LimitsV[1]), init_float<Real>(params.m_LimitsV[0])), init_uint<Real>(params.m_Height));

        // { 0, 1, 2, 3, 4, 5, 6, 7 }
        Real simdRowOffset = init_row_offsets<Real>();

#ifdef CPU_MULTITHREADED
#pragma omp parallel for schedule(dynamic)
#endif
        for (int32_t col = 0; col < params.m_Height; col++)
        {
            for (int32_t row = 0; row < params.m_Width; row += 8 /*AVX2 width*/)
            {
                // ((row + offset) * x_step) + x0
                // (col * y_step) + y0
                Real x0 = add(mul(xStep, add(simdRowOffset, init_uint<Real>(row))), init_float<Real>(params.m_LimitsH[0]));
                Real y0 = add(mul(yStep, init_uint<Real>(col)), init_float<Real>(params.m_LimitsV[0]));

                __m256 activeMask = reinterpret_vector<__m256>(_mm256_set1_epi32(0xFFFFFFFF));
                __m256i iterations = _mm256_set1_epi32(0);
                Real x = init_float<Real>(0.f);
                Real y = init_float<Real>(0.f);

                // Escape loop
                while (true)
                {
                    // x^2 = x * x
                    // y^2 = y * y
                    Real xSquared = mul(x, x);
                    Real ySquared = mul(y, y);

                    // x^2 + y^2 < BOUNDARY
                    Real maskBoundary = less_than(add(xSquared, ySquared), init_float<Real>(g_scFloatBoundary));
                    // maxIterations limit
                    __m256i maskIterations = less_than(iterations, _mm256_set1_epi32(params.m_MaxIterations));

                    // Current escape mask
                    __m256 activeLanes = compute_active_mask(maskIterations, maskBoundary);
                    // AND with existing active mask to mask-increment for non-active lanes only
                    activeMask = _mm256_and_ps(activeLanes, activeMask);
                    if (_mm256_movemask_ps(activeMask) == 0)
                    {
                        break;
                    }

                    // y = 2 * x * y + y0
                    // x = x^2 - y^2 + x0
                    y = add(mul(y, mul(init_uint<Real>(2), x)), y0);
                    x = add(sub(xSquared, ySquared), x0);

                    // iterations = iterations + 1
                    iterations = increment_masked(iterations, activeMask);
                }

                // Store 8 pixels at a time
                __m256i* pFrameBufferWriteAddress = reinterpret_cast<__m256i*>(&pFrameBuffer[row + col * params.m_Width]);
                _mm256_store_si256(pFrameBufferWriteAddress, _mm256_min_epi32(iterations, _mm256_set1_epi32(255)));
            }
        }

        // End timing
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> diff = end - begin;
        std::cout << "\tRuntime: " << diff.count() << " ms" << std::endl;

        // Operation completed, render results if needed
        if (renderResults)
        {
            OutputFrame(pFrameBuffer, renderFileName, params);
        }

        _mm_free(pFrameBuffer);
    }
}