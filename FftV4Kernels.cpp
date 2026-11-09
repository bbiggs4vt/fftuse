
#include "stdafx.h"
#include "FftV4Kernels.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

// Build notes are in FftV4Kernels.h. Every function here must stay correct under
// -funsafe-math-optimizations (reassociation, no signed zeros, no traps,
// denormals flushed to zero on ARMv7 NEON).

#if defined(_MSC_VER)
#define FFT_RESTRICT __restrict
#else
#define FFT_RESTRICT __restrict__
#endif

namespace
{
inline uint32_t FloatBits(float f)
{
	uint32_t u;
	std::memcpy(&u, &f, sizeof(u));
	return u;
}

inline float BitsFloat(uint32_t u)
{
	float f;
	std::memcpy(&f, &u, sizeof(f));
	return f;
}

/// 10*log10(x) for finite x > 0. Branch-free so the calling loop vectorizes:
/// conditions only drive selects, never a float op (GCC won't if-convert a
/// conditional float op that could trap, which includes int-to-float conversion).
/// Max error is about 1e-5 dB (degree-6 polynomial for log2 of the mantissa,
/// reduced to [sqrt(0.5), sqrt(2))). Denormal x is treated as FLT_MIN (-379.3 dB);
/// accumulated power that small has no precision left anyway.
inline float FastDb(float x)
{
	const float kDbPerOctave = 3.0102999566398120f; // 10*log10(2)

	// Clamp to FLT_MIN with an integer max on the bits: positive floats order like
	// their bit patterns, and negatives (sign bit set) are negative as int32. A
	// float select here would let GCC branch around the rest.
	const int32_t kFltMinBits = 0x00800000;
	const uint32_t bits = static_cast<uint32_t>(std::max(static_cast<int32_t>(FloatBits(x)), kFltMinBits));

	// Split into exponent and mantissa in [1, 2), then reduce the mantissa to
	// [sqrt(0.5), sqrt(2)) by halving it and bumping the exponent when it is above
	// sqrt(2). Done with integer ops so the only conversion runs unconditionally.
	const uint32_t mantissaBits = bits & 0x007fffffu;
	// high = (mantissaBits > 0x3504f3) as arithmetic, since GCC turns a compare into
	// a branch here. sqrt(2) is 0x3fb504f3, and 0x800000 - 0x3504f4 = 0x4afb0c.
	const uint32_t high = (mantissaBits + 0x004afb0cu) >> 23;
	const int32_t exponent = static_cast<int32_t>(bits >> 23) - 127 + static_cast<int32_t>(high);
	const float er = static_cast<float>(exponent);
	const float mr = BitsFloat((mantissaBits | 0x3f800000u) - (high << 23));
	const float t = mr - 1.0f;
	float p = -2.052988328e-01f;
	p = p * t + 3.220862295e-01f;
	p = p * t - 3.677040727e-01f;
	p = p * t + 4.793242921e-01f;
	p = p * t - 7.211225323e-01f;
	p = p * t + 1.442715776e+00f;
	p = p * t;

	return (er + p) * kDbPerOctave;
}

} // namespace

namespace fftv4_kernels
{
/// dst[i] = clamp(10*log10(src[i]) + offsetDb). Zero, negative and NaN power give
/// floorDb, as in CFftV2.
///
/// Two passes on purpose: with one, GCC (under its default -ftrapping-math) moves
/// the log math into the x > 0 branch and then refuses to vectorize. The data is
/// in L1 cache, so the second pass is nearly free.
void ConvertToDb(const float* FFT_RESTRICT src, float* FFT_RESTRICT dst, uint32_t count, float offsetDb, float floorDb)
{
	for (uint32_t i = 0; i < count; i++)
	{
		dst[i] = FastDb(src[i]) + offsetDb;
	}
	for (uint32_t i = 0; i < count; i++)
	{
		const float x = src[i];
		float db = dst[i];
		db = (x > 0.0f) ? db : floorDb;    // zero, negative and NaN fail x > 0
		db = (db > floorDb) ? db : floorDb;
		db = (db > FLT_MAX) ? FLT_MAX : db;
		db = (x > FLT_MAX) ? FLT_MAX : db; // +inf power
		dst[i] = db;
	}
}

/// dst[n] = window[n] * src[n]
void ApplyWindow(const std::complex<float>* FFT_RESTRICT src, const float* FFT_RESTRICT window,
	std::complex<float>* FFT_RESTRICT dst, uint32_t count)
{
	for (uint32_t n = 0; n < count; n++)
	{
		dst[n] = window[n] * src[n];
	}
}

/// accum[n] += |spectrum[n]|^2
///
/// Keep this in complex form: an equivalent loop over interleaved floats
/// (re = p[2n], im = p[2n+1]) measured about 4x slower with GCC 13 -O3, and this
/// loop runs once per segment.
void AccumulatePower(const std::complex<float>* FFT_RESTRICT spectrum, float* FFT_RESTRICT accum, uint32_t count)
{
	for (uint32_t n = 0; n < count; n++)
	{
		accum[n] += std::norm(spectrum[n]);
	}
}
} // namespace fftv4_kernels
