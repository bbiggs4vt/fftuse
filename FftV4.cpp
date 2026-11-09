
#include "stdafx.h"
#include "FftV4.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

#if defined(_MSC_VER)
#define FFT_RESTRICT __restrict
#else
#define FFT_RESTRICT __restrict__
#endif

namespace
{
const double PI = 3.1415926535897930;

/// Smallest power of two >= v. Requires 1 <= v <= 2^31.
uint32_t NextPowerOfTwo(uint32_t v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	return v + 1;
}

unsigned PlanFlags(CFftV4::PlanEffort effort)
{
	switch (effort)
	{
	case CFftV4::PlanEffort::Measure: return FFTW_MEASURE;
	case CFftV4::PlanEffort::Patient: return FFTW_PATIENT;
	case CFftV4::PlanEffort::WisdomOrEstimate: // handled in the constructor
	case CFftV4::PlanEffort::Estimate:
	default: return FFTW_ESTIMATE;
	}
}

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
} // namespace

CFftV4::CFftV4(uint32_t fftSize, float overlap, bool reverse, PlanEffort effort) :
	mFftSize(0),
	mStride(0),
	mApplyWindow(!reverse),
	mPlannedFromWisdom(false),
	mFloorDb(kDefaultFloorDb),
	mFftIn(nullptr),
	mFftOut(nullptr),
	mFftwPlan(nullptr)
{
	if (fftSize < 2 || fftSize > kMaxFftSize)
	{
		throw std::invalid_argument("CFftV4: fftSize must be in [2, 2^30]");
	}
	// Written so that NaN fails too
	if (!(overlap >= 0.0f && overlap < 1.0f))
	{
		throw std::invalid_argument("CFftV4: overlap must be in [0, 1)");
	}

	mFftSize = NextPowerOfTwo(fftSize);
	// Same arithmetic as the original class, so strides match exactly. Overlap
	// values just below 1 can still round to 0, so clamp to 1.
	mStride = static_cast<uint32_t>(std::floor(static_cast<double>(mFftSize) * (1 - overlap)));
	mStride = std::max<uint32_t>(mStride, 1);

	CreateFftWindow(reverse);
	mAccum.assign(mFftSize, 0.0f);

	const size_t bytes = sizeof(fftwf_complex) * mFftSize;

	dsp::utility::CFftwPlanLock lock;
	mFftIn = static_cast<std::complex<float>*>(fftwf_malloc(bytes));
	mFftOut = static_cast<std::complex<float>*>(fftwf_malloc(bytes));
	if (mFftIn == nullptr || mFftOut == nullptr)
	{
		Release();
		throw std::bad_alloc();
	}
	// DESTROY_INPUT: the input buffer is rewritten for every segment anyway, and it
	// lets FFTW use faster out-of-place algorithms
	auto plan = [&](unsigned flags) {
		return fftwf_plan_dft_1d(static_cast<int>(mFftSize),
			reinterpret_cast<fftwf_complex*>(mFftIn),
			reinterpret_cast<fftwf_complex*>(mFftOut),
			reverse ? FFTW_BACKWARD : FFTW_FORWARD,
			flags | FFTW_DESTROY_INPUT);
	};

	// Probe for measured wisdom. MEASURE | WISDOM_ONLY returns null unless wisdom
	// recorded at Measure or Patient rigor covers this plan, and it never measures.
	// Estimate-level wisdom entries don't count.
	if (effort != PlanEffort::Estimate)
	{
		const unsigned probe = (effort == PlanEffort::Patient) ? FFTW_PATIENT : FFTW_MEASURE;
		mFftwPlan = plan(probe | FFTW_WISDOM_ONLY);
		mPlannedFromWisdom = (mFftwPlan != nullptr);
	}
	if (mFftwPlan == nullptr)
	{
		mFftwPlan = plan(PlanFlags(effort));
	}
	if (mFftwPlan == nullptr)
	{
		// The destructor won't run for a throwing constructor
		Release();
		throw std::runtime_error("CFftV4: FFTW plan creation failed");
	}
}

CFftV4::~CFftV4()
{
	dsp::utility::CFftwPlanLock lock;
	Release();
}

void CFftV4::Release()
{
	if (mFftwPlan != nullptr)
	{
		fftwf_destroy_plan(mFftwPlan);
	}
	fftwf_free(mFftIn);
	fftwf_free(mFftOut);
	mFftwPlan = nullptr;
	mFftIn = nullptr;
	mFftOut = nullptr;
}

void CFftV4::Compute(const std::vector<std::complex<float>>& inData, std::vector<float>& outData, float shiftInDb, uint32_t bins)
{
	if (inData.size() > std::numeric_limits<uint32_t>::max())
	{
		throw std::invalid_argument("CFftV4::Compute: input too large");
	}
	if (bins > mFftSize)
	{
		throw std::invalid_argument("CFftV4::Compute: bins must be <= FFT size");
	}
	outData.resize(bins == 0 ? mFftSize : bins);
	Compute(inData.data(), static_cast<uint32_t>(inData.size()), outData.data(), shiftInDb, bins);
}

void CFftV4::Compute(const std::complex<float>* inData, uint32_t length, float* outData, float shiftInDb, uint32_t bins)
{
	if (bins == 0)
	{
		bins = mFftSize;
	}
	if (bins > mFftSize)
	{
		throw std::invalid_argument("CFftV4::Compute: bins must be <= FFT size");
	}
	if (length < mFftSize)
	{
		throw std::invalid_argument("CFftV4::Compute: input shorter than one FFT segment");
	}
	if (inData == nullptr || outData == nullptr)
	{
		throw std::invalid_argument("CFftV4::Compute: null pointer");
	}

	// Number of segments. Exact integer form of floor((length - N + stride) / stride),
	// and can't underflow because length >= N.
	const uint32_t K = (length - mFftSize) / mStride + 1;
	const uint32_t N = mFftSize;

	std::fill(mAccum.begin(), mAccum.end(), 0.0f);

	for (uint32_t k = 0; k < K; k++)
	{
		const std::complex<float>* segment = inData + static_cast<size_t>(k) * mStride;
		if (mApplyWindow)
		{
			ApplyWindow(segment, mWindow.data(), mFftIn, N);
		}
		else
		{
			std::copy(segment, segment + N, mFftIn);
		}

		fftwf_execute(mFftwPlan);

		AccumulatePower(mFftOut, mAccum.data(), N);
	}

	// fftshift and convert to dB. Output bin n is accum[(n - bins/2) mod N], which is
	// two contiguous runs of the accumulator: from `first` up to the end, then from 0.
	// Dividing by K is folded into the constant offset.
	const float offsetDb = shiftInDb - 10.0f * std::log10(static_cast<float>(K));
	const uint32_t first = (N - bins / 2) & (N - 1);
	const uint32_t run1 = std::min(bins, N - first);
	ConvertToDb(mAccum.data() + first, outData, run1, offsetDb, mFloorDb);
	ConvertToDb(mAccum.data(), outData + run1, bins - run1, offsetDb, mFloorDb);
}

void CFftV4::CreateFftWindow(bool reverse)
{
	if (reverse)
	{
		// No windowing (and no normalization), as in the original class
		mWindow.assign(mFftSize, 1.0f);
		return;
	}

	std::vector<double> w(mFftSize);

	// Symmetric Blackman, same as the original class. Sum in double so large FFTs
	// normalize accurately.
	double sum = 0.0;
	const double denom = static_cast<double>(mFftSize - 1);
	for (uint32_t n = 0; n < mFftSize; n++)
	{
		w[n] = 0.42
			- 0.5 * std::cos(2.0 * PI * static_cast<double>(n) / denom)
			+ 0.08 * std::cos(4.0 * PI * static_cast<double>(n) / denom);
		sum += w[n];
	}
	// Normalize to a sum of 1 (coherent gain): a full-scale tone's peak bin reads 0 dB.
	// Round to float the same way as v2.
	mWindow.resize(mFftSize);
	for (uint32_t n = 0; n < mFftSize; n++)
	{
		mWindow[n] = static_cast<float>(static_cast<float>(w[n]) / sum);
	}

}
