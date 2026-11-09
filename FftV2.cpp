
#include "stdafx.h"
#include "FftV2.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

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

unsigned PlanFlags(CFftV2::PlanEffort effort)
{
	switch (effort)
	{
	case CFftV2::PlanEffort::Measure: return FFTW_MEASURE;
	case CFftV2::PlanEffort::Patient: return FFTW_PATIENT;
	case CFftV2::PlanEffort::Estimate:
	default: return FFTW_ESTIMATE;
	}
}
} // namespace

CFftV2::CFftV2(uint32_t fftSize, float overlap, bool reverse, PlanEffort effort) :
	mFftSize(0),
	mStride(0),
	mApplyWindow(!reverse),
	mFloorDb(kDefaultFloorDb),
	mFftwBuffer(nullptr),
	mFftwPlan(nullptr)
{
	if (fftSize < 2 || fftSize > kMaxFftSize)
	{
		throw std::invalid_argument("CFftV2: fftSize must be in [2, 2^30]");
	}
	// Written so that NaN fails too
	if (!(overlap >= 0.0f && overlap < 1.0f))
	{
		throw std::invalid_argument("CFftV2: overlap must be in [0, 1)");
	}

	mFftSize = NextPowerOfTwo(fftSize);
	// Same arithmetic as the original class, so strides match exactly. Overlap
	// values just below 1 can still round to 0, so clamp to 1.
	mStride = static_cast<uint32_t>(std::floor(static_cast<double>(mFftSize) * (1 - overlap)));
	mStride = std::max<uint32_t>(mStride, 1);

	CreateFftWindow(reverse);
	mAccum.assign(mFftSize, 0.0f);

	dsp::utility::CFftwPlanLock lock;
	mFftwBuffer = static_cast<std::complex<float>*>(fftwf_malloc(sizeof(std::complex<float>) * mFftSize));
	if (mFftwBuffer == nullptr)
	{
		throw std::bad_alloc();
	}
	mFftwPlan = fftwf_plan_dft_1d(static_cast<int>(mFftSize),
		reinterpret_cast<fftwf_complex*>(mFftwBuffer),
		reinterpret_cast<fftwf_complex*>(mFftwBuffer),
		reverse ? FFTW_BACKWARD : FFTW_FORWARD,
		PlanFlags(effort));
	if (mFftwPlan == nullptr)
	{
		// The destructor won't run for a throwing constructor
		fftwf_free(mFftwBuffer);
		mFftwBuffer = nullptr;
		throw std::runtime_error("CFftV2: FFTW plan creation failed");
	}
}

CFftV2::~CFftV2()
{
	dsp::utility::CFftwPlanLock lock;
	fftwf_destroy_plan(mFftwPlan);
	fftwf_free(mFftwBuffer);
}

void CFftV2::Compute(const std::vector<std::complex<float>>& inData, std::vector<float>& outData, float shiftInDb, uint32_t bins)
{
	if (inData.size() > std::numeric_limits<uint32_t>::max())
	{
		throw std::invalid_argument("CFftV2::Compute: input too large");
	}
	if (bins > mFftSize)
	{
		throw std::invalid_argument("CFftV2::Compute: bins must be <= FFT size");
	}
	outData.resize(bins == 0 ? mFftSize : bins);
	Compute(inData.data(), static_cast<uint32_t>(inData.size()), outData.data(), shiftInDb, bins);
}

void CFftV2::Compute(const std::complex<float>* inData, uint32_t length, float* outData, float shiftInDb, uint32_t bins)
{
	if (bins == 0)
	{
		bins = mFftSize;
	}
	if (bins > mFftSize)
	{
		throw std::invalid_argument("CFftV2::Compute: bins must be <= FFT size");
	}
	if (length < mFftSize)
	{
		throw std::invalid_argument("CFftV2::Compute: input shorter than one FFT segment");
	}
	if (inData == nullptr || outData == nullptr)
	{
		throw std::invalid_argument("CFftV2::Compute: null pointer");
	}

	// Number of segments. Exact integer form of floor((length - N + stride) / stride),
	// and can't underflow because length >= N.
	const uint32_t K = (length - mFftSize) / mStride + 1;

	std::complex<float>* const buffer = mFftwBuffer;
	const float* const window = mWindow.data();
	float* const accum = mAccum.data();
	const uint32_t N = mFftSize;

	std::fill(mAccum.begin(), mAccum.end(), 0.0f);

	for (uint32_t k = 0; k < K; k++)
	{
		const std::complex<float>* segment = inData + static_cast<size_t>(k) * mStride;

		// Load the data into the FFTW buffer applying the window
		if (mApplyWindow)
		{
			for (uint32_t n = 0; n < N; n++)
			{
				buffer[n] = window[n] * segment[n];
			}
		}
		else
		{
			std::copy(segment, segment + N, buffer);
		}

		fftwf_execute(mFftwPlan);

		// Accumulate the power of the result
		for (uint32_t n = 0; n < N; n++)
		{
			accum[n] += std::norm(buffer[n]);
		}
	}

	// Convert to dB and fftshift in one pass, only for the bins we output.
	// Output bin n is accum[(n - bins/2) mod N], which is the same bin the original
	// read via the shifted fftOut[n + N/2 - bins/2]. Dividing by K is folded into
	// the constant offset.
	const float offsetDb = shiftInDb - 10.0f * std::log10(static_cast<float>(K));
	const float floorDb = mFloorDb;
	const uint32_t mask = N - 1;
	const uint32_t start = N - bins / 2;
	for (uint32_t n = 0; n < bins; n++)
	{
		float db = 10.0f * std::log10(accum[(start + n) & mask]) + offsetDb;
		// !(db > floor) also catches NaN and -inf from empty bins
		if (!(db > floorDb))
		{
			db = floorDb;
		}
		else if (db > FLT_MAX)
		{
			db = FLT_MAX;
		}
		outData[n] = db;
	}
}

void CFftV2::CreateFftWindow(bool reverse)
{
	if (reverse)
	{
		// No windowing (and no normalization), as in the original class
		mWindow.assign(mFftSize, 1.0f);
		return;
	}

	mWindow.resize(mFftSize);

	// Symmetric Blackman, same as the original class. Sum in double so large FFTs
	// normalize accurately.
	double sum = 0.0;
	const double denom = static_cast<double>(mFftSize - 1);
	for (uint32_t n = 0; n < mFftSize; n++)
	{
		const double w = 0.42
			- 0.5 * std::cos(2.0 * PI * static_cast<double>(n) / denom)
			+ 0.08 * std::cos(4.0 * PI * static_cast<double>(n) / denom);
		mWindow[n] = static_cast<float>(w);
		sum += w;
	}
	// Normalize to a sum of 1 (coherent gain): a full-scale tone's peak bin reads 0 dB
	for (uint32_t n = 0; n < mFftSize; n++)
	{
		mWindow[n] = static_cast<float>(mWindow[n] / sum);
	}
}
