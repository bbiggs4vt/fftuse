
#include "stdafx.h"
#include "FftV3.h"

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

unsigned PlanFlags(CFftV3::PlanEffort effort)
{
	switch (effort)
	{
	case CFftV3::PlanEffort::Measure: return FFTW_MEASURE;
	case CFftV3::PlanEffort::Patient: return FFTW_PATIENT;
	case CFftV3::PlanEffort::Estimate:
	default: return FFTW_ESTIMATE;
	}
}
} // namespace

CFftV3::CFftV3(uint32_t fftSize, float overlap, bool reverse, PlanEffort effort, uint32_t batchSize) :
	mFftSize(0),
	mStride(0),
	mApplyWindow(!reverse),
	mFloorDb(kDefaultFloorDb),
	mBatchSize(0),
	mFftwBuffer(nullptr),
	mBatchPlan(nullptr),
	mSinglePlan(nullptr)
{
	if (fftSize < 2 || fftSize > kMaxFftSize)
	{
		throw std::invalid_argument("CFftV3: fftSize must be in [2, 2^30]");
	}
	// Written so that NaN fails too
	if (!(overlap >= 0.0f && overlap < 1.0f))
	{
		throw std::invalid_argument("CFftV3: overlap must be in [0, 1)");
	}
	if (batchSize > kMaxBatchSize)
	{
		throw std::invalid_argument("CFftV3: batchSize must be <= kMaxBatchSize");
	}

	mFftSize = NextPowerOfTwo(fftSize);
	// Same arithmetic as the original class, so strides match exactly. Overlap
	// values just below 1 can still round to 0, so clamp to 1.
	mStride = static_cast<uint32_t>(std::floor(static_cast<double>(mFftSize) * (1 - overlap)));
	mStride = std::max<uint32_t>(mStride, 1);

	if (batchSize == 0)
	{
		const size_t segmentBytes = sizeof(std::complex<float>) * mFftSize;
		batchSize = static_cast<uint32_t>(std::min<size_t>(kMaxBatchSize, std::max<size_t>(1, kAutoBatchBytes / segmentBytes)));
	}
	mBatchSize = batchSize;

	CreateFftWindow(reverse);
	mAccum.assign(mFftSize, 0.0f);

	const int sign = reverse ? FFTW_BACKWARD : FFTW_FORWARD;
	const unsigned flags = PlanFlags(effort);
	const int n = static_cast<int>(mFftSize);

	dsp::utility::CFftwPlanLock lock;
	mFftwBuffer = static_cast<std::complex<float>*>(
		fftwf_malloc(sizeof(std::complex<float>) * static_cast<size_t>(mFftSize) * mBatchSize));
	if (mFftwBuffer == nullptr)
	{
		throw std::bad_alloc();
	}
	fftwf_complex* const buf = reinterpret_cast<fftwf_complex*>(mFftwBuffer);

	// Segments are contiguous in the buffer: stride 1, distance N between segments
	if (mBatchSize > 1)
	{
		mBatchPlan = fftwf_plan_many_dft(1, &n, static_cast<int>(mBatchSize),
			buf, nullptr, 1, n,
			buf, nullptr, 1, n,
			sign, flags);
	}
	mSinglePlan = fftwf_plan_dft_1d(n, buf, buf, sign, flags);
	if (mSinglePlan == nullptr || (mBatchSize > 1 && mBatchPlan == nullptr))
	{
		// The destructor won't run for a throwing constructor
		Release();
		throw std::runtime_error("CFftV3: FFTW plan creation failed");
	}
	if (mBatchSize == 1)
	{
		mBatchPlan = mSinglePlan;
	}
}

CFftV3::~CFftV3()
{
	dsp::utility::CFftwPlanLock lock;
	Release();
}

void CFftV3::Release()
{
	// Caller holds CFftwPlanLock
	if (mBatchPlan != nullptr && mBatchPlan != mSinglePlan)
	{
		fftwf_destroy_plan(mBatchPlan);
	}
	if (mSinglePlan != nullptr)
	{
		fftwf_destroy_plan(mSinglePlan);
	}
	fftwf_free(mFftwBuffer);
	mBatchPlan = nullptr;
	mSinglePlan = nullptr;
	mFftwBuffer = nullptr;
}

void CFftV3::Compute(const std::vector<std::complex<float>>& inData, std::vector<float>& outData, float shiftInDb, uint32_t bins)
{
	if (inData.size() > std::numeric_limits<uint32_t>::max())
	{
		throw std::invalid_argument("CFftV3::Compute: input too large");
	}
	if (bins > mFftSize)
	{
		throw std::invalid_argument("CFftV3::Compute: bins must be <= FFT size");
	}
	outData.resize(bins == 0 ? mFftSize : bins);
	Compute(inData.data(), static_cast<uint32_t>(inData.size()), outData.data(), shiftInDb, bins);
}

void CFftV3::LoadSegments(const std::complex<float>* first, uint32_t count)
{
	const uint32_t N = mFftSize;
	const float* const window = mWindow.data();
	for (uint32_t b = 0; b < count; b++)
	{
		const std::complex<float>* segment = first + static_cast<size_t>(b) * mStride;
		std::complex<float>* dst = mFftwBuffer + static_cast<size_t>(b) * N;
		if (mApplyWindow)
		{
			for (uint32_t n = 0; n < N; n++)
			{
				dst[n] = window[n] * segment[n];
			}
		}
		else
		{
			std::copy(segment, segment + N, dst);
		}
	}
}

void CFftV3::AccumulateSegments(uint32_t count)
{
	const uint32_t N = mFftSize;
	float* const accum = mAccum.data();
	for (uint32_t b = 0; b < count; b++)
	{
		const std::complex<float>* src = mFftwBuffer + static_cast<size_t>(b) * N;
		for (uint32_t n = 0; n < N; n++)
		{
			accum[n] += std::norm(src[n]);
		}
	}
}

void CFftV3::Compute(const std::complex<float>* inData, uint32_t length, float* outData, float shiftInDb, uint32_t bins)
{
	if (bins == 0)
	{
		bins = mFftSize;
	}
	if (bins > mFftSize)
	{
		throw std::invalid_argument("CFftV3::Compute: bins must be <= FFT size");
	}
	if (length < mFftSize)
	{
		throw std::invalid_argument("CFftV3::Compute: input shorter than one FFT segment");
	}
	if (inData == nullptr || outData == nullptr)
	{
		throw std::invalid_argument("CFftV3::Compute: null pointer");
	}

	// Number of segments. Exact integer form of floor((length - N + stride) / stride),
	// and can't underflow because length >= N.
	const uint32_t K = (length - mFftSize) / mStride + 1;

	std::fill(mAccum.begin(), mAccum.end(), 0.0f);

	// Whole batches, then leftover segments one at a time through the single plan.
	// Both plans work on the start of the same buffer, so the leftovers reuse slot 0.
	uint32_t k = 0;
	if (mBatchSize > 1)
	{
		for (; k + mBatchSize <= K; k += mBatchSize)
		{
			LoadSegments(inData + static_cast<size_t>(k) * mStride, mBatchSize);
			fftwf_execute(mBatchPlan);
			AccumulateSegments(mBatchSize);
		}
	}
	for (; k < K; k++)
	{
		LoadSegments(inData + static_cast<size_t>(k) * mStride, 1);
		fftwf_execute(mSinglePlan);
		AccumulateSegments(1);
	}

	// Convert to dB and fftshift in one pass, only for the bins we output.
	// Output bin n is accum[(n - bins/2) mod N]. Dividing by K is folded into
	// the constant offset.
	const uint32_t N = mFftSize;
	const float* const accum = mAccum.data();
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

void CFftV3::CreateFftWindow(bool reverse)
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
