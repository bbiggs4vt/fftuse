
#include "stdafx.h"
#include "FftV4.h"
#include "FftV4Kernels.h"

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
			fftv4_kernels::ApplyWindow(segment, mWindow.data(), mFftIn, N);
		}
		else
		{
			std::copy(segment, segment + N, mFftIn);
		}

		fftwf_execute(mFftwPlan);

		fftv4_kernels::AccumulatePower(mFftOut, mAccum.data(), N);
	}

	// fftshift and convert to dB. Output bin n is accum[(n - bins/2) mod N], which is
	// two contiguous runs of the accumulator: from `first` up to the end, then from 0.
	// Dividing by K is folded into the constant offset.
	const float offsetDb = shiftInDb - 10.0f * std::log10(static_cast<float>(K));
	const uint32_t first = (N - bins / 2) & (N - 1);
	const uint32_t run1 = std::min(bins, N - first);
	fftv4_kernels::ConvertToDb(mAccum.data() + first, outData, run1, offsetDb, mFloorDb);
	fftv4_kernels::ConvertToDb(mAccum.data(), outData + run1, bins - run1, offsetDb, mFloorDb);
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
	// A symmetric Blackman window is zero at both ends, so for N = 2 every sample
	// is zero and there is nothing to normalize. v1 and v2 normalize round-off
	// (about -1.4e-17 per sample) and so end up with a rectangular window; make
	// that explicit rather than depend on rounding.
	if (mFftSize == 2)
	{
		mWindow.assign(2, 0.5f);
		return;
	}

	// Normalize to a sum of 1 (coherent gain): a full-scale tone's peak bin reads 0 dB.
	// Round to float the same way as v2.
	mWindow.resize(mFftSize);
	for (uint32_t n = 0; n < mFftSize; n++)
	{
		mWindow[n] = static_cast<float>(static_cast<float>(w[n]) / sum);
	}

}
