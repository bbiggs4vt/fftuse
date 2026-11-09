
#include "stdafx.h"
#include "Fft.h"

#include <cmath>

static const double PI = 3.1415926535897930;

template<typename T>
T ConvertFromAccumualtedValue(const float& value, const float& scalingFactor, const float& shiftFactor)
{
	// Convert, but clamp to extents of the data type we are given
	return(static_cast<T>(std::min(
		std::max(
			float(std::numeric_limits<T>::lowest()),
			shiftFactor + (10.0f*static_cast<float>(std::log10(value / scalingFactor)))),
		float(std::numeric_limits<T>::max()))));
}

CFft::CFft(uint32_t fftSize, float overlap, bool reverse) :
	mFftSize(0),
	mStride(0),
	mFftwBuffer(NULL),
	mFftwPlan(NULL)
{
	mFftSize = static_cast<uint32_t>(std::pow(2.0f, std::ceil(std::log2(static_cast<double>(fftSize)))));
	mStride = static_cast<uint32_t>(std::floor(static_cast<double>(mFftSize)*(1 - overlap)));		
	CreateFftWindow(reverse);

	dsp::utility::CFftwPlanLock lock;
	assert(mFftwBuffer == NULL);
	mFftwBuffer = static_cast<std::complex<float>*>(fftwf_malloc(sizeof(std::complex<float>)* mFftSize));
	assert(mFftwBuffer != NULL);
	assert(mFftwPlan == NULL);
	mFftwPlan = fftwf_plan_dft_1d(mFftSize, 
		reinterpret_cast<fftwf_complex*>(mFftwBuffer), 
		reinterpret_cast<fftwf_complex*>(mFftwBuffer), 
		reverse == false ? FFTW_FORWARD : FFTW_BACKWARD, 
		FFTW_ESTIMATE);
	assert(mFftwPlan != NULL);
}

CFft::~CFft()
{
	dsp::utility::CFftwPlanLock lock;
	fftwf_destroy_plan(mFftwPlan);
	fftwf_free(mFftwBuffer);
}	

void CFft::Compute(const std::vector<std::complex<float>>& inData, std::vector<float>& outData, float shiftInDb, uint32_t bins)
{
	Compute(&inData[0], static_cast<uint32_t>(inData.size()), &outData[0], shiftInDb, bins);
}

void CFft::Compute(const std::complex<float>* inData, uint32_t length, float* outData, float shiftInDb, uint32_t bins)
{
	assert(mFftwPlan != NULL);
	assert(mFftwBuffer != NULL);
	assert(bins <= mFftSize);
	if (bins == 0)
	{
		bins = mFftSize;
	}
	uint32_t halfFftSize = mFftSize / 2;		

	// Allocate the accumulator array
	std::vector<float> accum(mFftSize, 0.0f);
	std::vector<float> fftOut(mFftSize, 0.0f);
	// Calculate how many FFT segment make up the PSD
	uint32_t K = static_cast<uint32_t>(std::floor(static_cast<double>(length - mFftSize + mStride) / static_cast<double>(mStride)));
	// Calculate the scaling factor
	float scaling = static_cast<float>(K);
	//float K2pi = static_cast<float>(K)*2.0f*static_cast<float>(PI);

	// For each segment
	for (uint32_t k = 0; k < K; k++)
	{
		// Load the data into the FFTW buffer applying the window
		for (uint32_t n = 0; n < mFftSize; n++)
		{
			mFftwBuffer[n] = mWindow[n] * inData[k*mStride + n];
		}

		// Execute the FFT
		fftwf_execute(mFftwPlan);

		// Accumulate the absolute value of the result
		for (uint32_t n = 0; n < mFftSize; n++)
		{
			accum[n] += mFftwBuffer[n].real()*mFftwBuffer[n].real() + mFftwBuffer[n].imag()*mFftwBuffer[n].imag();
		}
	}

	// Swap the two halves of the scaled accumulator in dB
	// equivalent of fftshift in matlab
	for (uint32_t n = 0; n < halfFftSize; n++)
	{			
		fftOut[n] = ConvertFromAccumualtedValue<float>(accum[halfFftSize + n], scaling, shiftInDb);
		fftOut[halfFftSize + n] = ConvertFromAccumualtedValue<float>(accum[n], scaling, shiftInDb);
	}
	// Copy out the center bins
	std::memcpy(outData, &fftOut[0 + halfFftSize-(bins/2)], bins*sizeof(float));
}

void CFft::CreateFftWindow(bool reverse)
{
	if(!reverse)
	{
		mWindow.resize(mFftSize);

		float sum = 0.0f;
		for (uint32_t n = 0; n < mFftSize; n++)
		{
			// Hamming 
			// mWindow[n] = static_cast<float>(0.54 - 
			//  0.46*std::cos(2.0*PI*static_cast<double>(n) / static_cast<double>(mFftSize)));
			// Blackman
			mWindow[n] = static_cast<float>(0.42 - 
				0.5*std::cos(2.0*PI*static_cast<double>(n) / static_cast<double>(mFftSize-1)) + 
				0.08*std::cos(4.0*PI*static_cast<double>(n) / static_cast<double>(mFftSize-1)));
			sum += mWindow[n];
		}
		// Scale window
		for (uint32_t n = 0; n < mFftSize; n++)
		{			
			mWindow[n] = mWindow[n] / sum;
		}
	}
	else
	{
		// No windowing
		mWindow.resize(mFftSize, 1.0);
	}
}		




