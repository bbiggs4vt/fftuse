
#pragma once

#include <complex>
#include <fftw3.h>

/// Class to do the PSD/FFT calculations
class CFft
{
public:
	/// Constructor
	/// @param [in] fftSize fft size for calculations
	/// @param [in] overlap overlap
	CFft(uint32_t fftSize, float overlap, bool reverse=false);

	/// Destructor
	~CFft();

	/// Gets the FFT size, which might not match constructor if not a power of 2
	uint32_t GetFftSize() { return(mFftSize); };

	/// Computes the FFT for the passed in data
	/// @param [in] inData pointer to complex IQ data
	/// @param [in] length length in samples for IQ data
	/// @param [in] outData pointer to output data
	/// @param [in] shiftInDb shift for output data in db (added directly to output)
	/// @param [in] bins number of bins to output from center of band (0=match FFT size)
	void Compute(const std::complex<float>* inData, uint32_t length, float* outData, float shiftInDb = 0.0, uint32_t bins=0);

	/// Computes the FFT for the passed in data
	/// @param [in] inData reference to complex IQ data		
	/// @param [in] outData reference to output data (should already be the proper size)
	/// @param [in] shiftInDb shift for output data in db (added directly to output)
	/// @param [in] bins number of bins to output from center of band (0=match FFT size)
	void Compute(const std::vector<std::complex<float>>& inData, std::vector<float>& outData, float shiftInDb = 0.0, uint32_t bins=0);

private:
	/// Creates the window for the FFT
	void CreateFftWindow(bool reverse);

	uint32_t mFftSize; //!< FFT size
	uint32_t mStride; //!< Stride for FFT
	std::complex<float>* mFftwBuffer; //!< buffer for calculations
	std::vector<float> mWindow;	//!< Window
	fftwf_plan mFftwPlan; //!< fftw plan
};

typedef boost::shared_ptr<CFft> CFftPtr;

