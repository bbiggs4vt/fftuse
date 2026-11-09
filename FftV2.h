
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <fftw3.h>

/// Class to do the PSD/FFT calculations (Welch-style averaged periodogram).
///
/// Not thread-safe: Compute() writes member buffers, so use one instance per
/// thread. Construction and destruction are safe from any thread (FFTW planner
/// calls are serialized with CFftwPlanLock).
///
/// Non-copyable: the instance owns an FFTW buffer and plan.
class CFftV2
{
public:
	/// How hard FFTW should work to find a fast plan. Estimate builds the plan
	/// instantly. Measure/Patient make the constructor much slower but can make
	/// Compute() faster. Worth it when one instance is reused many times.
	enum class PlanEffort
	{
		Estimate,
		Measure,
		Patient
	};

	/// Output value used for bins whose power is zero (or not a number), in dB.
	/// The original class produced -FLT_MAX here.
	static constexpr float kDefaultFloorDb = -300.0f;

	/// Largest supported FFT size.
	static constexpr uint32_t kMaxFftSize = 1u << 30;

	/// Constructor
	/// @param [in] fftSize FFT size, rounded up to a power of 2. Must be in [2, kMaxFftSize].
	/// @param [in] overlap fraction of overlap between segments. Must be in [0, 1).
	/// @param [in] reverse use an inverse FFT, with no window and no normalization
	///             (same behavior as the original class; magnitudes are about N^2
	///             larger than in the windowed forward mode)
	/// @param [in] effort FFTW planning effort
	/// @throws std::invalid_argument on bad arguments, std::bad_alloc or
	///         std::runtime_error if FFTW allocation or planning fails
	CFftV2(uint32_t fftSize, float overlap, bool reverse = false, PlanEffort effort = PlanEffort::Estimate);

	~CFftV2();

	CFftV2(const CFftV2&) = delete;
	CFftV2& operator=(const CFftV2&) = delete;

	/// Gets the FFT size, which might not match constructor if not a power of 2
	uint32_t GetFftSize() const { return mFftSize; }

	/// Gets the number of samples between the starts of consecutive segments
	uint32_t GetStride() const { return mStride; }

	/// Minimum input length accepted by Compute()
	uint32_t GetMinInputLength() const { return mFftSize; }

	/// Value written for empty bins (applied after shiftInDb)
	void SetFloorDb(float floorDb) { mFloorDb = floorDb; }
	float GetFloorDb() const { return mFloorDb; }

	/// Computes the averaged power spectrum in dB for the passed in data
	/// @param [in] inData pointer to complex IQ data
	/// @param [in] length length in samples for IQ data, must be >= GetFftSize()
	/// @param [out] outData pointer to output data, must hold at least `bins` floats
	///              (GetFftSize() floats when bins == 0)
	/// @param [in] shiftInDb shift for output data in db (added directly to output)
	/// @param [in] bins number of bins to output from center of band (0=match FFT size)
	/// @throws std::invalid_argument if length < GetFftSize(), bins > GetFftSize(),
	///         or a pointer is null
	void Compute(const std::complex<float>* inData, uint32_t length, float* outData, float shiftInDb = 0.0f, uint32_t bins = 0);

	/// Computes the averaged power spectrum in dB for the passed in data
	/// @param [in] inData reference to complex IQ data
	/// @param [out] outData reference to output data, resized to the number of output bins
	/// @param [in] shiftInDb shift for output data in db (added directly to output)
	/// @param [in] bins number of bins to output from center of band (0=match FFT size)
	void Compute(const std::vector<std::complex<float>>& inData, std::vector<float>& outData, float shiftInDb = 0.0f, uint32_t bins = 0);

private:
	/// Creates the window for the FFT
	void CreateFftWindow(bool reverse);

	uint32_t mFftSize; //!< FFT size (power of 2)
	uint32_t mStride; //!< Stride between segments, always >= 1
	bool mApplyWindow; //!< false in reverse mode (window is all ones)
	float mFloorDb; //!< output value for empty bins
	std::complex<float>* mFftwBuffer; //!< in-place FFTW buffer
	fftwf_plan mFftwPlan; //!< fftw plan
	std::vector<float> mWindow; //!< window, normalized to sum to 1 (forward mode)
	std::vector<float> mAccum; //!< power accumulator, reused across calls
};

typedef boost::shared_ptr<CFftV2> CFftV2Ptr;
