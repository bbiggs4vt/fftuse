#pragma once

// Stand-in for dsp::utility::CFftwPlanLock from the parent codebase, so Fft.cpp /
// FftV2.cpp compile on their own. If the real one is available, use it instead.
//
// Still required as of FFTW 3.3.10: only fftw(f)_execute is thread-safe. Plan
// creation and destruction share global planner state and must be serialized.
// (FFTW >= 3.3.5 also offers fftwf_make_planner_thread_safe(), which needs the
// fftw3f_threads library; this lock is the portable alternative.)

#include <mutex>

namespace dsp {
namespace utility {

class CFftwPlanLock
{
public:
	CFftwPlanLock() : mGuard(Mutex()) {}
	CFftwPlanLock(const CFftwPlanLock&) = delete;
	CFftwPlanLock& operator=(const CFftwPlanLock&) = delete;

private:
	static std::mutex& Mutex()
	{
		static std::mutex m;
		return m;
	}

	std::lock_guard<std::mutex> mGuard;
};

} // namespace utility
} // namespace dsp
