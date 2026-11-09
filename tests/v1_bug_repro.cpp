// Reproduces the reported bugs against the ORIGINAL CFft (Fft.cpp), and runs the
// same scenarios against CFftV2. Each scenario runs in a forked child so a crash
// or sanitizer abort is contained. Build with ASan/UBSan (the CMake target
// v1_bug_repro does) to catch out-of-bounds accesses that don't segfault.
//
// Linux/macOS only (uses fork).

#include "stdafx.h"
#include "Fft.h"
#include "FftV2.h"

#include <cfloat>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
// Child exit codes
const int kOk = 0;          // ran, output looked sane
const int kBadOutput = 10;  // ran, output was garbage
const int kThrew = 11;      // threw a std::exception (the v2 behavior we want)
const int kNoCopy = 12;     // type isn't copyable, so the scenario can't be written

std::string RunIsolated(const std::function<int()>& body)
{
	std::fflush(stdout);
	pid_t pid = fork();
	if (pid == 0)
	{
		// Silence sanitizer reports in the child; the parent reports the outcome
		if (!freopen("/dev/null", "w", stderr)) {}
		int rc;
		try
		{
			rc = body();
		}
		catch (const std::exception&)
		{
			rc = kThrew;
		}
		_exit(rc);
	}
	int status = 0;
	waitpid(pid, &status, 0);
	if (WIFSIGNALED(status))
	{
		return "CRASH (signal " + std::to_string(WTERMSIG(status)) + ")";
	}
	switch (WEXITSTATUS(status))
	{
	case kOk: return "ok";
	case kBadOutput: return "GARBAGE OUTPUT";
	case kThrew: return "throws (handled)";
	case kNoCopy: return "copy doesn't compile (deleted)";
	default: return "SANITIZER ERROR / abort (exit " + std::to_string(WEXITSTATUS(status)) + ")";
	}
}

std::vector<std::complex<float>> Ones(size_t n)
{
	return std::vector<std::complex<float>>(n, { 1.0f, 0.5f });
}

bool AllFinite(const float* p, size_t n)
{
	for (size_t i = 0; i < n; i++)
	{
		if (!std::isfinite(p[i]) || p[i] <= -1e30f)
		{
			return false;
		}
	}
	return true;
}

template <typename T>
struct Scenarios
{
	static int ShortInputWraps()
	{
		// length < N - stride: (length - N + stride) wraps in uint32_t
		T f(1024, 0.5f);
		auto in = Ones(100);
		std::vector<float> out(1024);
		f.Compute(in.data(), 100, out.data());
		return AllFinite(out.data(), out.size()) ? kOk : kBadOutput;
	}
	static int ShortInputZeroSegments()
	{
		// N - stride <= length < N: K = 0, every bin is log10(0/0)
		T f(1024, 0.5f);
		auto in = Ones(800);
		std::vector<float> out(1024);
		f.Compute(in.data(), 800, out.data());
		return AllFinite(out.data(), out.size()) ? kOk : kBadOutput;
	}
	static int OverlapOne()
	{
		T f(1024, 1.0f);
		auto in = Ones(4096);
		std::vector<float> out(1024);
		f.Compute(in.data(), 4096, out.data());
		return AllFinite(out.data(), out.size()) ? kOk : kBadOutput;
	}
	static int FftSizeZero()
	{
		T f(0, 0.5f);
		auto in = Ones(4096);
		std::vector<float> out(4096);
		f.Compute(in.data(), 4096, out.data());
		return f.GetFftSize() == 0 ? kBadOutput : kOk;
	}
	static int VectorEmptyOutput()
	{
		T f(1024, 0.5f);
		auto in = Ones(4096);
		std::vector<float> out; // caller forgot to size it
		f.Compute(in, out);
		return out.size() == 1024 ? kOk : kBadOutput;
	}
	static int VectorEmptyInput()
	{
		T f(1024, 0.5f);
		std::vector<std::complex<float>> in;
		std::vector<float> out(1024);
		f.Compute(in, out);
		return AllFinite(out.data(), out.size()) ? kOk : kBadOutput;
	}
	static int BinsTooLarge()
	{
		T f(1024, 0.5f);
		auto in = Ones(4096);
		std::vector<float> out(2048);
		f.Compute(in.data(), 4096, out.data(), 0.0f, 2048);
		return kOk;
	}
	static int CopyDoubleFree()
	{
		return copyIf<T>();
	}
	static int ZeroInput()
	{
		T f(1024, 0.5f);
		std::vector<std::complex<float>> in(4096);
		std::vector<float> out(1024);
		f.Compute(in.data(), 4096, out.data());
		return out[0] < -1e30f ? kBadOutput : kOk;
	}

	template <typename U>
	static typename std::enable_if<std::is_copy_constructible<U>::value, int>::type copyIf()
	{
		U a(1024, 0.5f);
		U b(a); // both destructors free the same buffer and plan
		return kOk;
	}
	template <typename U>
	static typename std::enable_if<!std::is_copy_constructible<U>::value, int>::type copyIf()
	{
		return kNoCopy;
	}
};

struct Row
{
	const char* name;
	int (*v1)();
	int (*v2)();
};
} // namespace

int main()
{
	const Row rows[] = {
		{ "1a short input (len < N-stride)", Scenarios<CFft>::ShortInputWraps, Scenarios<CFftV2>::ShortInputWraps },
		{ "1b short input (N-stride <= len < N)", Scenarios<CFft>::ShortInputZeroSegments, Scenarios<CFftV2>::ShortInputZeroSegments },
		{ "2a overlap = 1", Scenarios<CFft>::OverlapOne, Scenarios<CFftV2>::OverlapOne },
		{ "2b fftSize = 0", Scenarios<CFft>::FftSizeZero, Scenarios<CFftV2>::FftSizeZero },
		{ "3a vector overload, empty output", Scenarios<CFft>::VectorEmptyOutput, Scenarios<CFftV2>::VectorEmptyOutput },
		{ "3b vector overload, empty input", Scenarios<CFft>::VectorEmptyInput, Scenarios<CFftV2>::VectorEmptyInput },
		{ "4  copy constructor", Scenarios<CFft>::CopyDoubleFree, Scenarios<CFftV2>::CopyDoubleFree },
		{ "5  bins > N (release build)", Scenarios<CFft>::BinsTooLarge, Scenarios<CFftV2>::BinsTooLarge },
		{ "6  all-zero input -> -FLT_MAX", Scenarios<CFft>::ZeroInput, Scenarios<CFftV2>::ZeroInput },
	};

	std::printf("%-40s %-40s %s\n", "scenario", "v1 (Fft.cpp)", "v2 (FftV2.cpp)");
	std::printf("%s\n", std::string(110, '-').c_str());
	for (const Row& r : rows)
	{
		const std::string a = RunIsolated(r.v1);
		const std::string b = RunIsolated(r.v2);
		std::printf("%-40s %-40s %s\n", r.name, a.c_str(), b.c_str());
	}
	return 0;
}
