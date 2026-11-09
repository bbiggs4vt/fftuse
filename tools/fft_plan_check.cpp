// Diagnoses whether FFTW_MEASURE picks good plans on this machine.
//
// On a Cortex-A9 (Arria 10 SoC), plans from FFTW_MEASURE (and wisdom files made
// with it) ran 25-36% slower inside CFftV4 than FFTW_ESTIMATE plans. Two causes
// fit that:
//
//   1. Bad measurements. With no usable cycle counter (common on ARMv7), FFTW
//      times candidates with a coarse clock and can choose badly. Signs: the
//      measured plan is slower than the Estimate plan even when timed on its
//      own, and repeated Measure planning picks different plans.
//   2. Cache interaction. FFTW times each candidate alone, with warm caches. In
//      CFftV4 the window and accumulate passes run between transforms and
//      compete for cache. Signs: the measured plan is faster alone but slower
//      in the real loop.
//
// For each size this builds plans in CFftV4's shape (out-of-place,
// FFTW_DESTROY_INPUT) and CFftV2's shape (in-place): one Estimate plan and
// several Measure plans, forgetting wisdom before each. It then times each plan
// alone ("isolated") and inside a loop that does what CFftV4::Compute does per
// segment ("in situ"), and prints a verdict.
//
//   fft_plan_check                       # sizes 1024,4096
//   fft_plan_check --sizes 256,1024,4096 --measure-runs 5 --details

#include "stdafx.h"
#include "FftV4Kernels.h"

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
using Complex = std::complex<float>;
const double kPi = 3.14159265358979323846;

struct Options
{
	std::vector<uint32_t> sizes = { 1024, 4096 };
	int measureRuns = 3;
	double minTimeMs = 50.0;
	int samples = 5;
	uint32_t segments = 32;
	bool details = false;
};

struct Buffers
{
	explicit Buffers(uint32_t n) : N(n)
	{
		in = static_cast<Complex*>(fftwf_malloc(sizeof(Complex) * N));
		out = static_cast<Complex*>(fftwf_malloc(sizeof(Complex) * N));
		pristine.resize(N);
		std::mt19937 rng(7);
		std::normal_distribution<float> d(0.0f, 1.0f);
		for (auto& x : pristine)
		{
			x = { d(rng), d(rng) };
		}
	}
	~Buffers()
	{
		fftwf_free(in);
		fftwf_free(out);
	}
	Buffers(const Buffers&) = delete;
	Buffers& operator=(const Buffers&) = delete;

	uint32_t N;
	Complex* in;
	Complex* out;
	std::vector<Complex> pristine;
};

enum class Shape
{
	OutOfPlace, // CFftV4
	InPlace     // CFft, CFftV2, CFftV3
};

fftwf_plan MakePlan(Buffers& b, Shape shape, unsigned flags)
{
	fftwf_complex* in = reinterpret_cast<fftwf_complex*>(b.in);
	fftwf_complex* out = shape == Shape::OutOfPlace ? reinterpret_cast<fftwf_complex*>(b.out) : in;
	const unsigned extra = shape == Shape::OutOfPlace ? FFTW_DESTROY_INPUT : 0u;
	// Planning may scribble on the buffers; callers refill before timing
	return fftwf_plan_dft_1d(static_cast<int>(b.N), in, out, FFTW_FORWARD, flags | extra);
}

std::string PlanText(fftwf_plan p)
{
	char* s = fftwf_sprint_plan(p);
	std::string text = s ? s : "";
	std::free(s);
	return text;
}

/// Short fingerprint so identical plans can be spotted in the table
std::string PlanId(const std::string& text)
{
	uint32_t h = 2166136261u;
	for (unsigned char c : text)
	{
		h = (h ^ c) * 16777619u;
	}
	char buf[16];
	std::snprintf(buf, sizeof(buf), "%08x", h);
	return buf;
}

/// Median over samples of the mean time of f(), in ns, after calibrating the
/// iteration count to opt.minTimeMs per sample
double TimeNs(const Options& opt, const std::function<void()>& f)
{
	uint64_t iters = 1;
	for (;;)
	{
		const auto s = Clock::now();
		for (uint64_t i = 0; i < iters; i++)
		{
			f();
		}
		const double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
		if (ms >= opt.minTimeMs * 0.5 || iters > (1ull << 30))
		{
			iters = std::max<uint64_t>(1, static_cast<uint64_t>(iters * opt.minTimeMs / std::max(ms, 1e-6)));
			break;
		}
		iters *= 4;
	}
	std::vector<double> v;
	for (int s = 0; s < opt.samples; s++)
	{
		const auto start = Clock::now();
		for (uint64_t i = 0; i < iters; i++)
		{
			f();
		}
		v.push_back(std::chrono::duration<double, std::nano>(Clock::now() - start).count() / static_cast<double>(iters));
	}
	std::sort(v.begin(), v.end());
	return v[v.size() / 2];
}

struct Timing
{
	double isolatedNs;  // one transform, input refilled each time (refill cost subtracted)
	double inSituNs;    // per segment of the CFftV4 loop: window, transform, accumulate
};

/// The refill keeps FFTW_DESTROY_INPUT from feeding the transform its own
/// garbage, which could drift into denormals or infinities and distort timing.
double RefillNs(const Options& opt, Buffers& b)
{
	return TimeNs(opt, [&] { std::memcpy(b.in, b.pristine.data(), sizeof(Complex) * b.N); });
}

Timing TimePlan(const Options& opt, Buffers& b, fftwf_plan p, const Complex* result, double refillNs,
	const std::vector<Complex>& input, const std::vector<float>& window, uint32_t stride)
{
	Timing t{};
	const double withRefill = TimeNs(opt, [&] {
		std::memcpy(b.in, b.pristine.data(), sizeof(Complex) * b.N);
		fftwf_execute(p);
	});
	t.isolatedNs = std::max(0.0, withRefill - refillNs);

	// Same work per segment as CFftV4::Compute (minus the once-per-call dB pass)
	std::vector<float> accum(b.N, 0.0f);
	const double perCall = TimeNs(opt, [&] {
		std::fill(accum.begin(), accum.end(), 0.0f);
		for (uint32_t k = 0; k < opt.segments; k++)
		{
			fftv4_kernels::ApplyWindow(input.data() + static_cast<size_t>(k) * stride, window.data(), b.in, b.N);
			fftwf_execute(p);
			fftv4_kernels::AccumulatePower(result, accum.data(), b.N);
		}
	});
	t.inSituNs = perCall / opt.segments;
	return t;
}

struct PlanResult
{
	std::string label;
	std::string id;
	std::string text;
	double planMs;
	Timing timing;
};

void CheckSize(const Options& opt, uint32_t N, Shape shape, int* problemsTimer, int* problemsCache)
{
	Buffers b(N);
	const uint32_t stride = N / 2;
	std::vector<Complex> input(N + static_cast<size_t>(opt.segments - 1) * stride);
	{
		std::mt19937 rng(11);
		std::normal_distribution<float> d(0.0f, 1.0f);
		for (auto& x : input)
		{
			x = { d(rng), d(rng) };
		}
	}
	// Blackman, normalized to sum 1, as in CFftV4
	std::vector<float> window(N);
	double sum = 0;
	for (uint32_t n = 0; n < N; n++)
	{
		const double w = 0.42 - 0.5 * std::cos(2.0 * kPi * n / (N - 1)) + 0.08 * std::cos(4.0 * kPi * n / (N - 1));
		window[n] = static_cast<float>(w);
		sum += w;
	}
	for (float& w : window)
	{
		w = static_cast<float>(w / sum);
	}

	// In-place plans leave their result in the input buffer
	const Complex* const result = shape == Shape::OutOfPlace ? b.out : b.in;

	std::vector<PlanResult> results;
	auto plan = [&](const std::string& label, unsigned flags) {
		fftwf_forget_wisdom();
		const auto s = Clock::now();
		fftwf_plan p = MakePlan(b, shape, flags);
		const double ms = std::chrono::duration<double, std::milli>(Clock::now() - s).count();
		if (!p)
		{
			std::printf("  %s: plan creation failed\n", label.c_str());
			return;
		}
		const double refill = RefillNs(opt, b);
		PlanResult r;
		r.label = label;
		r.text = PlanText(p);
		r.id = PlanId(r.text);
		r.planMs = ms;
		r.timing = TimePlan(opt, b, p, result, refill, input, window, stride);
		results.push_back(r);
		fftwf_destroy_plan(p);
	};

	plan("estimate", FFTW_ESTIMATE);
	for (int i = 1; i <= opt.measureRuns; i++)
	{
		plan("measure #" + std::to_string(i), FFTW_MEASURE);
	}
	fftwf_forget_wisdom();
	if (results.empty())
	{
		return;
	}

	std::printf("\nN=%u, %s (%s)\n", N, shape == Shape::OutOfPlace ? "out-of-place" : "in-place",
		shape == Shape::OutOfPlace ? "CFftV4" : "CFft/V2/V3");
	std::printf("  %-11s %-8s %9s %14s %14s %10s\n", "plan", "id", "plan ms", "isolated ns", "in situ ns/seg", "vs est.");
	const PlanResult& est = results.front();
	for (const PlanResult& r : results)
	{
		std::printf("  %-11s %-8s %9.2f %14.0f %14.0f %9.2fx\n", r.label.c_str(), r.id.c_str(), r.planMs,
			r.timing.isolatedNs, r.timing.inSituNs, r.timing.inSituNs / est.timing.inSituNs);
	}

	// Verdict over the Measure runs
	int distinct = 0;
	std::vector<std::string> seen;
	double isoRatio = 0, situRatio = 0;
	int measured = 0;
	for (size_t i = 1; i < results.size(); i++)
	{
		const PlanResult& r = results[i];
		if (std::find(seen.begin(), seen.end(), r.id) == seen.end())
		{
			seen.push_back(r.id);
			distinct++;
		}
		isoRatio += r.timing.isolatedNs / est.timing.isolatedNs;
		situRatio += r.timing.inSituNs / est.timing.inSituNs;
		measured++;
	}
	if (measured == 0)
	{
		return;
	}
	isoRatio /= measured;
	situRatio /= measured;
	std::printf("  Measure picked %d distinct plan(s) in %d run(s). On average its plans take %.2fx the Estimate\n"
		"  plan's time isolated and %.2fx in situ.\n", distinct, measured, isoRatio, situRatio);

	const bool slowerIsolated = isoRatio > 1.05;
	const bool slowerInSitu = situRatio > 1.05;
	const bool unstable = distinct > 1;
	if (slowerIsolated || (unstable && slowerInSitu))
	{
		(*problemsTimer)++;
		std::printf("  -> Looks like bad measurements: Measure's plans are slower %s%s.\n",
			slowerIsolated ? "even when timed alone" : "in situ",
			unstable ? ", and it doesn't pick the same plan twice" : "");
	}
	else if (slowerInSitu)
	{
		(*problemsCache)++;
		std::printf("  -> Looks like cache interaction: Measure's plans hold up alone but are slower in the real loop.\n");
	}
	else
	{
		std::printf("  -> Measure's plans are at least as fast as Estimate's here.\n");
	}

	if (opt.details)
	{
		for (const PlanResult& r : results)
		{
			std::printf("\n  --- %s plan %s\n%s\n", r.label.c_str(), r.id.c_str(), r.text.c_str());
		}
	}
}

std::vector<uint32_t> ParseList(const char* s)
{
	std::vector<uint32_t> v;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, ','))
	{
		const unsigned long n = std::strtoul(item.c_str(), nullptr, 10);
		if (n >= 2 && (n & (n - 1)) == 0)
		{
			v.push_back(static_cast<uint32_t>(n));
		}
		else
		{
			std::printf("ignoring %s: sizes must be powers of 2\n", item.c_str());
		}
	}
	return v;
}
} // namespace

int main(int argc, char** argv)
{
	Options opt;
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		const bool hasValue = i + 1 < argc;
		if (a == "--sizes" && hasValue) opt.sizes = ParseList(argv[++i]);
		else if (a == "--measure-runs" && hasValue) opt.measureRuns = std::max(1, std::atoi(argv[++i]));
		else if (a == "--min-time" && hasValue) opt.minTimeMs = std::atof(argv[++i]);
		else if (a == "--samples" && hasValue) opt.samples = std::max(1, std::atoi(argv[++i]));
		else if (a == "--details") opt.details = true;
		else
		{
			std::printf("usage: fft_plan_check [--sizes a,b] [--measure-runs n] [--min-time ms] [--samples n] [--details]\n");
			return 2;
		}
	}

	std::printf("fftw: %s\n", fftwf_version);
	std::printf("Timing each plan alone (isolated) and in CFftV4's per-segment loop (in situ, %u segments).\n", opt.segments);
	std::printf("Wisdom is forgotten before every plan, so each Measure run starts cold.\n");

	int timer = 0, cache = 0;
	for (uint32_t N : opt.sizes)
	{
		CheckSize(opt, N, Shape::OutOfPlace, &timer, &cache);
		CheckSize(opt, N, Shape::InPlace, &timer, &cache);
	}

	std::printf("\nSummary: %d case(s) look like bad measurements, %d look like cache interaction.\n", timer, cache);
	if (timer > 0)
	{
		std::printf("Bad measurements usually mean FFTW has no cycle counter on this CPU. Use FFTW_ESTIMATE\n"
			"(PlanEffort::Estimate) here, and don't ship wisdom files made on this machine.\n");
	}
	else if (cache > 0)
	{
		std::printf("FFTW's isolated timing doesn't reflect the real loop. Use FFTW_ESTIMATE here, or generate\n"
			"wisdom some other way (for example, pick plans by timing CFftV4 itself).\n");
	}
	return 0;
}
