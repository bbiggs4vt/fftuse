// Performance harness for the CFft family.
//
// Times Compute() (and construction) for each registered implementation over a
// grid of FFT sizes and output-bin counts, and reports speedup against the first
// implementation (the original CFft). Results can be appended to a CSV so you can
// track progress across iterations:
//
//   ./fft_bench                          # default grid, table to stdout
//   ./fft_bench --quick                  # smaller grid, shorter runs
//   ./fft_bench --csv results.csv --label "v2 first pass"
//   ./fft_bench --sizes 1024,8192 --segments 64 --overlap 0.75 --min-time 200
//   ./fft_bench --only v2,v4             # just these (v1 always runs as the baseline)
//   ./fft_bench --only "v4*"             # v4 and v4-measure
//   ./fft_bench --machine pi5 --csv results.csv   # name the machine in the CSV
//   ./fft_bench --info                   # print the detected machine/build details and exit
//   ./fft_bench --wisdom fftwf.wisdom    # construct with this wisdom loaded (warm start);
//                                        # make it with tools/fft_wisdom
//
// To add a new version (e.g. CFftV3), include its header and add a line to
// Implementations() below.

#include "stdafx.h"
#include "Fft.h"
#include "FftV2.h"
#include "FftV3.h"
#include "FftV4.h"
#include "FftwWisdom.h"
#include "machine_info.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
using Complex = std::complex<float>;

/// Type-erased wrapper so different classes can be benchmarked the same way
struct IRunner
{
	virtual ~IRunner() = default;
	virtual void Compute(const Complex* in, uint32_t length, float* out, uint32_t bins) = 0;
	virtual uint32_t FftSize() const = 0;
};

template <typename T>
struct Runner : IRunner
{
	template <typename... Args>
	explicit Runner(Args&&... args) : fft(std::forward<Args>(args)...) {}
	void Compute(const Complex* in, uint32_t length, float* out, uint32_t bins) override
	{
		fft.Compute(in, length, out, 0.0f, bins);
	}
	uint32_t FftSize() const override { return const_cast<T&>(fft).GetFftSize(); }
	T fft;
};

struct Impl
{
	std::string name;
	std::function<std::unique_ptr<IRunner>(uint32_t fftSize, float overlap)> make;
};

std::vector<Impl> Implementations()
{
	return {
		// The first entry is the baseline for the speedup column
		{ "v1", [](uint32_t n, float ov) { return std::unique_ptr<IRunner>(new Runner<CFft>(n, ov)); } },
		{ "v2", [](uint32_t n, float ov) { return std::unique_ptr<IRunner>(new Runner<CFftV2>(n, ov)); } },
		{ "v2-measure", [](uint32_t n, float ov) {
			 return std::unique_ptr<IRunner>(new Runner<CFftV2>(n, ov, false, CFftV2::PlanEffort::Measure));
		 } },
		// v3: batched segments. "v3" picks the batch size automatically (~256 KB buffer).
		{ "v3", [](uint32_t n, float ov) { return std::unique_ptr<IRunner>(new Runner<CFftV3>(n, ov)); } },
		{ "v3-b4", [](uint32_t n, float ov) {
			 return std::unique_ptr<IRunner>(new Runner<CFftV3>(n, ov, false, CFftV3::PlanEffort::Estimate, 4));
		 } },
		{ "v3-b16", [](uint32_t n, float ov) {
			 return std::unique_ptr<IRunner>(new Runner<CFftV3>(n, ov, false, CFftV3::PlanEffort::Estimate, 16));
		 } },
		{ "v3-measure", [](uint32_t n, float ov) {
			 return std::unique_ptr<IRunner>(new Runner<CFftV3>(n, ov, false, CFftV3::PlanEffort::Measure));
		 } },
		// v4: out-of-place FFT, float-wise window, vectorized fast dB
		{ "v4", [](uint32_t n, float ov) { return std::unique_ptr<IRunner>(new Runner<CFftV4>(n, ov)); } },
		{ "v4-measure", [](uint32_t n, float ov) {
			 return std::unique_ptr<IRunner>(new Runner<CFftV4>(n, ov, false, CFftV4::PlanEffort::Measure));
		 } },
		{ "v4-wisdom", [](uint32_t n, float ov) {
			 return std::unique_ptr<IRunner>(new Runner<CFftV4>(n, ov, false, CFftV4::PlanEffort::WisdomOrEstimate));
		 } },
	};
}

struct Options
{
	std::vector<uint32_t> sizes = { 256, 1024, 4096, 16384, 65536 };
	std::vector<uint32_t> binDivisors = { 1, 4 }; // bins = N / divisor
	uint32_t segments = 32;
	float overlap = 0.5f;
	double minTimeMs = 100.0; // per sample
	int samples = 7;
	std::string csvPath;
	std::string label;
	std::string only;
	std::string wisdomPath; // empty: every construction is cold
	std::string machine;    // label for this machine; host name if empty
	bool infoOnly = false;
};

struct Result
{
	double ctorMs;
	double medianNs;
	double minNs;
	float maxDiffDb; // vs baseline output
};

std::vector<Complex> Noise(size_t n)
{
	std::mt19937 rng(12345);
	std::normal_distribution<float> d(0.0f, 1.0f);
	std::vector<Complex> v(n);
	for (auto& x : v)
	{
		x = { d(rng), d(rng) };
	}
	return v;
}

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

volatile float gSink = 0.0f; // keeps the optimizer honest

Result Measure(const Impl& impl, const Options& opt, uint32_t requestedSize, uint32_t binDivisor,
	const std::vector<Complex>& input, uint32_t length, std::vector<float>& out)
{
	Result r{};

	// Drop plans FFTW remembers from earlier constructions so every constructor
	// time is a cold one (otherwise Measure/Patient look free after the first).
	// With --wisdom, start from that file instead: the warm-start case.
	FftwWisdom::Forget();
	if (!opt.wisdomPath.empty())
	{
		FftwWisdom::Load(opt.wisdomPath);
	}
	const auto t0 = Clock::now();
	std::unique_ptr<IRunner> fft = impl.make(requestedSize, opt.overlap);
	r.ctorMs = Ms(Clock::now() - t0);

	const uint32_t N = fft->FftSize();
	const uint32_t bins = std::max<uint32_t>(1, N / binDivisor);
	out.assign(bins, 0.0f);

	// Warm up and calibrate iterations per sample
	uint64_t iters = 1;
	for (;;)
	{
		const auto s = Clock::now();
		for (uint64_t i = 0; i < iters; i++)
		{
			fft->Compute(input.data(), length, out.data(), bins);
		}
		const double ms = Ms(Clock::now() - s);
		if (ms >= opt.minTimeMs * 0.5 || iters > (1ull << 30))
		{
			iters = std::max<uint64_t>(1, static_cast<uint64_t>(iters * opt.minTimeMs / std::max(ms, 1e-6)));
			break;
		}
		iters *= 4;
	}

	std::vector<double> perCall;
	for (int s = 0; s < opt.samples; s++)
	{
		const auto start = Clock::now();
		for (uint64_t i = 0; i < iters; i++)
		{
			fft->Compute(input.data(), length, out.data(), bins);
		}
		perCall.push_back(Ms(Clock::now() - start) * 1e6 / static_cast<double>(iters));
		gSink = gSink + out[bins / 2];
	}
	std::sort(perCall.begin(), perCall.end());
	r.medianNs = perCall[perCall.size() / 2];
	r.minNs = perCall.front();
	return r;
}

/// --only takes a comma-separated list. An entry matches a name exactly, or as a
/// prefix when it ends in '*' (e.g. "v4*" selects v4 and v4-measure).
bool Selected(const std::string& name, const std::string& only)
{
	if (only.empty())
	{
		return true;
	}
	std::stringstream ss(only);
	std::string tok;
	while (std::getline(ss, tok, ','))
	{
		if (!tok.empty() && tok.back() == '*')
		{
			if (name.compare(0, tok.size() - 1, tok, 0, tok.size() - 1) == 0)
			{
				return true;
			}
		}
		else if (name == tok)
		{
			return true;
		}
	}
	return false;
}

std::vector<uint32_t> ParseList(const char* s)
{
	std::vector<uint32_t> v;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, ','))
	{
		v.push_back(static_cast<uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
	}
	return v;
}

void Usage()
{
	std::printf(
		"usage: fft_bench [--quick] [--sizes a,b,c] [--bins-div a,b] [--segments K]\n"
		"                 [--overlap f] [--min-time ms] [--samples n] [--only a,b,prefix*]\n"
		"                 [--csv path] [--label text] [--wisdom file] [--machine name] [--info]\n");
}

bool ParseArgs(int argc, char** argv, Options& opt)
{
	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		auto next = [&]() -> const char* {
			if (i + 1 >= argc)
			{
				std::printf("missing value for %s\n", a.c_str());
				std::exit(2);
			}
			return argv[++i];
		};
		if (a == "--quick")
		{
			opt.sizes = { 1024, 8192, 65536 };
			opt.binDivisors = { 1 };
			opt.minTimeMs = 30.0;
			opt.samples = 5;
		}
		else if (a == "--sizes") opt.sizes = ParseList(next());
		else if (a == "--bins-div") opt.binDivisors = ParseList(next());
		else if (a == "--segments") opt.segments = static_cast<uint32_t>(std::atoi(next()));
		else if (a == "--overlap") opt.overlap = static_cast<float>(std::atof(next()));
		else if (a == "--min-time") opt.minTimeMs = std::atof(next());
		else if (a == "--samples") opt.samples = std::max(1, std::atoi(next()));
		else if (a == "--only") opt.only = next();
		else if (a == "--csv") opt.csvPath = next();
		else if (a == "--label") opt.label = next();
		else if (a == "--wisdom") opt.wisdomPath = next();
		else if (a == "--machine") opt.machine = next();
		else if (a == "--info") opt.infoOnly = true;
		else
		{
			Usage();
			return false;
		}
	}
	return true;
}
} // namespace

int main(int argc, char** argv)
{
	Options opt;
	if (!ParseArgs(argc, argv, opt))
	{
		return 2;
	}

	const MachineInfo mi = DetectMachine(opt.machine);
	std::printf("machine:  %s\n", mi.machine.c_str());
	std::printf("cpu:      %s (%s, %u threads, governor %s)\n", mi.cpu.c_str(), mi.arch.c_str(), mi.cores, mi.governor.c_str());
	std::printf("compiler: %s, %s, simd %s\n", mi.compiler.c_str(), mi.build.c_str(), mi.simd.c_str());
	std::printf("fftw:     %s\n", mi.fftw.c_str());
	if ((mi.arch == "arm32" || mi.arch == "aarch64") && mi.fftw.find("neon") == std::string::npos)
	{
		std::printf("note:     FFTW was built without NEON; the FFT is most of the run time, so an FFTW\n"
			"          configured with --enable-float --enable-neon is likely the biggest speedup here\n");
	}
	if (mi.governor != "performance" && mi.governor != "unknown")
	{
		std::printf("note:     CPU governor is '%s'; results vary more than with 'performance'\n", mi.governor.c_str());
	}
	std::printf("\n");
	if (opt.infoOnly)
	{
		return 0;
	}

	if (!opt.wisdomPath.empty() && !FftwWisdom::Load(opt.wisdomPath))
	{
		std::printf("can't load wisdom from %s\n", opt.wisdomPath.c_str());
		return 2;
	}

	// Keep warm-start rows distinguishable in the CSV history
	if (!opt.wisdomPath.empty())
	{
		opt.label += opt.label.empty() ? "[wisdom]" : " [wisdom]";
	}

	std::vector<Impl> impls = Implementations();
	const std::string baselineName = impls.front().name;

	// Rows carry the machine and build so CSVs from different machines can be merged
	const std::string csvHeader = "timestamp,label,impl,fft_size,bins,segments,overlap,ctor_ms,median_ns,min_ns,"
		"ns_per_segment,msamples_per_s,speedup_vs_" + baselineName + ",max_diff_db,"
		"machine,arch,cpu,threads,governor,compiler,build,simd,fftw";
	const std::string machineCsv = "\"" + CsvText(mi.machine) + "\"," + mi.arch + ",\"" + CsvText(mi.cpu) + "\"," +
		std::to_string(mi.cores) + "," + mi.governor + ",\"" + CsvText(mi.compiler) + "\",\"" + CsvText(mi.build) +
		"\"," + mi.simd + ",\"" + CsvText(mi.fftw) + "\"";

	FILE* csv = nullptr;
	if (!opt.csvPath.empty())
	{
		std::string existingHeader;
		{
			std::ifstream probe(opt.csvPath);
			if (probe)
			{
				std::getline(probe, existingHeader);
			}
		}
		if (!existingHeader.empty() && existingHeader != csvHeader)
		{
			std::printf("%s has different columns (from an older fft_bench); use a new CSV file\n", opt.csvPath.c_str());
			return 2;
		}
		csv = std::fopen(opt.csvPath.c_str(), "a");
		if (!csv)
		{
			std::printf("can't open %s\n", opt.csvPath.c_str());
			return 2;
		}
		if (existingHeader.empty())
		{
			std::fprintf(csv, "%s\n", csvHeader.c_str());
		}
	}
	char stamp[32];
	const std::time_t now = std::time(nullptr);
	std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

	std::printf("segments/call=%u overlap=%.2f min-time=%.0fms samples=%d wisdom=%s%s%s\n\n", opt.segments, opt.overlap,
		opt.minTimeMs, opt.samples, opt.wisdomPath.empty() ? "none (cold)" : opt.wisdomPath.c_str(),
		opt.label.empty() ? "" : " label=", opt.label.c_str());
	std::printf("%-12s %7s %7s %10s %12s %12s %10s %9s %10s\n", "impl", "N", "bins", "ctor ms", "us/call",
		"ns/segment", "Msamp/s", "speedup", "max dB diff");
	std::printf("%s\n", std::string(98, '-').c_str());

	for (uint32_t size : opt.sizes)
	{
		// Input sized for exactly opt.segments segments
		CFftV2 probe(size, opt.overlap);
		const uint32_t N = probe.GetFftSize();
		const uint32_t length = N + (opt.segments - 1) * probe.GetStride();
		const std::vector<Complex> input = Noise(length);

		for (uint32_t div : opt.binDivisors)
		{
			std::vector<float> baselineOut;
			double baselineNs = 0.0;
			for (size_t i = 0; i < impls.size(); i++)
			{
				const Impl& impl = impls[i];
				const bool isBaseline = (i == 0);
				if (!isBaseline && !Selected(impl.name, opt.only))
				{
					continue;
				}
				std::vector<float> out;
				Result r = Measure(impl, opt, size, div, input, length, out);
				if (isBaseline)
				{
					baselineOut = out;
					baselineNs = r.medianNs;
				}
				float maxDiff = 0.0f;
				for (size_t b = 0; b < out.size() && b < baselineOut.size(); b++)
				{
					maxDiff = std::max(maxDiff, std::fabs(out[b] - baselineOut[b]));
				}
				r.maxDiffDb = maxDiff;

				const uint32_t bins = static_cast<uint32_t>(out.size());
				const double nsPerSeg = r.medianNs / opt.segments;
				const double msps = static_cast<double>(opt.segments) * N / r.medianNs * 1e3;
				const double speedup = baselineNs / r.medianNs;
				std::printf("%-12s %7u %7u %10.3f %12.2f %12.1f %10.1f %8.2fx %10.2g\n", impl.name.c_str(), N, bins,
					r.ctorMs, r.medianNs / 1e3, nsPerSeg, msps, speedup, r.maxDiffDb);
				if (csv)
				{
					std::fprintf(csv, "%s,\"%s\",%s,%u,%u,%u,%.3f,%.4f,%.1f,%.1f,%.2f,%.2f,%.4f,%.3g,%s\n", stamp,
						CsvText(opt.label).c_str(), impl.name.c_str(), N, bins, opt.segments, opt.overlap, r.ctorMs, r.medianNs,
						r.minNs, nsPerSeg, msps, speedup, r.maxDiffDb, machineCsv.c_str());
				}
			}
			std::printf("\n");
		}
	}

	if (csv)
	{
		std::fclose(csv);
		std::printf("appended results to %s\n", opt.csvPath.c_str());
	}
	return 0;
}
