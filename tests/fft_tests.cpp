// Correctness tests for CFftV2, including equivalence with the original CFft on
// valid inputs. Run: ./fft_tests  (exit code is the number of failures, capped).

#include "stdafx.h"
#include "Fft.h"
#include "FftV2.h"
#include "FftV3.h"
#include "FftV4.h"
#include "FftwWisdom.h"

#include <chrono>
#include <fstream>

#include <cfloat>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace
{
int gFailures = 0;
int gChecks = 0;

#define CHECK(cond, ...)                                                    \
	do                                                                      \
	{                                                                       \
		gChecks++;                                                          \
		if (!(cond))                                                        \
		{                                                                   \
			gFailures++;                                                    \
			std::printf("  FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);   \
			std::printf(__VA_ARGS__);                                       \
			std::printf("\n");                                              \
		}                                                                   \
	} while (0)

template <typename Ex>
bool Throws(const std::function<void()>& f)
{
	try
	{
		f();
	}
	catch (const Ex&)
	{
		return true;
	}
	catch (...)
	{
		return false;
	}
	return false;
}

std::vector<std::complex<float>> Noise(size_t n, uint32_t seed)
{
	std::mt19937 rng(seed);
	std::normal_distribution<float> d(0.0f, 1.0f);
	std::vector<std::complex<float>> v(n);
	for (auto& x : v)
	{
		x = { d(rng), d(rng) };
	}
	return v;
}

std::vector<std::complex<float>> Tone(size_t n, double cyclesPerSample, float amplitude)
{
	std::vector<std::complex<float>> v(n);
	for (size_t i = 0; i < n; i++)
	{
		const double ph = 2.0 * 3.14159265358979323846 * cyclesPerSample * static_cast<double>(i);
		v[i] = { static_cast<float>(amplitude * std::cos(ph)), static_cast<float>(amplitude * std::sin(ph)) };
	}
	return v;
}

void TestTypeProperties()
{
	std::printf("type properties\n");
	CHECK(!std::is_copy_constructible<CFftV2>::value, "CFftV2 must not be copyable");
	CHECK(!std::is_copy_assignable<CFftV2>::value, "CFftV2 must not be copy-assignable");
	// The original is copyable, which double-frees the FFTW buffer and plan
	CHECK(std::is_copy_constructible<CFft>::value, "(documenting v1: expected copyable)");
}

void TestConstructorValidation()
{
	std::printf("constructor validation\n");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f(0, 0.5f); }), "fftSize 0");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f(1, 0.5f); }), "fftSize 1");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f((1u << 30) + 1, 0.5f); }), "fftSize too big");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f(1024, 1.0f); }), "overlap 1");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f(1024, 1.5f); }), "overlap 1.5");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f(1024, -0.1f); }), "overlap negative");
	CHECK(Throws<std::invalid_argument>([] { CFftV2 f(1024, std::nanf("")); }), "overlap NaN");

	CFftV2 a(1000, 0.5f);
	CHECK(a.GetFftSize() == 1024, "got %u", a.GetFftSize());
	CHECK(a.GetStride() == 512, "got %u", a.GetStride());
	CFftV2 b(1024, 0.0f);
	CHECK(b.GetFftSize() == 1024 && b.GetStride() == 1024, "size %u stride %u", b.GetFftSize(), b.GetStride());
	CFftV2 c(2, 0.99999f);
	CHECK(c.GetStride() >= 1, "stride must be at least 1, got %u", c.GetStride());
	CFftV2 d(1025, 0.75f);
	CHECK(d.GetFftSize() == 2048 && d.GetStride() == 512, "size %u stride %u", d.GetFftSize(), d.GetStride());
}

void TestComputeValidation()
{
	std::printf("compute validation\n");
	CFftV2 f(1024, 0.5f);
	std::vector<float> out(1024);
	auto in = Noise(1023, 1);
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(in.data(), 1023, out.data()); }), "short input");
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(in.data(), 100, out.data()); }), "very short input");
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(in.data(), 0, out.data()); }), "zero-length input");
	auto ok = Noise(2048, 2);
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(ok.data(), 2048, out.data(), 0.0f, 1025); }), "bins > N");
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(nullptr, 2048, out.data()); }), "null input");
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(ok.data(), 2048, nullptr); }), "null output");

	// Vector overload sizes its own output and validates empty input
	std::vector<float> v;
	f.Compute(ok, v);
	CHECK(v.size() == 1024, "got %zu", v.size());
	f.Compute(ok, v, 0.0f, 100);
	CHECK(v.size() == 100, "got %zu", v.size());
	std::vector<std::complex<float>> empty;
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(empty, v); }), "empty vector input");
	CHECK(Throws<std::invalid_argument>([&] { f.Compute(ok, v, 0.0f, 2000); }), "vector bins > N");

	// Exactly one segment is fine
	f.Compute(ok.data(), 1024, out.data());
	CHECK(std::isfinite(out[0]), "one segment should produce finite output");
}

void TestFloor()
{
	std::printf("zero input -> floor\n");
	CFftV2 f(256, 0.5f);
	std::vector<std::complex<float>> zeros(1024);
	std::vector<float> out;
	f.Compute(zeros, out);
	bool allFloor = true;
	for (float x : out)
	{
		allFloor &= (x == CFftV2::kDefaultFloorDb);
	}
	CHECK(allFloor, "expected %g dB", CFftV2::kDefaultFloorDb);
	f.SetFloorDb(-150.0f);
	f.Compute(zeros, out);
	CHECK(out[10] == -150.0f, "got %g", out[10]);
}

void TestTone()
{
	std::printf("tone lands in the right bin at 0 dB\n");
	const uint32_t N = 1024;
	CFftV2 f(N, 0.5f);
	// Bin-centered tone at bin +100 from DC. After fftshift DC is at index N/2.
	auto in = Tone(N * 8, 100.0 / N, 1.0f);
	std::vector<float> out;
	f.Compute(in, out);
	const auto peak = std::max_element(out.begin(), out.end()) - out.begin();
	CHECK(peak == N / 2 + 100, "peak at %ld", static_cast<long>(peak));
	CHECK(std::fabs(out[peak]) < 0.01f, "peak %g dB, expected 0 (window normalized to sum 1)", out[peak]);

	// Centered subset of bins keeps DC in the middle
	f.Compute(in, out, 0.0f, 256);
	const auto peak2 = std::max_element(out.begin(), out.end()) - out.begin();
	CHECK(peak2 == 128 + 100, "peak at %ld", static_cast<long>(peak2));

	// Shift is additive
	f.Compute(in, out, 7.5f);
	CHECK(std::fabs(out[N / 2 + 100] - 7.5f) < 0.01f, "got %g", out[N / 2 + 100]);

	// Negative frequency
	auto neg = Tone(N * 4, -37.0 / N, 0.5f);
	f.Compute(neg, out);
	const auto peak3 = std::max_element(out.begin(), out.end()) - out.begin();
	CHECK(peak3 == N / 2 - 37, "peak at %ld", static_cast<long>(peak3));
	CHECK(std::fabs(out[peak3] - 20.0f * std::log10(0.5f)) < 0.01f, "got %g", out[peak3]);
}

// v2 must match v1 wherever v1 is well-defined (input >= N, bins <= N, non-zero bins)
void TestEquivalenceWithV1()
{
	std::printf("equivalence with v1\n");
	struct Case
	{
		uint32_t size;
		float overlap;
		bool reverse;
		uint32_t extra; // samples beyond the first segment
		uint32_t bins;
		float shift;
	};
	const Case cases[] = {
		{ 64, 0.5f, false, 0, 0, 0.0f },
		{ 64, 0.5f, false, 31, 0, 0.0f },
		{ 1000, 0.5f, false, 5000, 0, 3.0f },
		{ 1024, 0.75f, false, 4096, 300, 0.0f },
		{ 1024, 0.0f, false, 10000, 1, 0.0f },
		{ 4096, 0.5f, false, 40000, 1023, -10.0f },
		{ 4096, 0.3f, false, 4096 * 3 + 17, 4096, 0.0f },
		{ 512, 0.5f, true, 2048, 0, 0.0f },
		{ 512, 0.25f, true, 2048, 77, 0.0f },
		{ 65536, 0.5f, false, 65536 * 2, 0, 0.0f },
	};
	int idx = 0;
	for (const Case& c : cases)
	{
		CFft v1(c.size, c.overlap, c.reverse);
		CFftV2 v2(c.size, c.overlap, c.reverse);
		CHECK(v1.GetFftSize() == v2.GetFftSize(), "case %d size", idx);
		const uint32_t N = v2.GetFftSize();
		const uint32_t len = N + c.extra;
		const uint32_t bins = c.bins == 0 ? N : c.bins;
		auto in = Noise(len, 100 + idx);
		std::vector<float> o1(bins), o2(bins);
		v1.Compute(in.data(), len, o1.data(), c.shift, c.bins);
		v2.Compute(in.data(), len, o2.data(), c.shift, c.bins);
		float maxDiff = 0.0f;
		for (uint32_t i = 0; i < bins; i++)
		{
			maxDiff = std::max(maxDiff, std::fabs(o1[i] - o2[i]));
		}
		CHECK(maxDiff < 1e-3f, "case %d (N=%u ov=%g rev=%d bins=%u): max diff %g dB", idx, N, c.overlap, c.reverse, bins, maxDiff);
		idx++;
	}

	// Planning effort must not change results beyond float noise
	CFftV2 est(2048, 0.5f, false, CFftV2::PlanEffort::Estimate);
	CFftV2 mea(2048, 0.5f, false, CFftV2::PlanEffort::Measure);
	auto in = Noise(2048 * 6, 7);
	std::vector<float> a, b;
	est.Compute(in, a);
	mea.Compute(in, b);
	float maxDiff = 0.0f;
	for (size_t i = 0; i < a.size(); i++)
	{
		maxDiff = std::max(maxDiff, std::fabs(a[i] - b[i]));
	}
	CHECK(maxDiff < 1e-3f, "estimate vs measure diff %g dB", maxDiff);
}

void TestRepeatedCallsIndependent()
{
	std::printf("repeated calls don't leak state\n");
	CFftV2 f(512, 0.5f);
	auto a = Noise(4096, 11);
	auto b = Noise(2000, 12);
	std::vector<float> r1, r2, r3;
	f.Compute(a, r1);
	f.Compute(b, r2);
	f.Compute(a, r3);
	CHECK(r1 == r3, "same input should give bit-identical output");
}
void TestV3()
{
	std::printf("v3 validation\n");
	CHECK(!std::is_copy_constructible<CFftV3>::value, "CFftV3 must not be copyable");
	CHECK(Throws<std::invalid_argument>([] { CFftV3 f(0, 0.5f); }), "fftSize 0");
	CHECK(Throws<std::invalid_argument>([] { CFftV3 f(1024, 1.0f); }), "overlap 1");
	CHECK(Throws<std::invalid_argument>([] { CFftV3 f(1024, 0.5f, false, CFftV3::PlanEffort::Estimate, CFftV3::kMaxBatchSize + 1); }), "batch too big");
	{
		CFftV3 f(1024, 0.5f);
		std::vector<float> out(1024);
		auto in = Noise(1000, 1);
		CHECK(Throws<std::invalid_argument>([&] { f.Compute(in.data(), 1000, out.data()); }), "short input");
		CHECK(Throws<std::invalid_argument>([&] { f.Compute(in.data(), 1000, out.data(), 0.0f, 2000); }), "bins > N");
		std::vector<float> v;
		auto ok = Noise(4096, 2);
		f.Compute(ok, v, 0.0f, 10);
		CHECK(v.size() == 10, "got %zu", v.size());
	}
	// Auto batch size targets kAutoBatchBytes, at least 1
	CHECK(CFftV3(1024, 0.5f).GetBatchSize() == 32, "got %u", CFftV3(1024, 0.5f).GetBatchSize());
	CHECK(CFftV3(1u << 20, 0.5f).GetBatchSize() == 1, "got %u", CFftV3(1u << 20, 0.5f).GetBatchSize());
	CHECK(CFftV3(16, 0.5f).GetBatchSize() == CFftV3::kMaxBatchSize, "got %u", CFftV3(16, 0.5f).GetBatchSize());

	std::printf("v3 equivalence with v2 (whole batches + leftovers)\n");
	const uint32_t batches[] = { 0, 1, 3, 4, 16 };
	const uint32_t segmentCounts[] = { 1, 2, 3, 4, 5, 15, 16, 17, 33, 64 };
	const float overlaps[] = { 0.5f, 0.0f, 0.75f };
	int idx = 0;
	for (float ov : overlaps)
	{
		for (uint32_t batch : batches)
		{
			for (uint32_t segs : segmentCounts)
			{
				for (int reverse = 0; reverse < 2; reverse++)
				{
					CFftV2 v2(256, ov, reverse != 0);
					CFftV3 v3(256, ov, reverse != 0, CFftV3::PlanEffort::Estimate, batch);
					// Some extra samples that don't make a full segment
					const uint32_t len = 256 + (segs - 1) * v2.GetStride() + 7;
					auto in = Noise(len, 500 + idx);
					std::vector<float> a, b;
					v2.Compute(in, a, 1.0f, 100);
					v3.Compute(in, b, 1.0f, 100);
					float maxDiff = 0.0f;
					for (size_t i = 0; i < a.size(); i++)
					{
						maxDiff = std::max(maxDiff, std::fabs(a[i] - b[i]));
					}
					CHECK(maxDiff < 1e-3f, "ov=%g batch=%u segs=%u rev=%d: diff %g dB", ov, batch, segs, reverse, maxDiff);
					idx++;
				}
			}
		}
	}

	// Larger sizes, Measure planning, and the tone check
	for (uint32_t N : { 4096u, 65536u })
	{
		CFftV2 v2(N, 0.5f);
		CFftV3 v3(N, 0.5f, false, CFftV3::PlanEffort::Measure, 3);
		auto in = Noise(N * 6, N);
		std::vector<float> a, b;
		v2.Compute(in, a);
		v3.Compute(in, b);
		float maxDiff = 0.0f;
		for (size_t i = 0; i < a.size(); i++)
		{
			maxDiff = std::max(maxDiff, std::fabs(a[i] - b[i]));
		}
		CHECK(maxDiff < 1e-3f, "N=%u measure: diff %g dB", N, maxDiff);
	}
	{
		CFftV3 f(1024, 0.5f);
		auto in = Tone(1024 * 20, 100.0 / 1024, 1.0f);
		std::vector<float> out;
		f.Compute(in, out);
		const auto peak = std::max_element(out.begin(), out.end()) - out.begin();
		CHECK(peak == 512 + 100 && std::fabs(out[peak]) < 0.01f, "peak %ld at %g dB", static_cast<long>(peak), out[peak]);
		// Leftover state from a big call must not leak into a small one
		std::vector<float> r1, r2;
		auto small = Noise(1024, 9);
		f.Compute(small, r1);
		f.Compute(in, out);
		f.Compute(small, r2);
		CHECK(r1 == r2, "same input should give bit-identical output");
	}
}
float MaxDiff(const std::vector<float>& a, const std::vector<float>& b)
{
	float d = (a.size() == b.size()) ? 0.0f : 1e30f;
	for (size_t i = 0; i < a.size() && i < b.size(); i++)
	{
		d = std::max(d, std::fabs(a[i] - b[i]));
	}
	return d;
}

void TestV4()
{
	std::printf("v4 validation\n");
	CHECK(!std::is_copy_constructible<CFftV4>::value, "CFftV4 must not be copyable");
	CHECK(Throws<std::invalid_argument>([] { CFftV4 f(0, 0.5f); }), "fftSize 0");
	CHECK(Throws<std::invalid_argument>([] { CFftV4 f(1024, 1.0f); }), "overlap 1");
	CHECK(Throws<std::invalid_argument>([] { CFftV4 f(1024, std::nanf("")); }), "overlap NaN");
	{
		CFftV4 f(1024, 0.5f);
		std::vector<float> out(1024);
		auto in = Noise(1000, 1);
		CHECK(Throws<std::invalid_argument>([&] { f.Compute(in.data(), 1000, out.data()); }), "short input");
		auto ok = Noise(4096, 2);
		CHECK(Throws<std::invalid_argument>([&] { f.Compute(ok.data(), 4096, out.data(), 0.0f, 1025); }), "bins > N");
		CHECK(Throws<std::invalid_argument>([&] { f.Compute(nullptr, 4096, out.data()); }), "null input");
		std::vector<float> v;
		f.Compute(ok, v, 0.0f, 10);
		CHECK(v.size() == 10, "got %zu", v.size());
		std::vector<std::complex<float>> empty;
		CHECK(Throws<std::invalid_argument>([&] { f.Compute(empty, v); }), "empty vector input");
	}

	std::printf("v4 equivalence with v2\n");
	// Every bins value hits a different split of the fftshift into two runs
	const uint32_t sizes[] = { 2, 4, 64, 256, 1000, 4096, 65536 };
	const float overlaps[] = { 0.5f, 0.0f, 0.75f };
	int idx = 0;
	for (uint32_t size : sizes)
	{
		for (float ov : overlaps)
		{
			for (int reverse = 0; reverse < 2; reverse++)
			{
				CFftV2 v2(size, ov, reverse != 0);
				CFftV4 v4(size, ov, reverse != 0);
				const uint32_t N = v2.GetFftSize();
				auto in = Noise(N * 3 + 5, 900 + idx++);
				const uint32_t binChoices[] = { 0, 1, 2, 3, N / 2, N / 2 + 1, N - 1, N };
				for (uint32_t bins : binChoices)
				{
					if (bins > N)
					{
						continue;
					}
					std::vector<float> a, b;
					v2.Compute(in, a, -4.0f, bins);
					v4.Compute(in, b, -4.0f, bins);
					const float d = MaxDiff(a, b);
					CHECK(d < 1e-3f, "N=%u ov=%g rev=%d bins=%u: diff %g dB", N, ov, reverse, bins, d);
				}
			}
		}
	}

	std::printf("v4 fast log across 60 decades of power\n");
	{
		CFftV2 v2(512, 0.5f);
		CFftV4 v4(512, 0.5f);
		auto base = Noise(512 * 4, 77);
		float worst = 0.0f;
		for (int decade = -15; decade <= 15; decade++)
		{
			const float scale = std::pow(10.0f, static_cast<float>(decade));
			std::vector<std::complex<float>> in(base.size());
			for (size_t i = 0; i < in.size(); i++)
			{
				in[i] = base[i] * scale;
			}
			std::vector<float> a, b;
			v2.Compute(in, a);
			v4.Compute(in, b);
			worst = std::max(worst, MaxDiff(a, b));
		}
		CHECK(worst < 1e-3f, "worst diff %g dB", worst);
	}

	std::printf("v4 special values\n");
	{
		CFftV4 f(256, 0.5f);
		std::vector<std::complex<float>> zeros(1024);
		std::vector<float> out;
		f.Compute(zeros, out);
		CHECK(*std::max_element(out.begin(), out.end()) == CFftV4::kDefaultFloorDb, "zeros -> floor");
		// A big shift must not lift empty bins above the floor
		f.Compute(zeros, out, 500.0f);
		CHECK(*std::max_element(out.begin(), out.end()) == CFftV4::kDefaultFloorDb, "zeros + shift -> floor, got %g", out[0]);
		f.SetFloorDb(-150.0f);
		f.Compute(zeros, out);
		CHECK(out[3] == -150.0f, "custom floor, got %g", out[3]);

		CFftV2 g2(256, 0.5f);
		CFftV4 g4(256, 0.5f);
		std::vector<std::complex<float>> nan(1024, { std::nanf(""), 0.0f });
		std::vector<float> a, b;
		g2.Compute(nan, a);
		g4.Compute(nan, b);
		CHECK(a == b, "NaN input should match v2 (floor)");
		std::vector<std::complex<float>> huge(1024, { 3e38f, 3e38f });
		g2.Compute(huge, a);
		g4.Compute(huge, b);
		CHECK(a == b, "overflowing input should match v2 (FLT_MAX), %g vs %g", a[128], b[128]);
	}
	{
		CFftV4 f(1024, 0.5f, false, CFftV4::PlanEffort::Measure);
		auto in = Tone(1024 * 8, 100.0 / 1024, 1.0f);
		std::vector<float> out;
		f.Compute(in, out);
		const auto peak = std::max_element(out.begin(), out.end()) - out.begin();
		CHECK(peak == 512 + 100 && std::fabs(out[peak]) < 0.01f, "peak %ld at %g dB", static_cast<long>(peak), out[peak]);
	}
}
bool FileExists(const std::string& path)
{
	std::ifstream f(path);
	return f.good();
}

/// Whether FFTW can build v4's forward plan for N from wisdom alone
bool V4PlanInWisdom(uint32_t N)
{
	dsp::utility::CFftwPlanLock lock;
	fftwf_complex* in = fftwf_alloc_complex(N);
	fftwf_complex* out = fftwf_alloc_complex(N);
	fftwf_plan p = fftwf_plan_dft_1d(static_cast<int>(N), in, out, FFTW_FORWARD,
		FFTW_MEASURE | FFTW_DESTROY_INPUT | FFTW_WISDOM_ONLY);
	const bool ok = p != nullptr;
	if (p)
	{
		fftwf_destroy_plan(p);
	}
	fftwf_free(in);
	fftwf_free(out);
	return ok;
}

double MeasureCtorMs(uint32_t N)
{
	const auto start = std::chrono::steady_clock::now();
	CFftV4 f(N, 0.5f, false, CFftV4::PlanEffort::Measure);
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void TestWisdom()
{
	std::printf("wisdom\n");
	const std::string base = "fft_tests_wisdom_" +
		std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
	const std::string path = base + ".wisdom";
	const std::string junk = base + ".junk";

	CHECK(!FftwWisdom::Load(path), "missing file must fail");
	{
		std::ofstream(junk) << "not wisdom";
	}
	CHECK(!FftwWisdom::Load(junk), "garbage file must fail");
	CHECK(!FftwWisdom::Import("garbage"), "garbage text must fail");
	CHECK(!FftwWisdom::Save("no_such_dir_xyz/sub/fftwf.wisdom"), "unwritable path must fail");

	// Gather, save, forget, reload
	FftwWisdom::Forget();
	CHECK(!V4PlanInWisdom(512), "no wisdom after Forget");
	const double coldMs = MeasureCtorMs(512);
	CHECK(V4PlanInWisdom(512), "Measure construction should record wisdom");
	CHECK(FftwWisdom::Save(path), "save");
	CHECK(FileExists(path), "file written");
	FftwWisdom::Forget();
	CHECK(!V4PlanInWisdom(512), "forgotten");
	CHECK(FftwWisdom::Load(path), "load");
	CHECK(V4PlanInWisdom(512), "plan available from loaded wisdom");
	const double warmMs = MeasureCtorMs(512);
	CHECK(warmMs < coldMs / 4.0 || warmMs < 1.0, "warm %.2f ms should be far below cold %.2f ms", warmMs, coldMs);

	// String round trip
	const std::string text = FftwWisdom::Export();
	CHECK(!text.empty(), "export");
	FftwWisdom::Forget();
	CHECK(FftwWisdom::Import(text) && V4PlanInWisdom(512), "import round trip");

	// Results with wisdom match results from a cold Estimate plan
	{
		auto in = Noise(512 * 5, 3);
		CFftV4 warm(512, 0.5f, false, CFftV4::PlanEffort::Measure);
		FftwWisdom::Forget();
		CFftV4 cold(512, 0.5f);
		std::vector<float> a, b;
		warm.Compute(in, a);
		cold.Compute(in, b);
		CHECK(MaxDiff(a, b) < 1e-3f, "diff %g dB", MaxDiff(a, b));
	}

	// CFftwWisdomFile: saves only when wisdom changed
	std::remove(path.c_str());
	FftwWisdom::Forget();
	{
		CFftwWisdomFile scope(path);
		CHECK(!scope.Loaded(), "nothing to load yet");
		CHECK(scope.SaveIfChanged() && !FileExists(path), "no new wisdom, so no file");
		// FFTW records Estimate plans too, but they can't stand in for Measure ones
		CFftV4 estimateOnly(1024, 0.5f);
	}
	FftwWisdom::Forget();
	if (FileExists(path))
	{
		CHECK(FftwWisdom::Load(path), "estimate wisdom loads");
	}
	CHECK(!V4PlanInWisdom(1024), "estimate wisdom must not satisfy a Measure plan");
	{
		CFftwWisdomFile scope(path);
		CFftV4 measured(1024, 0.5f, false, CFftV4::PlanEffort::Measure);
	}
	CHECK(FileExists(path), "new wisdom saved on destruction");
	FftwWisdom::Forget();
	{
		CFftwWisdomFile scope(path);
		CHECK(scope.Loaded(), "loads the saved file");
		CHECK(V4PlanInWisdom(1024), "and its plans");
		CHECK(scope.SaveIfChanged(), "nothing changed is success");
	}

	std::remove(path.c_str());
	std::remove(junk.c_str());
	FftwWisdom::Forget();
}
} // namespace

int main()
{
	TestTypeProperties();
	TestConstructorValidation();
	TestComputeValidation();
	TestFloor();
	TestTone();
	TestEquivalenceWithV1();
	TestRepeatedCallsIndependent();
	TestV3();
	TestV4();
	TestWisdom();
	std::printf("\n%d/%d checks passed\n", gChecks - gFailures, gChecks);
	return gFailures == 0 ? 0 : 1;
}
