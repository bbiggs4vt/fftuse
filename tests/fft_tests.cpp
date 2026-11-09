// Correctness tests for CFftV2, including equivalence with the original CFft on
// valid inputs. Run: ./fft_tests  (exit code is the number of failures, capped).

#include "stdafx.h"
#include "Fft.h"
#include "FftV2.h"

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
	std::printf("\n%d/%d checks passed\n", gChecks - gFailures, gChecks);
	return gFailures == 0 ? 0 : 1;
}
