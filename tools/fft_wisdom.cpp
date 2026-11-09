// Generates FFTW wisdom for the CFft classes, so applications can load it at
// startup and plan Measure/Patient FFTs instantly. Run it once per machine (and
// again after an FFTW upgrade), for the sizes and classes the application uses.
//
//   fft_wisdom -o fftwf.wisdom --sizes 1024,4096,16384
//   fft_wisdom -o fftwf.wisdom --sizes 4096 --class v2,v4 --effort patient --reverse
//
// An existing output file is loaded first and extended, so running it again for
// extra sizes keeps what's already there.

#include "stdafx.h"
#include "FftV2.h"
#include "FftV3.h"
#include "FftV4.h"
#include "FftwWisdom.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace
{
std::vector<std::string> Split(const std::string& s)
{
	std::vector<std::string> out;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, ','))
	{
		if (!item.empty())
		{
			out.push_back(item);
		}
	}
	return out;
}

void Usage()
{
	std::printf(
		"usage: fft_wisdom -o <file> [--sizes a,b,c] [--class v2,v3,v4] [--effort measure|patient] [--reverse]\n"
		"  --sizes    FFT sizes (rounded up to powers of 2). Default 256,1024,4096,16384,65536\n"
		"  --class    classes to plan for. Wisdom is per plan shape, so match the application. Default v4\n"
		"  --effort   measure (default) or patient. Patient plans need patient wisdom\n"
		"  --reverse  also plan the reverse (inverse, unwindowed) direction\n");
}

template <typename T>
double PlanMs(uint32_t size, bool reverse, typename T::PlanEffort effort, const char** note)
{
	const auto start = std::chrono::steady_clock::now();
	T fft(size, 0.5f, reverse, effort);
	*note = "";
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// v4 can tell whether wisdom already covered the plan
template <>
double PlanMs<CFftV4>(uint32_t size, bool reverse, CFftV4::PlanEffort effort, const char** note)
{
	const auto start = std::chrono::steady_clock::now();
	CFftV4 fft(size, 0.5f, reverse, effort);
	*note = fft.PlannedFromWisdom() ? "  (already in wisdom)" : "";
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

int main(int argc, char** argv)
{
	std::string out;
	std::vector<std::string> sizes = { "256", "1024", "4096", "16384", "65536" };
	std::vector<std::string> classes = { "v4" };
	std::string effort = "measure";
	bool reverse = false;

	for (int i = 1; i < argc; i++)
	{
		const std::string a = argv[i];
		const bool hasValue = i + 1 < argc;
		if ((a == "-o" || a == "--out") && hasValue) out = argv[++i];
		else if (a == "--sizes" && hasValue) sizes = Split(argv[++i]);
		else if (a == "--class" && hasValue) classes = Split(argv[++i]);
		else if (a == "--effort" && hasValue) effort = argv[++i];
		else if (a == "--reverse") reverse = true;
		else
		{
			Usage();
			return 2;
		}
	}
	if (out.empty() || (effort != "measure" && effort != "patient"))
	{
		Usage();
		return 2;
	}
	const bool patient = (effort == "patient");

	const bool existed = FftwWisdom::Load(out);
	std::printf("%s %s\n", existed ? "extending" : "creating", out.c_str());

	try
	{
		for (const std::string& sizeText : sizes)
		{
			const uint32_t size = static_cast<uint32_t>(std::strtoul(sizeText.c_str(), nullptr, 10));
			for (int dir = 0; dir < (reverse ? 2 : 1); dir++)
			{
				for (const std::string& cls : classes)
				{
					double ms;
					const char* note = "";
					if (cls == "v2")
						ms = PlanMs<CFftV2>(size, dir == 1, patient ? CFftV2::PlanEffort::Patient : CFftV2::PlanEffort::Measure, &note);
					else if (cls == "v3")
						ms = PlanMs<CFftV3>(size, dir == 1, patient ? CFftV3::PlanEffort::Patient : CFftV3::PlanEffort::Measure, &note);
					else if (cls == "v4")
						ms = PlanMs<CFftV4>(size, dir == 1, patient ? CFftV4::PlanEffort::Patient : CFftV4::PlanEffort::Measure, &note);
					else
					{
						std::printf("unknown class %s\n", cls.c_str());
						return 2;
					}
					std::printf("  %-3s N=%-8u %-8s %-7s %9.1f ms%s\n", cls.c_str(), size, dir ? "reverse" : "forward",
						effort.c_str(), ms, note);
				}
			}
		}
	}
	catch (const std::exception& e)
	{
		std::printf("error: %s\n", e.what());
		return 1;
	}

	if (!FftwWisdom::Save(out))
	{
		std::printf("error: couldn't write %s\n", out.c_str());
		return 1;
	}
	std::printf("saved %s\n", out.c_str());
	return 0;
}
