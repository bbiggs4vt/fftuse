#pragma once

// Describes the machine and build a benchmark ran on, so results from different
// machines (x86, ARM, ...) can be compared and reproduced. Everything here is
// best effort: fields that can't be determined read "unknown".

#include <fftw3.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#ifndef FFT_BUILD_TYPE
#define FFT_BUILD_TYPE "unknown"
#endif
#ifndef FFT_CXX_FLAGS
#define FFT_CXX_FLAGS "unknown"
#endif

struct MachineInfo
{
	std::string machine;   // user label (--machine) or host name
	std::string arch;      // compile-time target architecture
	std::string cpu;       // CPU model
	unsigned cores = 0;    // hardware threads
	std::string governor;  // Linux cpufreq governor of the benchmark CPU
	std::string compiler;  // compiler and version
	std::string build;     // build type and flags
	std::string simd;      // SIMD the compiler targeted for our code
	std::string fftw;      // FFTW version string, which lists FFTW's SIMD support
};

namespace machine_info_detail
{
inline std::string Trim(std::string s)
{
	const char* ws = " \t\r\n";
	s.erase(0, s.find_first_not_of(ws));
	const size_t end = s.find_last_not_of(ws);
	s.erase(end == std::string::npos ? 0 : end + 1);
	return s;
}

inline std::string ReadFirstLine(const char* path)
{
	std::ifstream f(path);
	std::string line;
	if (f && std::getline(f, line))
	{
		// /proc/device-tree strings are NUL-terminated
		const size_t nul = line.find('\0');
		if (nul != std::string::npos)
		{
			line.erase(nul);
		}
		return Trim(line);
	}
	return std::string();
}

inline std::string RunCommand(const char* cmd)
{
#if defined(_WIN32)
	(void)cmd;
	return std::string();
#else
	FILE* p = popen(cmd, "r");
	if (!p)
	{
		return std::string();
	}
	std::string out;
	char buf[256];
	while (std::fgets(buf, sizeof(buf), p))
	{
		out += buf;
	}
	pclose(p);
	return Trim(out);
#endif
}

/// Value of the first "key : value" line in /proc/cpuinfo whose key matches
inline std::string CpuInfoField(const std::string& key)
{
	std::ifstream f("/proc/cpuinfo");
	std::string line;
	while (std::getline(f, line))
	{
		const size_t colon = line.find(':');
		if (colon != std::string::npos && Trim(line.substr(0, colon)) == key)
		{
			return Trim(line.substr(colon + 1));
		}
	}
	return std::string();
}

/// ARM cores report an implementer/part pair, not a model name
inline std::string ArmPartName(const std::string& implementer, const std::string& part)
{
	struct Entry { const char* impl; const char* part; const char* name; };
	static const Entry kParts[] = {
		{ "0x41", "0xd03", "Cortex-A53" }, { "0x41", "0xd04", "Cortex-A35" },
		{ "0x41", "0xd05", "Cortex-A55" }, { "0x41", "0xd07", "Cortex-A57" },
		{ "0x41", "0xd08", "Cortex-A72" }, { "0x41", "0xd09", "Cortex-A73" },
		{ "0x41", "0xd0a", "Cortex-A75" }, { "0x41", "0xd0b", "Cortex-A76" },
		{ "0x41", "0xd0c", "Neoverse-N1" }, { "0x41", "0xd0d", "Cortex-A77" },
		{ "0x41", "0xd40", "Neoverse-V1" }, { "0x41", "0xd41", "Cortex-A78" },
		{ "0x41", "0xd42", "Cortex-A78AE" }, { "0x41", "0xd44", "Cortex-X1" },
		{ "0x41", "0xd46", "Cortex-A510" }, { "0x41", "0xd47", "Cortex-A710" },
		{ "0x41", "0xd48", "Cortex-X2" }, { "0x41", "0xd49", "Neoverse-N2" },
		{ "0x41", "0xd4a", "Neoverse-E1" }, { "0x41", "0xd4b", "Cortex-A78C" },
		{ "0x41", "0xd4d", "Cortex-A715" }, { "0x41", "0xd4e", "Cortex-X3" },
		{ "0x41", "0xd4f", "Neoverse-V2" }, { "0x41", "0xd80", "Cortex-A520" },
		{ "0x41", "0xd81", "Cortex-A720" }, { "0x41", "0xd82", "Cortex-X4" },
		{ "0x41", "0xd84", "Neoverse-V3" }, { "0x41", "0xd8e", "Neoverse-N3" },
		{ "0x41", "0xc07", "Cortex-A7" }, { "0x41", "0xc09", "Cortex-A9" },
		{ "0x41", "0xc0f", "Cortex-A15" },
		{ "0x4e", "0x004", "NVIDIA Carmel" }, { "0x61", "0x022", "Apple M1" },
		{ "0xc0", "0xac3", "Ampere-1" }, { "0x51", "0x805", "Kryo 4xx Silver" },
	};
	for (const Entry& e : kParts)
	{
		if (implementer == e.impl && part == e.part)
		{
			return e.name;
		}
	}
	return "implementer " + implementer + " part " + part;
}

inline std::string DetectCpu()
{
	// x86 Linux, and some ARM kernels
	std::string model = CpuInfoField("model name");
	if (!model.empty() && model.find("ARMv") != 0)
	{
		return model;
	}
	// macOS (Intel and Apple Silicon)
	std::string mac = RunCommand("sysctl -n machdep.cpu.brand_string 2>/dev/null");
	if (!mac.empty())
	{
		return mac;
	}
	// ARM Linux: lscpu knows more part names than the table here
	std::string lscpu = RunCommand("lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | sort -u | paste -sd '+' -");
	std::string arm;
	const std::string impl = CpuInfoField("CPU implementer");
	if (!impl.empty())
	{
		// big.LITTLE parts differ per core; report the distinct ones
		std::ifstream f("/proc/cpuinfo");
		std::string line, parts;
		while (std::getline(f, line))
		{
			const size_t colon = line.find(':');
			if (colon != std::string::npos && Trim(line.substr(0, colon)) == "CPU part")
			{
				const std::string name = ArmPartName(impl, Trim(line.substr(colon + 1)));
				if (parts.find(name) == std::string::npos)
				{
					parts += (parts.empty() ? "" : "+") + name;
				}
			}
		}
		arm = parts;
	}
	std::string result = !lscpu.empty() && lscpu != "-" ? lscpu : arm;
	// Boards often name themselves (Raspberry Pi, Jetson, ...)
	const std::string board = ReadFirstLine("/proc/device-tree/model");
	if (!board.empty())
	{
		result += (result.empty() ? "" : " on ") + board;
	}
	if (result.empty())
	{
		result = model;
	}
	return result.empty() ? "unknown" : result;
}

inline std::string DetectArch()
{
#if defined(__aarch64__) || defined(_M_ARM64)
	return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
	return "arm32";
#elif defined(__x86_64__) || defined(_M_X64)
	return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
	return "x86";
#elif defined(__riscv)
	return "riscv";
#else
	return "unknown";
#endif
}

inline std::string DetectSimd()
{
	std::string s;
	auto add = [&s](const char* x) { s += (s.empty() ? "" : "+"); s += x; };
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	add("neon");
#endif
#if defined(__ARM_FEATURE_SVE)
	add("sve");
#endif
#if defined(__ARM_FEATURE_FMA)
	add("fma");
#endif
#if defined(__SSE2__) || defined(_M_X64)
	add("sse2");
#endif
#if defined(__AVX__)
	add("avx");
#endif
#if defined(__AVX2__)
	add("avx2");
#endif
#if defined(__FMA__)
	add("fma");
#endif
#if defined(__AVX512F__)
	add("avx512f");
#endif
	return s.empty() ? "none" : s;
}

inline std::string DetectCompiler()
{
#if defined(__clang__)
	return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
	return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
	return "msvc " + std::to_string(_MSC_VER);
#else
	return "unknown";
#endif
}

inline std::string DetectGovernor(int cpu)
{
	const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu < 0 ? 0 : cpu) + "/cpufreq/scaling_governor";
	const std::string g = ReadFirstLine(path.c_str());
	return g.empty() ? "unknown" : g;
}

inline int CurrentCpu()
{
#if defined(__linux__)
	const std::string stat = ReadFirstLine("/proc/self/stat");
	// Field 39 is the CPU the process last ran on; fields after the ")" are space separated
	const size_t paren = stat.rfind(')');
	if (paren == std::string::npos)
	{
		return -1;
	}
	size_t pos = paren + 2;
	for (int field = 3; field < 39 && pos != std::string::npos; field++)
	{
		pos = stat.find(' ', pos);
		if (pos != std::string::npos)
		{
			pos++;
		}
	}
	return pos == std::string::npos ? -1 : std::atoi(stat.c_str() + pos);
#else
	return -1;
#endif
}

inline std::string HostName()
{
#if !defined(_WIN32)
	char buf[256] = {};
	if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0])
	{
		return buf;
	}
#endif
	return "unknown";
}
} // namespace machine_info_detail

inline MachineInfo DetectMachine(const std::string& label)
{
	using namespace machine_info_detail;
	MachineInfo m;
	m.machine = label.empty() ? HostName() : label;
	m.arch = DetectArch();
	m.cpu = DetectCpu();
	m.cores = std::thread::hardware_concurrency();
	m.governor = DetectGovernor(CurrentCpu());
	m.compiler = DetectCompiler();
	m.build = std::string(FFT_BUILD_TYPE) + " " + Trim(FFT_CXX_FLAGS);
	m.simd = DetectSimd();
	m.fftw = Trim(fftwf_version);
	return m;
}

/// CSV-safe: quotes become apostrophes so a value can sit inside "..."
inline std::string CsvText(std::string s)
{
	for (char& c : s)
	{
		if (c == '"')
		{
			c = '\'';
		}
		else if (c == '\n' || c == '\r')
		{
			c = ' ';
		}
	}
	return s;
}
