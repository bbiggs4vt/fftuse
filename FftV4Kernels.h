#pragma once

// The per-sample loops of CFftV4, in their own translation unit so they can be
// built with different floating-point flags than the rest of the class.
//
// On 32-bit ARM (ARMv7, e.g. Cortex-A8/A9), GCC only vectorizes float loops for
// NEON when given -funsafe-math-optimizations, because ARMv7 NEON flushes
// denormals to zero and so isn't IEEE-compliant. Building just this file with
//     -mfpu=neon -funsafe-math-optimizations
// (CMake: -DFFT_ARMV7_NEON=ON) vectorizes all four loops, while the
// constructor's window and plan setup keep strict IEEE semantics. These kernels
// are written so the results are unchanged: they don't rely on denormals, and
// -funsafe-math-optimizations (unlike -ffast-math) keeps NaN and infinity
// handling. Don't use -ffast-math here: its -ffinite-math-only lets the compiler
// drop the NaN and infinity checks in ConvertToDb.
//
// Optimization level matters too. GCC's -O2 (the default in Yocto and many
// distro builds) uses its "very cheap" vectorizer cost model, which leaves all
// four loops scalar. Build this file with -O3, or add
//     -ftree-vectorize -fvect-cost-model=dynamic
// (CMake does this automatically).
//
// Link-time optimization (-flto) undoes all of the above: the kernels are
// inlined into callers built without these flags and run scalar (measured
// about 1.5x slower for CFftV4 on x86). Build this file with -fno-lto; CMake
// does that too.
//
// So a Yocto build for a Cortex-A9, whose tune already supplies -mfpu=neon,
// needs for this file:
//     -funsafe-math-optimizations -ftree-vectorize -fvect-cost-model=dynamic -fno-lto
//
// On x86-64 and AArch64, -O3 (or the two vectorizer flags) is all these loops need.
//
// Internal to CFftV4; not a public interface.

#include <complex>
#include <cstdint>

namespace fftv4_kernels
{
/// dst[i] = clamp(10*log10(src[i]) + offsetDb). Zero, negative and NaN power
/// give floorDb; +inf power gives FLT_MAX. Max error about 1e-5 dB.
void ConvertToDb(const float* src, float* dst, uint32_t count, float offsetDb, float floorDb);

/// dst[n] = window[n] * src[n]. The buffers must not overlap.
void ApplyWindow(const std::complex<float>* src, const float* window, std::complex<float>* dst, uint32_t count);

/// accum[n] += |spectrum[n]|^2. The buffers must not overlap.
void AccumulatePower(const std::complex<float>* spectrum, float* accum, uint32_t count);
} // namespace fftv4_kernels
