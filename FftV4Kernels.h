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
// On x86-64 and AArch64 the default flags already vectorize these loops.
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
