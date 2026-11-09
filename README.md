# fftuse

| File | Class | What it is |
|---|---|---|
| `Fft.h` / `Fft.cpp` | `CFft` | The original, untouched |
| `FftV2.h` / `FftV2.cpp` | `CFftV2` | Bug fixes and the first speedups |
| `FftV3.h` / `FftV3.cpp` | `CFftV3` | Experiment: batched segments with `fftwf_plan_many_dft`. Not faster; kept for the record |
| `FftV4.h` / `FftV4.cpp` | `CFftV4` | Fastest: out-of-place FFT and vectorized dB conversion. Use this one |
| `FftV4Kernels.h` / `FftV4Kernels.cpp` | (internal) | CFftV4's per-sample loops, in their own file so they can get NEON / unsafe-math flags on 32-bit ARM |
| `FftwWisdom.h` / `FftwWisdom.cpp` | `FftwWisdom::*`, `CFftwWisdomFile` | Save and load FFTW wisdom so measured plans are instant (works with v2–v4) |
| `tools/fft_wisdom.cpp` | | Command-line tool that generates a wisdom file |

All four build side by side so they can be tested and benchmarked against each
other. v2, v3 and v4 share the same interface and behavior.

## Build

Needs single-precision FFTW (`libfftw3-dev`, which provides `fftw3f`) and the Boost
headers.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fft_tests       # correctness, including v2 == v1 on valid inputs
./build/v1_bug_repro    # runs each reported bug against v1 and v2 (ASan/UBSan, Linux/macOS)
./build/fft_bench       # performance
./build/fft_wisdom -o fftwf.wisdom   # generate FFTW wisdom for this machine
bench/collect.sh --name <machine>   # everything above, packed into one archive to share
```

`compat/` has stand-ins for `stdafx.h` and `dsp::utility::CFftwPlanLock`, a mutex
around FFTW plan creation and destruction. FFTW 3.3.10 still requires that lock,
because only `fftwf_execute` is thread-safe. If the real headers are available, put
them on the include path in place of `compat/`.

## Benchmark

```
./build/fft_bench --quick                                    # fast check
./build/fft_bench --csv bench/results.csv --label "what changed"
./build/fft_bench --sizes 2048,8192 --segments 64 --overlap 0.75 --only v2,v4
./build/fft_bench --only "v4*"                               # v4 and v4-measure
```

Compare versions within one run (`--only a,b`), because run-to-run noise on a shared
machine is 5–10%.

For each FFT size and output-bin count, it reports the constructor time (FFTW
wisdom is cleared first, so every construction is cold), the median time per
`Compute`, throughput, speedup over v1, and the largest dB difference from v1's
output. The difference column catches a speedup that changes the results.
`--csv` appends rows, so the file keeps a history across iterations. To benchmark a
new version, add one line to `Implementations()` in `bench/fft_bench.cpp`.

`bench/results.csv` holds the runs made during development, all on one 4-core x86
cloud VM (`machine` = `cloud-vm-x86`), so treat the absolute numbers as rough.

## Running the benchmarks on another machine (ARM, etc.)

On the target machine, from a checkout:

```
bench/collect.sh --name <machine>           # full run, about 5–15 minutes
bench/collect.sh --name <machine> --quick   # about 1–3 minutes
bench/collect.sh --name <machine> --sizes 1024,4096   # only the sizes you care about
```

It produces `bench-results-<machine>-<timestamp>.tar.gz`. Send that file back. The
script:

1. Records system information: CPU and cores (`lscpu`), board model, compiler,
   commit, governor and frequency range per core, temperatures before and after,
   memory, and load.
2. Builds in Release mode. Pass extra CMake options with `CMAKE_ARGS`, for example
   `CMAKE_ARGS=-DCMAKE_PREFIX_PATH=/opt/fftw`.
3. Runs the correctness tests (`tests.log`). They matter as much as the timings on
   a new architecture, so the script still benchmarks when they fail, but says so.
4. Runs the v1 bug repro if the toolchain supports sanitizers. This step is
   informational. Each scenario is killed after 30 seconds (`FFT_REPRO_TIMEOUT`)
   and reported as `HANG`. On ARM, v1 with `overlap = 1` hangs rather than
   returning garbage: its segment count is +inf converted to `uint32_t`, which
   saturates to 4294967295 on ARM but becomes 0 on x86. v1 then runs about 4.3
   billion FFTs.
5. Generates FFTW wisdom for this machine.
6. Benchmarks every version twice, without and with the wisdom, into one
   `results.csv`.

The benchmark is pinned with `taskset` to the core with the highest maximum
frequency, which picks a big core on big.LITTLE chips. Set `BENCH_CPU=<n>` to choose
a different core.

Every CSV row records the machine, architecture, CPU, thread count, governor,
compiler, build flags, the SIMD the compiler targeted, and FFTW's version string
(which lists FFTW's SIMD support, e.g. `fftw-3.3.10-neon`). CSVs from different
machines can be concatenated, minus their header lines, and compared directly.
`./build/fft_bench --info` prints the same details without running anything.

For steadier numbers on boards that throttle:

* Set the governor to `performance` while benchmarking: `sudo cpupower frequency-set -g performance`,
  or write to `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`. The output
  warns when it isn't.
* Use a heatsink or fan. The before and after temperatures in `system.txt` show
  whether it got hot.
* Close other workloads.

### Checked on AArch64

The code was cross-compiled with GCC 13 for AArch64 and run under QEMU against
Ubuntu's arm64 FFTW 3.3.10, which is built with NEON. Emulated timings mean nothing,
so this checks correctness only:

* All 758 test checks pass, including the fast-log and wisdom tests.
* v4's window, accumulate and dB kernels vectorize to 128-bit NEON, the same as SSE
  on x86.
* `fft_bench` and `fft_wisdom` run and record `aarch64`, `neon+fma` and
  `fftw-3.3.10-neon`.

Not checked: real ARM timings, and detecting the CPU model on real ARM hardware.

### 32-bit ARM (ARMv7, e.g. Cortex-A8)

Two things matter much more on an ARMv7 core like the Cortex-A8 than elsewhere. Its
scalar floating-point unit ("VFPLite") isn't pipelined, so plain float code is
slow, and NEON is where the throughput is.

**1. FFTW built with NEON (likely the biggest win).** The FFT is 70–80% of the
time. Distribution armhf packages target a baseline without NEON. Ubuntu 24.04's
`libfftw3-dev:armhf`, for example, reports plain `fftw-3.3.10`, with no `-neon`,
so its FFT runs entirely on the scalar FPU. The `fftw` field in `--info` and in
the CSV shows what your FFTW was built with, and `fft_bench` prints a note when
it lacks NEON. To build one:

```
./configure --enable-float --enable-neon --disable-fortran \
    CFLAGS="-O3 -mcpu=cortex-a8 -mfpu=neon -mfloat-abi=hard"
make && make install          # then point CMake at it with -DCMAKE_PREFIX_PATH=<prefix>
```

Add `--host=arm-linux-gnueabihf` to cross-compile. Wisdom files are per FFTW
build, so regenerate them after switching.

**2. CFftV4's kernels with NEON:** `-DFFT_ARMV7_NEON=ON`. On ARMv7, GCC vectorizes
float loops for NEON only with `-funsafe-math-optimizations`, because ARMv7 NEON
flushes denormals to zero. The option applies `-mfpu=neon
-funsafe-math-optimizations` to `FftV4Kernels.cpp` only. That file holds the four
per-sample loops: window, accumulate, and the two passes of dB conversion.
Everything else, including the window setup and FFTW planning in the constructor,
keeps strict IEEE semantics. In your own build system, give `FftV4Kernels.cpp`
those two flags. Use `-funsafe-math-optimizations`, not `-ffast-math`: the latter
implies finite-only math, which lets the compiler drop the NaN and infinity
handling in the dB conversion. The binary then needs a CPU with NEON. The A8 has
it; a few A9s, such as Tegra 2, don't.

Scoping the flag to the kernels matters. The first attempt applied it to all of
`FftV4.cpp`, and the N=2 window broke. A symmetric Blackman window is exactly zero
at both points for N=2, so v1 and v2 normalize round-off (about -1.4e-17 per
sample). Unsafe math is allowed to turn that round-off into an exact 0, giving
0/0. The kernels now live in their own file, and v4 handles N=2 explicitly with
the rectangular window v1 and v2 end up with.

`bench/collect.sh` detects ARMv7 with NEON and benchmarks both builds, toolchain
defaults and `FFT_ARMV7_NEON=ON`, in one run. The build column of `results.csv`
tells them apart.

**Checked, under emulation only** (QEMU as a Cortex-A8, cross-compiled with GCC 13
for armhf):

* With the toolchain defaults (`-march=armv7-a+fp`), none of v4's kernels
  vectorize. With `-mfpu=neon` alone, GCC still refuses. With `-mfpu=neon
  -funsafe-math-optimizations`, all four vectorize to 128-bit NEON.
* All 758 test checks pass with `FFT_ARMV7_NEON=ON` against the distribution FFTW
  and against an FFTW built with `--enable-neon`, and also with the option off.
  That includes NaN, infinity and 60 decades of power through the fast log.
* `fft_bench --info` reports the kernel flags in the build field and
  `fftw-3.3.10-neon` vs `fftw-3.3.10`.

QEMU doesn't model the A8's unpipelined FPU, so emulated timings say nothing about
the speedup. That needs your board.

Running `collect.sh` on an A8 is slow. Start with `--quick` and `--sizes` set to
the FFT sizes your application uses, for example
`bench/collect.sh --name a8 --quick --sizes 1024,4096`.

## Behavior changes in v2

* Invalid arguments throw `std::invalid_argument`:
  * `fftSize` outside `[2, 2^30]`
  * `overlap` outside `[0, 1)`, or NaN
  * `length < GetFftSize()`
  * `bins > GetFftSize()`
  * a null pointer
* A failed FFTW allocation throws `std::bad_alloc`, and a failed plan throws
  `std::runtime_error`.
* Empty bins come out as `GetFloorDb()`, which defaults to -300 dB. v1 returned
  -FLT_MAX.
* The vector overload resizes `outData` itself.
* The class can't be copied.
* An optional `PlanEffort` constructor argument selects `FFTW_MEASURE` or
  `FFTW_PATIENT` planning.
* Results are otherwise unchanged: on valid inputs, v2 matches v1 to within 1e-4 dB.

## v3: batched segments (experiment)

`CFftV3` is `CFftV2` with one change: it windows `B` segments into a buffer of
`B * N` samples and transforms them with one `fftwf_plan_many_dft` call. Segments
left over after the last full batch go through a single-segment plan on the same
buffer. The batch size is a constructor argument. With 0 (the default), it is picked
so the buffer is about 256 KB: for example, 32 segments at N = 1024, and 1 segment
for N of 32768 and up. The output matches v2.

The batching doesn't pay off. With FFTW_ESTIMATE and 32 segments per call, compared
with v2:

| N | v3 (auto batch) | best fixed batch | v2-measure |
|---|---|---|---|
| 256 | 0.97–1.06x | 1.1–1.2x (B=4) | 1.23x |
| 1024 | 0.98–1.08x | about 1.0x | 1.25x |
| 4096 | 0.94–1.04x | 1.05x (B=4) | 1.2–1.3x |
| 16384 | about 1.0x | B=16 is 0.82x | 1.2–1.45x |
| 65536 | 1.0x | B=4 and B=16 are 0.9x | 1.2–1.3x |

Run-to-run noise is about 5–10%, so most of these differences are within the noise.
Why it doesn't help:

* The per-call overhead of `fftwf_execute` is tiny next to the N log N work, so
  there is almost nothing to save.
* Batching hurts locality. In v2, a segment stays in L1/L2 cache from windowing,
  through the FFT, to accumulation. In v3, the batch buffer is bigger than the cache
  by the time it is accumulated. That is why large batches at large N are slower.
* FFTW already vectorizes within one contiguous transform. Batching mainly helps
  with many tiny transforms (N of 64 or less), which isn't this use case.

`FFTW_MEASURE` gives more than batching does, on v2 or v3.

### Where the time goes (v2, per segment, -O3)

An earlier version of this table was measured at -O2, where GCC 13 doesn't vectorize
the window and accumulate loops. That overstated them (35–50%). At the -O3 Release
build:

| N | window | FFT | accumulate | dB conversion |
|---|---|---|---|---|
| 256–4096 | 12–15% | 62–67% | 8–13% | 12–14% |
| 16384–65536 | 8–18% | 74–80% | 4–5% | 5–7% |

The FFT dominates, and building with AVX2 (`-mavx2 -mfma`) doesn't change that.

## v4: out-of-place FFT and vectorized dB conversion

`CFftV4` is `CFftV2` with two changes, and its output still matches v2:

* **Out-of-place FFT.** FFTW reads from one buffer and writes to a second, with
  `FFTW_DESTROY_INPUT`. That flag is free because the input buffer is rewritten for
  every segment. FFTW picks faster algorithms out-of-place, and `FFTW_MEASURE`
  planning takes about half as long.
* **Vectorized fast dB conversion.** The fftshift reads two contiguous runs, not a
  masked index per bin, and `log10` is replaced by a polynomial `log2` that is exact
  to about 1e-5 dB. This kernel is 4.3x faster than the `std::log10` loop. Power
  below FLT_MIN (-379 dB, before the shift) reads as -379 dB. Zero, NaN and
  infinite power behave as in v2.

With 32 segments per call (recorded in `bench/results.csv` with the label "v4 ..."):

| N | v4 vs v2 | v4-measure vs v2-measure | v4-measure vs v1 | Measure constructor |
|---|---|---|---|---|
| 256 | 1.4–1.7x | 1.15–1.45x | 2.2x | 40 ms (v2: 77 ms) |
| 1024 | 1.25–1.5x | 1.1–1.35x | 1.9x | 75 ms (v2: 150 ms) |
| 4096 | 1.15–1.3x | 1.1–1.25x | 1.8–1.9x | 122 ms (v2: 250 ms) |
| 16384 | 1.05–1.15x | 1.1–1.25x | 1.9x | 255 ms (v2: 490 ms) |
| 65536 | 1.05–1.2x | 1.05–1.15x | 1.4–1.5x | 860 ms (v2: 1580 ms) |

The ranges cover the recorded run and two more same-run comparisons.

### What didn't work (measured, then dropped)

* **Interleaved float loops.** These treat the buffers as `re, im, re, im` floats,
  with the window stored twice per sample. The window loop was no faster, and the
  accumulate loop (`re*re + im*im` at stride 2) was about 4x slower than
  `std::norm` over `std::complex<float>`, because GCC vectorizes the complex form
  better. Including it made v4 10–30% slower than v2, so v4 keeps v2's complex
  loops. The comment on `AccumulatePower` records this.
* **AVX2 build flags.** No measurable change. FFTW picks its own SIMD at run time.

### Vectorization notes

GCC's default `-ftrapping-math` stops it from vectorizing a loop with a conditional
float op, and it creates those itself: it moves work into one arm of a `?:`, or
threads a branch around a constant case. `FastDb` avoids that in three ways:

* It uses only integer ops for the range reduction.
* It clamps with an integer `max` on the bits.
* `ConvertToDb` runs two passes: the log math first, then the selects.

`g++ -O3 -fopt-info-vec-missed -c FftV4Kernels.cpp` should report every kernel loop as
vectorized. If a loop shows "control flow in loop", one of these patterns has crept
back in. Clang vectorizes it either way. MSVC hasn't been checked.

## FFTW wisdom

Wisdom is FFTW's record of which algorithms ran fastest on this machine. With it
loaded, `PlanEffort::Measure` costs nothing at construction time.

### Using it

Load the wisdom once at startup, before creating any FFT objects:

```cpp
#include "FftwWisdom.h"

int main()
{
    // Loads the file if it exists. On destruction, saves it again if new plans were measured.
    CFftwWisdomFile wisdom(cacheDir + "/fftwf.wisdom");

    CFftV4 fft(4096, 0.5f, false, CFftV4::PlanEffort::Measure);   // instant once the file has this size
    ...
}
```

* **First run:** Measure plans pay their measurement cost, and the file is written
  at exit.
* **Later runs:** the file loads and Measure plans are built instantly.

You can also generate the file ahead of time, for example at install, so no run is
slow:

```
./build/fft_wisdom -o fftwf.wisdom --sizes 1024,4096,16384 --class v4
./build/fft_wisdom -o fftwf.wisdom --sizes 32768            # extends the same file
./build/fft_wisdom -o fftwf.wisdom --effort patient --reverse
```

There are also lower-level functions: `FftwWisdom::Load`, `Save`, `Import`,
`Export` and `Forget`. `Save` writes a temporary file and renames it into place, so
readers never see a partial file. Every function holds `CFftwPlanLock`, so they are
safe to call while other threads create FFT objects.

### Always-fast construction: `PlanEffort::WisdomOrEstimate`

`CFftV4` has a fourth planning mode that never measures. If loaded wisdom has a
measured (Measure or Patient) plan for this size and direction, it uses that plan.
Otherwise it falls back to an Estimate plan. Either way the constructor takes well
under a millisecond.

```cpp
CFftwWisdomFile wisdom(cacheDir + "/fftwf.wisdom");   // or FftwWisdom::Load(path)
CFftV4 fft(4096, 0.5f, false, CFftV4::PlanEffort::WisdomOrEstimate);
if (!fft.PlannedFromWisdom())
{
    log("no FFTW wisdom for N=4096; run fft_wisdom to speed this up");
}
```

Use it when a stall at startup is unacceptable, such as a UI or a real-time chain,
and wisdom is generated separately with `tools/fft_wisdom`. Use `Measure` instead
when it's fine for the first run to pay the measurement cost and fill in the file.

`PlannedFromWisdom()` is true only when measured wisdom covered the plan.
Estimate-level wisdom entries don't count, and `Estimate` mode always reports
false. It also works with `Measure` and `Patient`: there it tells you whether the
constructor measured or was served from wisdom. `fft_wisdom` uses it to flag sizes
the file already covers.

From the benchmark (`v4-wisdom` rows in `bench/results.csv`):

| | constructor | per call |
|---|---|---|
| No wisdom file | same as `v4` (Estimate): 0.05–1.5 ms | same as `v4` |
| With wisdom | same as `v4-measure`: 0.03–1.5 ms | same as `v4-measure` (FFTW builds the identical plan) |

### What to know about wisdom

* **Per machine and per FFTW version.** Keep the file in a per-machine location,
  such as a user cache directory, not in source control. FFTW rejects wisdom from
  another version (`Load` returns false). It accepts wisdom from another CPU, but
  the plans won't be tuned for this one.
* **Per plan shape.** Wisdom depends on size, direction, in-place vs out-of-place,
  and flags. v2/v3 (in-place) and v4 (out-of-place) need their own entries, so
  generate with `--class` set to the class the application uses. FFTW does share
  sub-transforms between the two, which is why some v4 sizes planned instantly
  right after v2 in the generator output.
* **Estimate gets the benefit too.** An object built with `PlanEffort::Estimate`
  uses Measure wisdom if it is loaded, so it gets the measured plan with no risk of
  a slow constructor. In the benchmark below, v4 and v4-measure run at the same
  speed once wisdom is loaded.
* **Patient needs Patient wisdom.** Measure wisdom doesn't satisfy a Patient plan.
* **Estimate entries are harmless.** FFTW also records Estimate plans as wisdom, so
  the file can be rewritten after a run with only Estimate plans. Those entries
  never stand in for a Measure plan.

### Results

`./build/fft_bench --wisdom fftwf.wisdom` loads the file before each construction.
Those rows are tagged `[wisdom]` in `bench/results.csv`. Compared with the cold
runs:

| N | v4-measure constructor, cold | with wisdom | v4 (Estimate) per call, cold | with wisdom |
|---|---|---|---|---|
| 256 | 40 ms | 0.03 ms | 10.4 us | 10.3 us |
| 1024 | 75 ms | 0.07 ms | 52 us | 45 us |
| 4096 | 122 ms | 0.11 ms | 258 us | 214 us |
| 16384 | 260 ms | 0.7 ms | 1940 us | 1400 us |
| 65536 | 877 ms | 1.4 ms | 8290 us | 6810 us |

With wisdom, the time left in the constructor is mostly computing the window, not
planning. Wisdom also speeds up v1, which uses in-place Estimate plans and picks up
v2's wisdom. That makes the speedup-vs-v1 column in `[wisdom]` rows lower than in
cold rows. Compare absolute times across rows instead.

### Next ideas

* The FFT is now 70–80% of the time and runs with measured plans, so the remaining
  levers are application-level: less overlap, fewer segments, or a smaller FFT where
  resolution allows.
