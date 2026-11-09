# fftuse

| File | Class | What it is |
|---|---|---|
| `Fft.h` / `Fft.cpp` | `CFft` | The original, untouched |
| `FftV2.h` / `FftV2.cpp` | `CFftV2` | Bug fixes and the first speedups |
| `FftV3.h` / `FftV3.cpp` | `CFftV3` | Experiment: batched segments with `fftwf_plan_many_dft`. Not faster; kept for the record |
| `FftV4.h` / `FftV4.cpp` | `CFftV4` | Fastest: out-of-place FFT and vectorized dB conversion. Use this one |

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

`bench/results.csv` holds the first run, taken in a 4-core cloud VM, so treat the
absolute numbers as rough.

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

`g++ -O3 -fopt-info-vec-missed -c FftV4.cpp` should report every kernel loop as
vectorized. If a loop shows "control flow in loop", one of these patterns has crept
back in. Clang vectorizes it either way. MSVC hasn't been checked.

### Next ideas

* Save FFTW wisdom to a file (`fftwf_export_wisdom_to_filename`), so the Measure
  constructor cost is paid once per machine, not once per process. Measure gives v4
  another 1.05–1.5x.
* The FFT is now 70–80% of the time. Beyond Measure planning, the remaining lever is
  doing fewer transforms, for example less overlap if the application allows it.
