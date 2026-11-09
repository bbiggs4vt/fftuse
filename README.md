# fftuse

`Fft.h` / `Fft.cpp` are the original `CFft`, left untouched. `FftV2.h` / `FftV2.cpp`
hold `CFftV2`, which has the fixes. `FftV3.h` / `FftV3.cpp` hold `CFftV3`, an
experiment that batches segments with `fftwf_plan_many_dft` (results below). All
three build side by side so they can be tested and benchmarked against each other.

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
./build/fft_bench --sizes 2048,8192 --segments 64 --overlap 0.75 --only v2
```

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
| 256 | 0.97–1.06x | 1.10x (B=4) | 1.23x |
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

### Where the time goes (v2, per segment, measured separately)

| N | window | FFT | accumulate | dB conversion |
|---|---|---|---|---|
| 256–4096 | ~20–24% | 36–51% | ~20–29% | ~9–13% |
| 16384–65536 | ~15–21% | 52–66% | ~12–19% | ~5–8% |

The window and accumulate loops are 35–50% of the time at small and mid N, so they
are the next thing to speed up, not the FFT calls. Ideas for v4, each checkable with
`fft_bench`:

* Treat the buffers as interleaved float arrays, with the window duplicated per
  real/imag pair, so the compiler can use SIMD. Try `-march=native` or AVX2.
* Accumulate straight from the FFTW output without a separate pass over memory, for
  example with FFTW's split real/imag guru interface.
* Use a fast `log10` approximation for the dB conversion, if about 0.01 dB of error
  is acceptable.
* Use FFTW wisdom files so MEASURE plans are computed once rather than in every
  process.
