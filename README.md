# fftuse

`Fft.h` / `Fft.cpp` are the original `CFft`, left untouched. `FftV2.h` / `FftV2.cpp`
hold `CFftV2`, which has the fixes. Both classes build side by side so they can be
tested and benchmarked against each other.

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
