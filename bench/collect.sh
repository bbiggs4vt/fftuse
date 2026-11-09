#!/usr/bin/env bash
# Builds, verifies and benchmarks this project on the current machine, then packs
# everything into one archive to send back for comparison.
#
#   bench/collect.sh                 # full run (about 5-15 minutes, depending on the machine)
#   bench/collect.sh --quick         # smaller grid, about 1-3 minutes
#   bench/collect.sh --name pi5      # name used in the CSV and the archive (default: host name)
#   bench/collect.sh --sizes 1024,4096   # only these FFT sizes (use the sizes your application uses)
#   bench/collect.sh --no-neon-variant   # on 32-bit ARM, skip the second build (see below)
#
# On 32-bit ARM with NEON (e.g. Cortex-A8/A9), it builds and benchmarks twice:
# with the toolchain's default flags, and with -DFFT_ARMV7_NEON=ON (CFftV4's
# kernels built with -mfpu=neon -funsafe-math-optimizations). The two runs are
# told apart by the build column in results.csv.
#
# Environment:
#   BENCH_CPU=<n>     pin the benchmark to this CPU (default: the fastest core, if taskset exists)
#   CMAKE_ARGS="..."  extra arguments for cmake, e.g. -DCMAKE_PREFIX_PATH=/opt/fftw
#
# Needs: cmake, a C++14 compiler, single-precision FFTW (libfftw3-dev or a build
# with --enable-float), Boost headers. Run it on the target machine; it can't
# collect timings from a cross-compiled build.

set -euo pipefail

QUICK=0
NAME=""
SIZES=""
NEON_VARIANT=auto
while [ $# -gt 0 ]; do
	case "$1" in
		--quick) QUICK=1 ;;
		--name) NAME="$2"; shift ;;
		--sizes) SIZES="$2"; shift ;;
		--no-neon-variant) NEON_VARIANT=no ;;
		-h|--help) sed -n '2,24p' "$0"; exit 0 ;;
		*) echo "unknown option $1"; exit 2 ;;
	esac
	shift
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${NAME:-$(hostname -s 2>/dev/null || hostname)}"
# Keep the name safe for file names
SAFE_NAME="$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9._-' '_')"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/bench-results-$SAFE_NAME-$STAMP"
BUILD="$ROOT/build-collect"
mkdir -p "$OUT"

log() { printf '\n== %s\n' "$*" | tee -a "$OUT/collect.log"; }
JOBS="$( (nproc || sysctl -n hw.ncpu || echo 2) 2>/dev/null | head -1)"

log "System information"
{
	echo "name:      $NAME"
	echo "date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	echo "uname:     $(uname -a)"
	echo "commit:    $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo ' (modified)')"
	echo "compiler:  $(${CXX:-c++} --version 2>/dev/null | head -1 || echo unknown)"
	echo "cmake:     $(cmake --version | head -1)"
	if [ -r /proc/device-tree/model ]; then echo "board:     $(tr -d '\0' < /proc/device-tree/model)"; fi
	echo
	if command -v lscpu >/dev/null; then lscpu; elif command -v sysctl >/dev/null; then sysctl -a 2>/dev/null | grep -E '^(machdep.cpu|hw\.(ncpu|memsize|perflevel|cachelinesize|l[123]))' || true; fi
	echo
	echo "--- cpufreq per CPU (governor, min/max/current kHz)"
	for c in /sys/devices/system/cpu/cpu[0-9]*; do
		[ -d "$c/cpufreq" ] || continue
		printf '%s %s %s %s %s\n' "$(basename "$c")" "$(cat "$c/cpufreq/scaling_governor" 2>/dev/null)" \
			"$(cat "$c/cpufreq/cpuinfo_min_freq" 2>/dev/null)" "$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null)" \
			"$(cat "$c/cpufreq/scaling_cur_freq" 2>/dev/null)"
	done
	echo
	echo "--- thermal zones (millidegrees C)"
	for z in /sys/class/thermal/thermal_zone*; do
		[ -r "$z/temp" ] && echo "$(cat "$z/type" 2>/dev/null) $(cat "$z/temp")"
	done
	echo
	echo "--- memory"
	(free -h 2>/dev/null || vm_stat 2>/dev/null || true)
	echo
	echo "--- load"
	uptime
} > "$OUT/system.txt" 2>&1
cat "$OUT/system.txt" | head -8

# Build variants: name and extra CMake arguments
VARIANTS=("default")
VARIANT_ARGS=("")
case "$(uname -m)" in
	armv6*|armv7*|armv8l|arm)
		if [ "$NEON_VARIANT" = auto ] && grep -qiw neon /proc/cpuinfo 2>/dev/null; then
			VARIANTS+=("neon-kernels")
			VARIANT_ARGS+=("-DFFT_ARMV7_NEON=ON")
		fi
		;;
esac

build_variant() { # name extra-args
	local dir="$BUILD-$1"
	log "Build ($1, Release)"
	# shellcheck disable=SC2086
	cmake -S "$ROOT" -B "$dir" -DCMAKE_BUILD_TYPE=Release $2 ${CMAKE_ARGS:-} > "$OUT/build-$1.log" 2>&1 &&
	cmake --build "$dir" -j "$JOBS" >> "$OUT/build-$1.log" 2>&1 || { tail -30 "$OUT/build-$1.log"; echo "build $1 failed; see $OUT/build-$1.log"; exit 1; }
}

TESTS_OK=1
test_variant() { # name
	local dir="$BUILD-$1"
	log "Correctness tests ($1)"
	if ! "$dir/fft_tests" > "$OUT/tests-$1.log" 2>&1; then
		TESTS_OK=0
		echo "TESTS FAILED for $1 (benchmarks still run; failures are in tests-$1.log):"
		grep FAIL "$OUT/tests-$1.log" | head -20
	fi
	tail -1 "$OUT/tests-$1.log"
}

for i in "${!VARIANTS[@]}"; do
	build_variant "${VARIANTS[$i]}" "${VARIANT_ARGS[$i]}"
	test_variant "${VARIANTS[$i]}"
done
FIRST="$BUILD-${VARIANTS[0]}"
"$FIRST/fft_bench" --info --machine "$NAME" | tee "$OUT/machine.txt"
if [ -x "$FIRST/v1_bug_repro" ]; then
	"$FIRST/v1_bug_repro" > "$OUT/v1_bug_repro.log" 2>&1 || true
fi

# Pin to one CPU so the scheduler doesn't move the benchmark between cores. On
# big.LITTLE parts, pick the fastest core by its maximum frequency.
PIN=()
if command -v taskset >/dev/null; then
	CPU="${BENCH_CPU:-}"
	if [ -z "$CPU" ]; then
		best=-1
		for c in /sys/devices/system/cpu/cpu[0-9]*; do
			f="$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null || echo 0)"
			n="${c##*cpu}"
			if [ "$f" -ge "$best" ]; then best="$f"; CPU="$n"; fi
		done
	fi
	if [ -n "$CPU" ]; then
		PIN=(taskset -c "$CPU")
		echo "pinning benchmarks to CPU $CPU" | tee -a "$OUT/collect.log"
	fi
fi

if [ "$QUICK" = 1 ]; then
	GRID=(--bins-div 1 --min-time 30 --samples 5)
	WISDOM_SIZES="${SIZES:-256,1024,4096,16384}"
	IMPLS="v2*,v3,v4*"
else
	GRID=()
	WISDOM_SIZES="${SIZES:-256,1024,4096,16384,65536}"
	IMPLS="v2*,v3*,v4*"
fi
GRID+=(--sizes "$WISDOM_SIZES")

# Wisdom belongs to FFTW, which both variants share
log "Generate FFTW wisdom (v2 and v4 shapes)"
${PIN[@]+"${PIN[@]}"} "$FIRST/fft_wisdom" -o "$OUT/fftwf.wisdom" --class v2,v4 --sizes "$WISDOM_SIZES" | tee "$OUT/wisdom.log"

for v in "${VARIANTS[@]}"; do
	log "Benchmark $v, cold (no wisdom)"
	${PIN[@]+"${PIN[@]}"} "$BUILD-$v/fft_bench" --machine "$NAME" --only "$IMPLS" "${GRID[@]}" \
		--csv "$OUT/results.csv" --label "collect $v cold" | tee "$OUT/bench-$v-cold.txt"

	log "Benchmark $v, warm (wisdom loaded)"
	${PIN[@]+"${PIN[@]}"} "$BUILD-$v/fft_bench" --machine "$NAME" --only "$IMPLS" "${GRID[@]}" --wisdom "$OUT/fftwf.wisdom" \
		--csv "$OUT/results.csv" --label "collect $v" | tee "$OUT/bench-$v-warm.txt"
done

# Temperatures after the run show whether the board throttled
{
	echo "--- thermal zones after benchmarks (millidegrees C)"
	for z in /sys/class/thermal/thermal_zone*; do
		[ -r "$z/temp" ] && echo "$(cat "$z/type" 2>/dev/null) $(cat "$z/temp")"
	done
	echo "--- current frequencies after benchmarks (kHz)"
	for c in /sys/devices/system/cpu/cpu[0-9]*; do
		[ -r "$c/cpufreq/scaling_cur_freq" ] && echo "$(basename "$c") $(cat "$c/cpufreq/scaling_cur_freq")"
	done
} >> "$OUT/system.txt" 2>&1

ARCHIVE="$OUT.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$OUT")" "$(basename "$OUT")"
log "Done"
echo "tests: $([ "$TESTS_OK" = 1 ] && echo passed || echo FAILED)"
echo "send back: $ARCHIVE"
