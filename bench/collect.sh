#!/usr/bin/env bash
# Builds, verifies and benchmarks this project on the current machine, then packs
# everything into one archive to send back for comparison.
#
#   bench/collect.sh                 # full run (about 5-15 minutes, depending on the machine)
#   bench/collect.sh --quick         # smaller grid, about 1-3 minutes
#   bench/collect.sh --name pi5      # name used in the CSV and the archive (default: host name)
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
while [ $# -gt 0 ]; do
	case "$1" in
		--quick) QUICK=1 ;;
		--name) NAME="$2"; shift ;;
		-h|--help) sed -n '2,16p' "$0"; exit 0 ;;
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

log "Build (Release)"
# shellcheck disable=SC2086
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-} > "$OUT/build.log" 2>&1
cmake --build "$BUILD" -j "$JOBS" >> "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; echo "build failed; see $OUT/build.log"; exit 1; }
"$BUILD/fft_bench" --info --machine "$NAME" | tee "$OUT/machine.txt"

log "Correctness tests"
TESTS_OK=1
if ! "$BUILD/fft_tests" > "$OUT/tests.log" 2>&1; then
	TESTS_OK=0
	echo "TESTS FAILED (benchmarks still run; failures are in tests.log):"
	grep FAIL "$OUT/tests.log" | head -20
fi
tail -1 "$OUT/tests.log"
if [ -x "$BUILD/v1_bug_repro" ]; then
	"$BUILD/v1_bug_repro" > "$OUT/v1_bug_repro.log" 2>&1 || true
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
	GRID=(--sizes 256,1024,4096,16384 --bins-div 1 --min-time 30 --samples 5)
	WISDOM_SIZES=256,1024,4096,16384
	IMPLS="v2*,v3,v4*"
else
	GRID=()
	WISDOM_SIZES=256,1024,4096,16384,65536
	IMPLS="v2*,v3*,v4*"
fi

log "Generate FFTW wisdom (v2 and v4 shapes)"
${PIN[@]+"${PIN[@]}"} "$BUILD/fft_wisdom" -o "$OUT/fftwf.wisdom" --class v2,v4 --sizes "$WISDOM_SIZES" | tee "$OUT/wisdom.log"

log "Benchmark, cold (no wisdom)"
${PIN[@]+"${PIN[@]}"} "$BUILD/fft_bench" --machine "$NAME" --only "$IMPLS" ${GRID[@]+"${GRID[@]}"} \
	--csv "$OUT/results.csv" --label "collect cold" | tee "$OUT/bench-cold.txt"

log "Benchmark, warm (wisdom loaded)"
${PIN[@]+"${PIN[@]}"} "$BUILD/fft_bench" --machine "$NAME" --only "$IMPLS" ${GRID[@]+"${GRID[@]}"} --wisdom "$OUT/fftwf.wisdom" \
	--csv "$OUT/results.csv" --label "collect" | tee "$OUT/bench-warm.txt"

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
