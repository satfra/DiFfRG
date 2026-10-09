#!/usr/bin/env bash
# ISA guard for a portable dependency bundle.
#
# Two-sided check on the compiled objects:
#  - no AVX-512 anywhere (zmm registers / EVEX-only instructions would SIGILL
#    on the x86-64-v3 baseline this bundle promises), which catches an
#    accidental -march=native leak from the build host;
#  - AVX (ymm registers) IS present in the heavyweight libraries, which proves
#    the -march flag actually reached their compilers -- a plumbing regression
#    that silently built plain x86-64 would otherwise ship unnoticed.
#
# Usage: check-isa.sh <bundle-dir> <march>
set -euo pipefail

BUNDLE="$1"
MARCH="$2"

fail() { echo "check-isa: FAIL: $*" >&2; exit 1; }

case "$MARCH" in
  x86-64-v3) ;;
  *) echo "check-isa: no ISA policy for -march=$MARCH, skipping"; exit 0 ;;
esac

# Every ELF object in lib/ and lib64/: shared libraries and static archives.
libdirs=("$BUNDLE/lib")
[ -d "$BUNDLE/lib64" ] && libdirs+=("$BUNDLE/lib64")
mapfile -t objects < <(find "${libdirs[@]}" \( -name '*.so*' -o -name '*.a' \) -type f)
[ "${#objects[@]}" -gt 0 ] || fail "no objects found under $BUNDLE/lib{,64}"

# grep must consume objdump's full output: an early-exit grep (-q/-m1) sends
# objdump SIGPIPE, which pipefail then reports as a pipeline failure -- making
# the positive probe fail exactly when it matches, and the negative one racy.
count_insn() { objdump -d --no-show-raw-insn "$1" 2>/dev/null | grep -c "$2" || true; }

for obj in "${objects[@]}"; do
  file -b "$obj" | grep -qE 'ELF|ar archive' || continue
  if [ "$(count_insn "$obj" '%zmm')" -gt 0 ]; then
    fail "$obj contains AVX-512 (zmm) instructions"
  fi
done
echo "check-isa: no AVX-512 in ${#objects[@]} objects"

# Positive check: these are large, hot, compiled-here libraries; with a working
# -march=x86-64-v3 they cannot plausibly contain zero ymm usage. At least one
# probe must exist -- an empty glob would pass vacuously, hiding exactly the
# regression this check exists for.
probed=0
for probe in "$BUNDLE"/lib*/libdeal_II.so.* "$BUNDLE"/lib*/libboost_math_tr1.so.*; do
  [ -f "$probe" ] || continue
  [ "$(count_insn "$probe" '%ymm')" -gt 0 ] \
    || fail "$probe contains no AVX (ymm) instructions -- did -march=$MARCH reach its build?"
  echo "check-isa: AVX present in $(basename "$probe")"
  probed=$((probed + 1))
done
[ "$probed" -gt 0 ] || fail "no probe libraries found -- bundle layout changed?"
