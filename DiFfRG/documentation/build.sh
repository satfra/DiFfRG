#!/usr/bin/env bash
#
# Build the unified DiFfRG documentation site (C++ / Python / Mathematica).
#
# Steps, in order:
#   1. Generate the Mathematica reference from ::usage strings (no Wolfram).
#   2. Run Doxygen on the C++ headers -> HTML inside the Sphinx source tree.
#   3. Run doxysphinx to wrap the Doxygen HTML as Sphinx documents.
#   4. Run sphinx-build to produce the final site.
#
# Usage: build.sh [OUTPUT_DIR]   (default: <documentation>/_build)
#
# Requirements (see requirements.txt): doxygen, plus a Python 3.9–3.12
# environment with sphinx, pydata-sphinx-theme, myst-parser, sphinx-autoapi,
# sphinx-design and doxysphinx installed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # documentation/
DIFFRG_DIR="$(cd "${HERE}/.." && pwd)"                 # DiFfRG/ library root
OUT="${1:-${HERE}/_build}"
PYTHON="${PYTHON:-python3}"

DOXYGEN_BASE="${HERE}/doxygen/DiFfRG"        # OUTPUT_DIRECTORY for Doxygen
DOXYGEN_HTML="${DOXYGEN_BASE}/html"          # where index.html ends up
SPHINX_HTML="${OUT}/html"

# Start from a clean slate so stale pages are never carried over / deployed.
cd "${HERE}"
rm -rf "${OUT}" "${HERE}/doxygen" "${HERE}/_generated_mathematica" "${HERE}/autoapi"

echo "==> [1/4] Generating Mathematica reference"
"${PYTHON}" "${HERE}/gen_mathematica.py" \
    --src "${DIFFRG_DIR}/Mathematica/DiFfRG" \
    --out "${HERE}/_generated_mathematica"

echo "==> [2/4] Running Doxygen"
mkdir -p "${DOXYGEN_BASE}"
DOXYFILE="${HERE}/doxygen/Doxyfile.generated"
DOXYGEN_LOG="${HERE}/doxygen/doxygen.log"
sed -e "s|@CMAKE_CURRENT_SOURCE_DIR@|${DIFFRG_DIR}|g" \
    -e "s|@DOXYGEN_OUTPUT_DIR@|${DOXYGEN_BASE}|g" \
    "${HERE}/Doxyfile.in" > "${DOXYFILE}"
# Keep the terminal clean: Doxygen's progress (QUIET=YES) and warnings go to a
# log; only surface it if Doxygen actually fails. Set DOXYGEN_VERBOSE=1 to tee.
if [ -n "${DOXYGEN_VERBOSE:-}" ]; then
    doxygen "${DOXYFILE}" 2>&1 | tee "${DOXYGEN_LOG}"
else
    if ! doxygen "${DOXYFILE}" > "${DOXYGEN_LOG}" 2>&1; then
        echo "Doxygen failed — log follows:" >&2
        cat "${DOXYGEN_LOG}" >&2
        exit 1
    fi
fi
# Doxygen >= 1.13 emits CSS relative-color functions that doxysphinx's libsass
# cannot parse; neutralize them (no-op on older Doxygen).
"${PYTHON}" "${HERE}/fix_doxygen_css.py" "${DOXYGEN_HTML}"

echo "==> [3/4] Running doxysphinx"
doxysphinx build "${HERE}" "${SPHINX_HTML}" "${DOXYGEN_HTML}"

echo "==> [4/4] Running sphinx-build"
sphinx-build -b html "${HERE}" "${SPHINX_HTML}"

echo "==> Done. Open ${SPHINX_HTML}/index.html"
