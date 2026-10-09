#!/bin/bash
# Run commands in a Docker/OCI image through SingularityCE/Apptainer.
#
# Usage:
#   containers/singularity-run.sh [-g] [-W] [-b host:container] [-w dir] <image-ref-or-sif> <command...>
#
# Image refs may be local SIF files, docker:// references, or Docker-daemon tags.
# Docker-daemon tags are passed as docker-daemon://<tag> and converted to a
# temporary SIF before execution.
set -euo pipefail

scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

use_gpu=0
use_writable_tmpfs=0
workdir=""
binds=()

while getopts gWb:w: flag; do
  case "${flag}" in
  g) use_gpu=1 ;;
  W) use_writable_tmpfs=1 ;;
  b) binds+=("${OPTARG}") ;;
  w) workdir=${OPTARG} ;;
  *)
    echo "Usage: singularity-run.sh [-g] [-W] [-b host:container] [-w dir] <image-ref-or-sif> <command...>" >&2
    exit 2
    ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -lt 2 ]]; then
  echo "Usage: singularity-run.sh [-g] [-W] [-b host:container] [-w dir] <image-ref-or-sif> <command...>" >&2
  exit 2
fi

image=$1
shift

if command -v singularity >/dev/null 2>&1; then
  singularity_bin=singularity
elif command -v apptainer >/dev/null 2>&1; then
  singularity_bin=apptainer
else
  echo "Neither singularity nor apptainer was found on PATH." >&2
  exit 127
fi

cache_dir="${SINGULARITY_RUN_CACHE:-${scriptpath}/logs/sif-cache}"
mkdir -p "${cache_dir}"

sif=""
if [[ ${image} == *.sif && -f ${image} ]]; then
  sif=${image}
elif [[ ${image} == docker-daemon://* ]]; then
  tag=${image#docker-daemon://}
  safe_tag=${tag//[^A-Za-z0-9_.-]/_}
  sif="${cache_dir}/${safe_tag}.sif"
  "${singularity_bin}" build --force "${sif}" "${image}"
elif [[ ${image} == docker://* || ${image} == oras://* || ${image} == library://* ]]; then
  safe_ref=${image//[^A-Za-z0-9_.-]/_}
  sif="${cache_dir}/${safe_ref}.sif"
  "${singularity_bin}" pull --force "${sif}" "${image}"
else
  echo "Unsupported image reference: ${image}" >&2
  echo "Use a .sif path, docker://..., oras://..., library://..., or docker-daemon://<tag>." >&2
  exit 2
fi

exec_args=(exec --cleanenv)
[[ ${use_gpu} -eq 1 ]] && exec_args+=(--nv)
[[ ${use_writable_tmpfs} -eq 1 ]] && exec_args+=(--writable-tmpfs)
for bind in "${binds[@]}"; do
  exec_args+=(--bind "${bind}")
done
[[ -n ${workdir} ]] && exec_args+=(--pwd "${workdir}")

"${singularity_bin}" "${exec_args[@]}" "${sif}" "$@"
