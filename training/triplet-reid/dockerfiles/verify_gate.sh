#!/usr/bin/env bash
# Builds a dedicated per-TF-version verification image (dockerfiles/<version>/Dockerfile)
# and runs the same 3-part gate as host_runbook/phase4_verification_gate.sh
# against it: (a) checkpoint var-name diff, (b) numeric equivalence
# atol=1e-4, (c) signature parity vs the wibex_model_v03 baseline.
#
# Reuses the existing Phase 4a/4b/4c probe scripts from
# tmp/inference/host_runbook/ (pure Python, not Dockerfiles -- not subject
# to the Sonar Dockerfile-naming constraint that drove this per-version
# directory layout). Network is confined to `docker build`; every actual
# execution step runs with --network=none.
#
# Usage: ./verify_gate.sh tf2180   (or tf2181, tf2210, ...)
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <version-dir>  (e.g. tf2180, tf2181, tf2210)\n' "$0" >&2
  exit 1
fi
readonly VERSION_DIR="$1"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
declare REPO_ROOT
REPO_ROOT="$(cd -- "${script_dir}/.." && pwd -P)"
readonly REPO_ROOT
declare INFERENCE_DIR
INFERENCE_DIR="$(cd -- "${REPO_ROOT}/../../tmp/inference" && pwd -P)"
readonly INFERENCE_DIR
readonly HOST_RUNBOOK_DIR="${INFERENCE_DIR}/host_runbook"
readonly DOCKERFILE_DIR="${script_dir}/${VERSION_DIR}"
readonly IMAGE_TAG="triplet-reid-verify:${VERSION_DIR}"

readonly CKPT_REL="experiments/test_inference/checkpoint-4000"
readonly FIXTURE_REL="data/query/9999/unmarked_21_06_12_173310_left.png"
readonly BASELINE_H5_REL="test_embedding_old.h5"
readonly SIG_BASELINE="${INFERENCE_DIR}/wibex_model_v03_signature_baseline.txt"

require_cmd() {
  command -v "$1" &>/dev/null || { printf 'required command not found: %s\n' "$1" >&2; exit 1; }
}
require_file() {
  [[ -e "$1" ]] || { printf 'required path not found: %s\n' "$1" >&2; exit 1; }
}

require_cmd docker
docker info &>/dev/null || { printf 'Docker daemon not reachable -- is Docker running?\n' >&2; exit 1; }
require_file "${DOCKERFILE_DIR}/Dockerfile"
require_file "${REPO_ROOT}/export_saved_model.py"
require_file "${INFERENCE_DIR}/${CKPT_REL}.index"
require_file "${INFERENCE_DIR}/${FIXTURE_REL}"
require_file "${INFERENCE_DIR}/${BASELINE_H5_REL}"
require_file "${SIG_BASELINE}"
require_file "${HOST_RUNBOOK_DIR}/phase4a_checkpoint_varnames.py"
require_file "${HOST_RUNBOOK_DIR}/phase4b_numeric_gate.py"
require_file "${HOST_RUNBOOK_DIR}/phase4c_extract_signature.py"

tmpdir="$(mktemp -d)"
trap 'rm -rf -- "${tmpdir}"' EXIT

printf '\n== Building %s from %s ==\n' "${IMAGE_TAG}" "${DOCKERFILE_DIR}"
docker build -t "${IMAGE_TAG}" "${DOCKERFILE_DIR}"

printf '\n== [%s] Phase 4a: checkpoint variable-name diff -- --network=none ==\n' "${VERSION_DIR}"
docker run --rm --network=none \
  -v "${INFERENCE_DIR}:/data:ro" \
  -v "${REPO_ROOT}:/work:ro" \
  -v "${HOST_RUNBOOK_DIR}:/probe:ro" \
  -v "${tmpdir}:/out" \
  -w /work \
  -e PYTHONPATH=/work \
  "${IMAGE_TAG}" \
  python3 /probe/phase4a_checkpoint_varnames.py --checkpoint "/data/${CKPT_REL}" --out /out/varname_diff.txt || phase4a_exit=$?
[[ -s "${tmpdir}/varname_diff.txt" ]] && cat -- "${tmpdir}/varname_diff.txt"
if [[ "${phase4a_exit:-0}" -ne 0 ]]; then
  printf '[%s] Phase 4a FAILED (exit %d) -- stopping before restore.\n' "${VERSION_DIR}" "${phase4a_exit}" >&2
  exit "${phase4a_exit}"
fi

printf '\n== [%s] Phase 4b: running the export -- --network=none ==\n' "${VERSION_DIR}"
docker run --rm --network=none \
  -v "${INFERENCE_DIR}:/data:ro" \
  -v "${REPO_ROOT}:/work:ro" \
  -v "${tmpdir}:/out" \
  -w /work \
  "${IMAGE_TAG}" \
  python3 export_saved_model.py --checkpoint "/data/${CKPT_REL}" --export-dir /out/export_test

printf '\n== [%s] Phase 4b: numeric equivalence check (atol=1e-4) -- --network=none ==\n' "${VERSION_DIR}"
docker run --rm --network=none \
  -v "${INFERENCE_DIR}:/data:ro" \
  -v "${tmpdir}:/out:ro" \
  -v "${HOST_RUNBOOK_DIR}:/probe:ro" \
  "${IMAGE_TAG}" \
  python3 /probe/phase4b_numeric_gate.py \
    --export-dir /out/export_test \
    --fixture "/data/${FIXTURE_REL}" \
    --baseline-h5 "/data/${BASELINE_H5_REL}"

printf '\n== [%s] Phase 4c: signature parity -- --network=none (saved_model_cli ships in the base image) ==\n' "${VERSION_DIR}"
docker run --rm --network=none \
  -v "${tmpdir}:/out:ro" \
  "${IMAGE_TAG}" \
  saved_model_cli show --dir /out/export_test --all \
  > "${tmpdir}/new_export_signature.txt"
cat -- "${tmpdir}/new_export_signature.txt"

docker run --rm --network=none \
  -v "${HOST_RUNBOOK_DIR}:/probe:ro" \
  -v "${INFERENCE_DIR}:/data:ro" \
  "${IMAGE_TAG}" \
  python3 /probe/phase4c_extract_signature.py "/data/wibex_model_v03_signature_baseline.txt" \
  > "${tmpdir}/baseline_serving_default.txt"
docker run --rm --network=none \
  -v "${HOST_RUNBOOK_DIR}:/probe:ro" \
  -v "${tmpdir}:/out:ro" \
  "${IMAGE_TAG}" \
  python3 /probe/phase4c_extract_signature.py "/out/new_export_signature.txt" \
  > "${tmpdir}/new_serving_default.txt"

if diff -u -- "${tmpdir}/baseline_serving_default.txt" "${tmpdir}/new_serving_default.txt"; then
  printf '\n== [%s] Phase 4c PASSED -- serving_default signature matches exactly. ==\n' "${VERSION_DIR}"
else
  printf '[%s] Phase 4c: serving_default signature differs (see diff above).\n' "${VERSION_DIR}" >&2
  exit 1
fi

printf '\n== [%s] ALL PARTS PASSED. ==\n' "${VERSION_DIR}"
