#!/usr/bin/env bash
# Guard/argument-parsing test harness for ../start_local_rp_server.sh.
#
# Full real-Docker E2E of the hardened script is explicitly out of scope
# here (no Docker daemon reachable in the sandbox that authored this
# harness) -- these 10 scenarios (SH01-SH10) exercise only the guard order
# and argument-parsing logic via a fake `docker` binary placed on PATH.
#
# Run directly: bash training/triplet-reid/dockerfiles/tests/test_start_local_rp_server.sh
# Not collected by pytest (pytest.ini's python_files pattern only matches
# .py files, and this lives outside `testpaths` = "core simple_landmarks").
#
# Sequencing note: there was no prior *hardened* version of the script to be
# RED against -- the original tmp/inference/host_runbook/start_local_rp_server.sh
# had zero argument parsing (a single hardcoded DEVCONTAINER_ID constant) and
# zero guards, so none of SH01-SH10's argv shapes could have been exercised
# against it meaningfully (every scenario would fail for the wrong reason:
# "script ignores its arguments entirely," not "guard rejects bad input").
# This harness was therefore written together with the hardened script and
# confirmed to pass against it, rather than run RED against an unhardened
# predecessor that shares no argument contract with what's being tested.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

[[ -n "${TRACE:-}" && -z "${BASH_XTRACEFD:-}" ]] && set -x

# Resolve bash BEFORE any PATH restriction (SH01 restricts PATH to a
# docker-less directory) -- always invoke the SUT via this absolute path,
# never by relying on PATH to resolve `bash` or `env`.
REAL_BASH="$(command -v bash)"
readonly REAL_BASH

# Original PATH, restored inside each scenario immediately after invoking
# the SUT -- scenarios temporarily narrow PATH to a stub/empty dir for the
# SUT call, but the harness itself (mktemp, cat, grep) still needs a
# working PATH for every statement that follows.
HARNESS_PATH="${PATH}"
readonly HARNESS_PATH

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SUT="${script_dir}/../start_local_rp_server.sh"

pass_count=0
fail_count=0

report() {
  local name="$1" ok="$2" detail="${3:-}"
  if [[ "${ok}" -eq 0 ]]; then
    printf 'PASS %s\n' "${name}"
    pass_count=$((pass_count + 1))
  else
    printf 'FAIL %s: %s\n' "${name}" "${detail}" >&2
    fail_count=$((fail_count + 1))
  fi
}

# Builds a private bin/ dir containing a `docker` stub that (a) appends its
# full received argv to $DOCKER_STUB_LOG, (b) returns a controllable exit
# code / stdout based on FAKE_DOCKER_* env var toggles.
make_docker_stub_dir() {
  local dir="$1"
  mkdir -p -- "${dir}"
  # Absolute shebang (not `#!/usr/bin/env bash`) -- scenarios restrict PATH
  # to this stub dir only when invoking the SUT, so `env` would not be able
  # to resolve `bash` either. An absolute interpreter path bypasses PATH
  # lookup at exec() time entirely.
  printf '#!%s\n' "${REAL_BASH}" > "${dir}/docker"
  cat >> "${dir}/docker" <<'STUB'
set -Eeuo pipefail
: "${DOCKER_STUB_LOG:?DOCKER_STUB_LOG not set}"
printf '%s\n' "$*" >> "${DOCKER_STUB_LOG}"
case "${1:-}" in
  info)
    exit "${FAKE_DOCKER_INFO_RC:-0}"
    ;;
  container)
    if [[ "${2:-}" == "inspect" ]]; then
      case "${FAKE_DOCKER_CONTAINER_STATE:-running}" in
        absent) exit 1 ;;
        stopped) printf 'false\n'; exit 0 ;;
        running) printf 'true\n'; exit 0 ;;
        *) exit 1 ;;
      esac
    fi
    exit 1
    ;;
  image)
    if [[ "${2:-}" == "inspect" ]]; then
      case "${FAKE_DOCKER_IMAGE:-present}" in
        absent) exit 1 ;;
        present) printf '{}\n'; exit 0 ;;
        *) exit 1 ;;
      esac
    fi
    exit 1
    ;;
  run)
    exit "${FAKE_DOCKER_RUN_RC:-0}"
    ;;
  *)
    exit 1
    ;;
esac
STUB
  chmod +x -- "${dir}/docker"
}

# --- SH01 --------------------------------------------------------------
scenario_sh01() {
  local name="SH01_docker_binary_missing"
  local -x PATH="${PATH}"  # scoped to this function, restored on return
  local empty_dir err_file ec stderr_content
  empty_dir="$(mktemp -d)"
  err_file="$(mktemp)"
  PATH="${empty_dir}"

  set +e
  "${REAL_BASH}" "${SUT}" some-container-id 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]] && grep -qi "docker" <<<"${stderr_content}" \
      && grep -qi "not found" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -rf -- "${empty_dir}"
  rm -f -- "${err_file}"
}
# FALSIFIED op:PRECOND inv:"require_cmd docker guard rejects when the docker binary is absent from PATH"

# --- SH02 --------------------------------------------------------------
scenario_sh02() {
  local name="SH02_docker_info_fails"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local -x FAKE_DOCKER_INFO_RC=1
  local stub_dir log_file err_file ec stderr_content
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" some-container-id 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]] && grep -qi "daemon" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:PRECOND inv:"docker info reachability guard rejects when the daemon is unreachable"

# --- SH03 --------------------------------------------------------------
scenario_sh03() {
  local name="SH03_zero_args"
  local err_file ec stderr_content
  err_file="$(mktemp)"

  set +e
  "${REAL_BASH}" "${SUT}" 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]] && grep -qi "usage" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -f -- "${err_file}"
}
# FALSIFIED op:PRECOND inv:"arg-count guard rejects when zero positional args are supplied"

# --- SH04 --------------------------------------------------------------
scenario_sh04() {
  local name="SH04_container_not_found"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local -x FAKE_DOCKER_CONTAINER_STATE=absent
  local stub_dir log_file err_file ec stderr_content
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" some-container-id 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]] && grep -qi "container not found" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:PRECOND inv:"container inspect guard rejects when the container id does not exist"

# --- SH05 --------------------------------------------------------------
scenario_sh05() {
  local name="SH05_container_not_running"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local -x FAKE_DOCKER_CONTAINER_STATE=stopped
  local stub_dir log_file err_file ec stderr_content
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" some-container-id 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]] && grep -qi "container not running" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:PRECOND inv:"container-running guard rejects when State.Running reports false"

# --- SH06 --------------------------------------------------------------
scenario_sh06() {
  local name="SH06_image_not_found"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local -x FAKE_DOCKER_IMAGE=absent
  local stub_dir log_file err_file ec stderr_content
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" some-container-id 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]] && grep -qi "image not found" <<<"${stderr_content}" \
      && grep -q "e2e_tf2210_manual_test.sh" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:PRECOND inv:"image inspect guard rejects when the image tag is not present locally"

# --- SH07 --------------------------------------------------------------
scenario_sh07() {
  local bad_port
  for bad_port in "80a1" "-1" "8001;ls" "abc"; do
    local name="SH07_invalid_port_${bad_port}"
    local -x PATH="${PATH}"
    local -x DOCKER_STUB_LOG
    local stub_dir log_file err_file ec stderr_content
    stub_dir="$(mktemp -d)"
    make_docker_stub_dir "${stub_dir}"
    log_file="$(mktemp)"
    err_file="$(mktemp)"
    PATH="${stub_dir}"
    DOCKER_STUB_LOG="${log_file}"

    set +e
    "${REAL_BASH}" "${SUT}" some-container-id some-image:tag "${bad_port}" 2>"${err_file}"
    ec=$?
    set -e
    PATH="${HARNESS_PATH}"

    stderr_content="$(cat -- "${err_file}")"
    if [[ "${ec}" -ne 0 ]] && grep -qi "invalid port" <<<"${stderr_content}"; then
      report "${name}" 0
    else
      report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
    fi
    rm -rf -- "${stub_dir}"
    rm -f -- "${log_file}" "${err_file}"
  done

  # Empty $3 must NOT hit the invalid-port path -- it must default to 8001
  # and reach a successful docker run instead.
  local name="SH07_empty_port_defaults_not_rejected"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local stub_dir log_file err_file ec stderr_content
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" some-container-id some-image:tag "" 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -eq 0 ]] && ! grep -qi "invalid port" <<<"${stderr_content}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:ROR inv:"numeric port regex rejects non-numeric/negative/injection-shaped values but not an empty (default-triggering) value"

# --- SH08 --------------------------------------------------------------
scenario_sh08() {
  local name="SH08_happy_path_defaults"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local stub_dir log_file err_file ec stderr_content run_line
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" my-devcontainer-id 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]]; then
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
    rm -rf -- "${stub_dir}"
    rm -f -- "${log_file}" "${err_file}"
    return
  fi

  [[ -s "${log_file}" ]] || { report "${name}" 1 "docker stub log is empty"; rm -rf -- "${stub_dir}"; rm -f -- "${log_file}" "${err_file}"; return; }
  run_line="$(tail -n 1 -- "${log_file}")"
  if grep -q -- "--network" <<<"${run_line}" \
      && grep -q "container:my-devcontainer-id" <<<"${run_line}" \
      && grep -q "ibex-embedding-tf2210:e2e-test" <<<"${run_line}" \
      && grep -q -- "--rp_api_port 8001" <<<"${run_line}"; then
    report "${name}" 0
  else
    report "${name}" 1 "run_line=${run_line}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:SDL inv:"docker run receives --network container:<id> plus the default image tag and port when only devcontainer_id is supplied"

# --- SH09 --------------------------------------------------------------
scenario_sh09() {
  local name="SH09_happy_path_all_args_custom"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local stub_dir log_file err_file ec stderr_content run_line
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" my-devcontainer-id custom-image:custom-tag 9999 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  if [[ "${ec}" -ne 0 ]]; then
    report "${name}" 1 "ec=${ec} stderr=${stderr_content}"
    rm -rf -- "${stub_dir}"
    rm -f -- "${log_file}" "${err_file}"
    return
  fi

  [[ -s "${log_file}" ]] || { report "${name}" 1 "docker stub log is empty"; rm -rf -- "${stub_dir}"; rm -f -- "${log_file}" "${err_file}"; return; }
  run_line="$(tail -n 1 -- "${log_file}")"
  if grep -q "custom-image:custom-tag" <<<"${run_line}" \
      && grep -q -- "--rp_api_port 9999" <<<"${run_line}" \
      && ! grep -q "ibex-embedding-tf2210:e2e-test" <<<"${run_line}" \
      && ! grep -q -- "--rp_api_port 8001" <<<"${run_line}"; then
    report "${name}" 0
  else
    report "${name}" 1 "run_line=${run_line}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:SDL inv:"docker run receives the custom image tag and port, not the defaults, when all three args are supplied"

# --- SH10 --------------------------------------------------------------
scenario_sh10() {
  local name="SH10_option_injection_shaped_devcontainer_id"
  local -x PATH="${PATH}"
  local -x DOCKER_STUB_LOG
  local -x FAKE_DOCKER_CONTAINER_STATE=absent
  local stub_dir log_file err_file ec stderr_content inspect_line
  stub_dir="$(mktemp -d)"
  make_docker_stub_dir "${stub_dir}"
  log_file="$(mktemp)"
  err_file="$(mktemp)"
  PATH="${stub_dir}"
  DOCKER_STUB_LOG="${log_file}"

  set +e
  "${REAL_BASH}" "${SUT}" --privileged 2>"${err_file}"
  ec=$?
  set -e
  PATH="${HARNESS_PATH}"

  stderr_content="$(cat -- "${err_file}")"
  [[ -s "${log_file}" ]] || { report "${name}" 1 "docker stub log is empty"; rm -rf -- "${stub_dir}"; rm -f -- "${log_file}" "${err_file}"; return; }
  inspect_line="$(grep -- "^container inspect" "${log_file}" || true)"
  if [[ "${ec}" -ne 0 ]] \
      && grep -qi "container not found" <<<"${stderr_content}" \
      && grep -q -- "-- --privileged" <<<"${inspect_line}"; then
    report "${name}" 0
  else
    report "${name}" 1 "ec=${ec} stderr=${stderr_content} inspect_line=${inspect_line}"
  fi
  rm -rf -- "${stub_dir}"
  rm -f -- "${log_file}" "${err_file}"
}
# FALSIFIED op:PRECOND inv:"-- before the devcontainer_id argument prevents an option-injection-shaped value from being parsed as a docker flag"

main() {
  scenario_sh01
  scenario_sh02
  scenario_sh03
  scenario_sh04
  scenario_sh05
  scenario_sh06
  scenario_sh07
  scenario_sh08
  scenario_sh09
  scenario_sh10

  printf '\n%d passed, %d failed\n' "${pass_count}" "${fail_count}"
  [[ "${fail_count}" -eq 0 ]]
}

main
