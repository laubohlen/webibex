#!/usr/bin/env bash
# Starts the tf2210 RunPod inference image as a local HTTP server (RunPod
# SDK's local-server test mode), joined to a devcontainer's OWN network
# namespace (--network container:<id>) instead of port-published to the
# host. The webibex devcontainer's outbound network is sandboxed --
# host.docker.internal and the Docker Desktop gateway IP both timed out when
# the server was port-published to the host instead. Sharing the netns means
# "localhost" inside the devcontainer reaches this container directly, no
# tunnel needed.
#
# Hardened version of the former tmp/inference/host_runbook/ scratch script:
# the devcontainer id, image tag, and port are now positional arguments
# instead of a hardcoded, ephemeral value from one prior session.
#
# RUN THIS ON HOST -- needs a real Docker daemon.
#
# Usage: ./start_local_rp_server.sh <devcontainer_id> [image_tag] [port]
#   devcontainer_id  required. The container ID/name whose network namespace
#                    to join. Get the RIGHT one by running `hostname` (or
#                    `cat /etc/hostname`) INSIDE the target devcontainer
#                    itself -- Docker sets a container's hostname to its own
#                    short ID by default, so this is exact and unambiguous.
#                    Do NOT try to guess it from the host via
#                    `docker ps -f name=...`: devcontainer-guard-managed
#                    workspace containers carry no label identifying which
#                    repo/project they're hosting (only a launcher-tool
#                    project name like "claude-devcontainer" or
#                    "claude-devcontainer-2"), so with multiple concurrent
#                    devcontainer sessions running, a name-based guess can
#                    silently pick the WRONG container -- the script would
#                    still run without error, but --network container:<id>
#                    would join the wrong netns and "localhost:<port>" from
#                    the intended devcontainer would not reach it.
#   image_tag        default: ibex-embedding-tf2210:e2e-test
#   port             default: 8001 (port 8000 inside the shared namespace is
#                    already used by the devcontainer's own Django dev
#                    server).
#
# If --rp_serve_api / --rp_api_port error out or don't behave as expected,
# run: docker run --rm <image_tag> python3 /handler.py --help
# to confirm the actual flag names for this runpod SDK version, then edit
# this script accordingly.

set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true

readonly DEFAULT_IMAGE_TAG="ibex-embedding-tf2210:e2e-test"
readonly DEFAULT_PORT="8001"

require_cmd() {
  command -v "$1" &>/dev/null || { printf 'required command not found: %s\n' "$1" >&2; exit 1; }
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf 'usage: %s <devcontainer_id> [image_tag] [port]\n' "$0" >&2
  printf '  image_tag default: %s\n' "${DEFAULT_IMAGE_TAG}" >&2
  printf '  port default: %s\n' "${DEFAULT_PORT}" >&2
  exit 1
fi

readonly DEVCONTAINER_ID="$1"
readonly IMAGE_TAG="${2:-${DEFAULT_IMAGE_TAG}}"
readonly PORT="${3:-${DEFAULT_PORT}}"

require_cmd docker
docker info &>/dev/null || { printf 'Docker daemon not reachable -- is Docker running?\n' >&2; exit 1; }

# `--` before the user-controlled devcontainer_id defends against a value
# like `--privileged` being interpreted as a docker flag instead of the
# container-id argument.
if ! container_state="$(docker container inspect -f '{{.State.Running}}' -- "${DEVCONTAINER_ID}" 2>/dev/null)"; then
  printf 'container not found: %s\n' "${DEVCONTAINER_ID}" >&2
  exit 1
fi
if [[ "${container_state}" != "true" ]]; then
  printf 'container not running: %s\n' "${DEVCONTAINER_ID}" >&2
  exit 1
fi

if ! docker image inspect -- "${IMAGE_TAG}" &>/dev/null; then
  printf 'image not found: %s -- build it via e2e_tf2210_manual_test.sh first\n' "${IMAGE_TAG}" >&2
  exit 1
fi

if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
  printf 'invalid port: %s\n' "${PORT}" >&2
  exit 1
fi

docker run --rm --network "container:${DEVCONTAINER_ID}" \
  -- "${IMAGE_TAG}" \
  python3 -u /handler.py --rp_serve_api --rp_api_port "${PORT}"
