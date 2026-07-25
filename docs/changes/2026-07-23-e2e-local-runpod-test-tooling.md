# CR: local e2e test tooling for RunPod-backed TF pipeline changes

**What changed:**
- New (gitignored scratch, `tmp/` — not committed):
  `tmp/inference/host_runbook/e2e_tf2210_manual_test.sh` — exports a
  persisted SavedModel via a `training/triplet-reid/dockerfiles/<version>`
  image, builds a throwaway RunPod-inference image from it, runs the real
  `handler.py` in RunPod SDK local-test mode (`test_input.json`).
- New (same, gitignored):
  `tmp/inference/host_runbook/start_local_rp_server.sh` — runs that
  RunPod-inference image as a persistent local HTTP server
  (`--rp_serve_api`), joined to the devcontainer's own network namespace
  (`--network container:<devcontainer-id>`) rather than port-published to
  the host.

**Follow-up action:** none required — these are host-only scratch scripts,
not app code. Re-derive them from this doc if they're lost (they don't
persist across `tmp/` cleanup).

**Do NOT:**
- Port-publish the local RunPod server to the host (`-p 8000:8000`) and
  expect the devcontainer to reach it via `host.docker.internal` or the
  Docker Desktop gateway IP — confirmed unreachable this session even
  with the server verified listening on the host side. Use
  `--network container:<devcontainer-id>` instead (see
  `[[project_devcontainer_network_sandbox]]` memory for the full
  writeup) — this makes "localhost:<port>" from inside the devcontainer
  reach the sibling container directly.
- Assume the devcontainer's own primary port (e.g. 8000, used by
  `manage.py runserver`) is free inside that shared network namespace —
  pick a different port for the RunPod server (this session used 8001).
- Confuse this with `verify_gate.sh`'s own Phase 4a/4b/4c gate — that
  gate already proves numeric/signature correctness at the export level.
  This tooling instead proves the *full request path* (Django →
  `endpoint_inference()` → HTTP → `handler.py` → model), which the gate
  doesn't touch at all.

**Trigger:** whenever verifying a TF-version change (or any change to
`training/triplet-reid/{nets,export_saved_model.py}` or
`tmp/inference/runpod_ibex_embedding_endpoint/handler.py`) actually works
end-to-end through the real Django app, not just at the export/handler
level in isolation.

**Why:** the existing `verify_gate.sh` gate and the handler-level
`--rp_serve_api` local-test-input run both already existed as verification
tools, but neither exercises the real Django→HTTP request path a user
actually triggers. Standing up a real local RunPod server that Django's
own `core/utils.py:endpoint_inference()` can call (via a temporary URL
override, reverted after) closes that gap without needing an actual
RunPod cloud deployment. The network-namespace-sharing technique was
required because this devcontainer's outbound network is more restricted
than plain "no internet" — it can't reach arbitrary host IPs at all, only
specific carve-outs (PyPI via squid-proxy, SonarQube's documented
`host.docker.internal:9000`).

**Verify:** `curl http://localhost:<port>/` from inside the devcontainer
returns `200` once `start_local_rp_server.sh` is running; a full
`e2e_tf2210_manual_test.sh` run ends with `Job local_test completed
successfully` and a real embedding vector printed.

**Rollback:** delete the two scripts from `tmp/inference/host_runbook/`
(gitignored, not tracked — no git history to revert). No app code is
touched by this CR.
