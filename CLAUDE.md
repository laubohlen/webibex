# webibex — Project Instructions

Local to this project, loaded alongside `~/.claude/CLAUDE.md`.

## Environment gotchas

- **Module-level/def-time-eager imports break prod but not test venv.** Code that imports or reads config at module scope (not lazily, e.g. Django settings, `mypy_boto3_s3`, `setuptools`/`pkg_resources`) can behave correctly in the dev/test venv while breaking in a prod-shaped environment with only `requirements.txt` installed — caused a real production outage (2026-08-15, `mypy_boto3_s3` unconditional module-level import). A `prod_parity` pytest marker was added afterward; as a standing pre-deploy check, also verify `webibex.wsgi` imports cleanly with only `requirements.txt` installed and prod-shaped env vars, not just via pytest-django.
- **Secrets fallback**: `~/.config/secrets/<NAME>` works as a token source in this sandbox when macOS Keychain doesn't — try both before declaring a secret unreachable (e.g. `SONAR_TOKEN`, `BRAIN_TOKEN`).
