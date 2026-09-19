# webibex — Project Instructions

Local to this project, loaded alongside `~/.claude/CLAUDE.md`.

## Environment gotchas

- **Module-level/def-time-eager imports break prod but not test venv.** Code that imports or reads config at module scope (not lazily, e.g. Django settings, `mypy_boto3_s3`, `setuptools`/`pkg_resources`) can behave correctly in the dev/test venv while breaking in a prod-shaped environment with only `requirements.txt` installed — caused a real production outage (2026-08-15, `mypy_boto3_s3` unconditional module-level import). A `prod_parity` pytest marker was added afterward; as a standing pre-deploy check, also verify `webibex.wsgi` imports cleanly with only `requirements.txt` installed and prod-shaped env vars, not just via pytest-django.
- **Secrets fallback**: `~/.config/secrets/<NAME>` works as a token source in this sandbox when macOS Keychain doesn't — try both before declaring a secret unreachable (e.g. `SONAR_TOKEN`, `BRAIN_TOKEN`).

## Supply Chain Exceptions

- **`node/` uses `pnpm-lock.yaml`, not `package-lock.json`/`npm ci`** (deviation from the global convention). `npm` is permanently disabled in this project's devcontainer image (`npm-guard` policy, no runtime override) — `pnpm` is the only workable package manager here. `package.json` still pins exact versions (no `^`/`~`); `pnpm audit` is the CVE-scan equivalent of `npm audit`. See `docs/security-remediation-plan.md` (2026-09-19 JS transitive CVE fix) for the migration history.
