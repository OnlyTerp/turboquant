# Releasing TurboQuant to PyPI

This repo ships `turboquant` to [PyPI](https://pypi.org/project/turboquant/) via
**Trusted Publisher OIDC** — no long-lived `PYPI_API_TOKEN` lives in GitHub.

## One-time setup (do this once, as the PyPI project owner)

1. Reserve the project on PyPI: https://pypi.org/manage/account/publishing/
2. Add a **Pending publisher** with:
   - PyPI Project Name: `turboquant`
   - Owner: `OnlyTerp`
   - Repository name: `turboquant`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In GitHub → Settings → Environments → **New environment** `pypi`.
   - Add a deployment branch rule: tags matching `v*`.
   - No secrets needed (OIDC handles auth).
4. First-time publish: push a `v0.2.0` tag. The pending-publisher entry will
   auto-upgrade to a live publisher on the first successful run.

## Per-release workflow

```bash
# 1. Bump version in two places (they must match)
#    - pyproject.toml    -> [project] version = "X.Y.Z"
#    - src/__init__.py   -> __version__ = "X.Y.Z"

# 2. Update CHANGELOG (inline below or in a CHANGELOG.md — convention TBD)

# 3. Commit + push to master through a PR

# 4. Tag and push
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z

# 5. Watch the Actions run. On success:
#    - sdist + wheel are built
#    - twine check passes
#    - wheel smoke-installs into a clean venv with CPU torch
#    - gh-action-pypi-publish uploads via OIDC
#    - Release appears at https://pypi.org/project/turboquant/
```

## Versioning policy

We follow SemVer with some pragmatism:

- **0.Y.Z** pre-1.0: the `b_mse` / `b_outlier` / `rotation_mode` constructor
  knobs may change between minor versions. The `TurboQuantCache.__call__`
  attention path stays stable.
- **1.0.0**: target after (a) real-model LongBench parity is published and
  (b) at least one upstream engine (vLLM or SGLang) has merged a stable
  `--kv-cache-dtype turboquant` path.
- Patch releases (0.2.x) are for pure bugfixes and doc updates.

## Pre-release checklist

- [ ] `pytest src/test_turboquant.py` passes locally
- [ ] `python src/demo.py` produces expected 3.5-bit ≥ 0.95 avg cosine
- [ ] `python reports/scripts/check_thresholds.py` passes
- [ ] `python -m build` succeeds and `twine check dist/*` is green
- [ ] Version in `pyproject.toml` and `src/__init__.py` match
- [ ] `README.md` Quick Start block still works on the next tagged version
- [ ] If the public API changed, update `src/__init__.py` `__all__` and
      re-check `INTEGRATIONS.md` code examples

## Emergency yank

If a broken release is uploaded:

```bash
pip install twine
python -m twine upload --skip-existing dist/turboquant-X.Y.Z.tar.gz  # no-op; version is already taken
# ... instead, on PyPI web UI, "Yank release" the bad version and re-tag vX.Y.(Z+1)
```

PyPI versions are immutable: always bump the patch and re-tag.
