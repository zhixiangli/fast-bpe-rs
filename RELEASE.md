# Release guide

This project is set up to publish `fast-bpe-rs` to PyPI using GitHub Actions, `maturin`, and PyPI trusted publishing.

## One-time repository setup

### 1. Create the PyPI project

1. Create the `fast-bpe-rs` project on PyPI (and optionally TestPyPI).
2. Upload an initial release manually only if you need to reserve the name before enabling automation.

### 2. Configure PyPI trusted publishing

In PyPI, add a **trusted publisher** for this repository with:

- **Owner**: your GitHub organization or username
- **Repository name**: this repository
- **Workflow name**: `release.yml`
- **Environment name**: `pypi`

If you also want a TestPyPI dry-run workflow later, create a separate trusted publisher for the corresponding workflow/environment.

### 3. GitHub repository settings

Recommended repository settings:

- protect your default branch,
- require the `CI` workflow to pass before merging,
- allow GitHub Actions to create releases,
- optionally require signed tags/releases if your organization uses them.

## Normal release flow

1. Update `Cargo.toml` with the next semantic version.
2. Make sure the package metadata in `pyproject.toml` and `README.md` still matches the release.
3. Run the local verification commands:

   ```bash
   cargo fmt --all --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --all-features
   uv sync --extra dev
   uv run maturin develop --release
   uv run pytest
   uv run maturin build --release --strip --sdist --out dist
   uv run twine check dist/*
   ```

4. Commit the version bump.
5. Create and push a git tag that matches the version, for example:

   ```bash
   git tag v0.1.0
   git push origin main --follow-tags
   ```

6. The `Release` workflow will:
   - build wheels for Linux, macOS, and Windows,
   - build an sdist,
   - verify the generated distributions,
   - publish them to PyPI using trusted publishing,
   - create a GitHub Release and upload the artifacts.

## Rollback / recovery

- If PyPI publication fails before upload, fix the workflow or PyPI settings and re-run it.
- If a bad version is published to PyPI, do not overwrite it. Publish a new version instead.
- If a GitHub Release exists without a PyPI upload, fix the pipeline and publish a new tag/version rather than mutating old artifacts.
