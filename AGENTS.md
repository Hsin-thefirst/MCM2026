# Agent instructions

- Read `README.md`, `docs/INTERFACES.md`, and `LOG.md` before starting work.
- Keep one source of truth: the root README and log, `docs/` for documentation, `configs/` for inputs, `tests/` for tests, and `scripts/` for utility commands. Do not recreate parallel copies under `benchmark/`.
- Write human-facing documentation and progress logs in Chinese. Write agent instructions, code identifiers, and developer comments in English. Preserve source quotations and competition-required English customer explanations where appropriate.
- After a completed work segment, append a short dated entry to `LOG.md`: executor, completed changes, files, verification, and next step. Preserve earlier entries and their meaning.
- Update implementation and `docs/INTERFACES.md` together when a public interface changes.
- Use scenarios 1–120 only for development. Routine tests must use generated fixtures or the development partition, never the real final holdout. Final evaluation requires the frozen-bundle workflow in `docs/PROTOCOL.md`.
- Use `make test` for the unified test suite. Record new development experiments in `benchmark/experiment_registry.json`; do not overwrite historical manifests or change their hashes after refactoring.
- Dependency ranges live in `pyproject.toml`; `requirements.lock.txt` records the exact verified runtime environment. Do not create a second manually maintained dependency-range list.
