"""Check delivery, package fixed evidence, restore outputs, and clean disposable caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmark")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_archive(path):
    """Verify an exact manifest and safe paths before any evidence is restored."""
    path = Path(path)
    if path.stat().st_size > 50_000_000:
        raise ValueError("Source ZIP exceeds 50 MB")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or sum(i.file_size for i in infos) > 500_000_000:
            raise ValueError("Duplicate members or excessive expanded size")
        for info in infos:
            name = info.filename
            parts = PurePosixPath(name)
            if (parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name
                    or parts.as_posix() != name or info.is_dir()
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError("Unsafe ZIP member: " + name)
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        if set(names) != set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}:
            raise ValueError("ZIP members and manifest differ")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("ZIP member hash mismatch: " + name)
    receipt = path.with_suffix(".verification.json")
    archive_sha = sha(path)
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if recorded["sha256"] != archive_sha or recorded["size_bytes"] != path.stat().st_size:
            raise ValueError("Archive and external receipt differ")
    return dict(manifest=manifest, files=len(names), sha256=archive_sha,
                size_bytes=path.stat().st_size, all_checks_passed=True)


def restore_evidence(archive_path, root=ROOT):
    """Restore only output evidence, without overwriting a differing local file."""
    root = Path(root).resolve()
    checked = inspect_archive(archive_path)
    files = checked["manifest"]["files"]
    for name in ["benchmark/submission_frozen_v1.json", "benchmark/FINAL_EVALUATION_LOCK.json"]:
        if name not in files or sha(root / name) != files[name]:
            raise ValueError("Archive belongs to a different frozen program: " + name)
    output_root = (root / "benchmark/outputs").resolve()
    if (root / "benchmark/outputs").is_symlink():
        raise ValueError("Output root must not be a symlink")
    pending = []
    for name, expected in files.items():
        if not name.startswith("benchmark/outputs/"):
            continue
        destination = root / name
        destination.resolve().relative_to(output_root)
        for parent in destination.parents:
            if parent == root:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError("Evidence parent is a file: " + str(parent))
            if parent.relative_to(root).as_posix() in files:
                raise ValueError("Archive member is both a file and a parent directory")
        if any(p.is_symlink() for p in [destination, *destination.parents] if p != root.parent):
            raise ValueError("Evidence path must not contain symlinks")
        if destination.exists():
            if not destination.is_file() or sha(destination) != expected:
                raise ValueError("Existing evidence differs; nothing overwritten: " + name)
        else:
            pending.append((name, destination))
    with zipfile.ZipFile(archive_path) as archive:
        for name, destination in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(archive.read(name))
    return dict(restored_files=len(pending), source_sha256=checked["sha256"],
                code_overwritten=False, final_evaluation_run=False)


def clean_caches(root=ROOT, tracked=()):
    """Delete only generated cache directories; never descend into research or environments."""
    root = Path(root).resolve()
    tracked = {Path(p).as_posix() for p in tracked}
    prune = {".git", ".venv", "dist", ".aws", ".agents", ".codex"}
    cache_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".mplconfig"}
    targets = []
    for current, directories, _ in os.walk(root, followlinks=False):
        for name in directories[:]:
            path = Path(current) / name
            if (name in prune or path.is_symlink() or path == root / "benchmark/outputs"):
                directories.remove(name)
            elif name in cache_names:
                relative = path.relative_to(root).as_posix()
                if any(p == relative or p.startswith(relative + "/") for p in tracked):
                    raise ValueError("Refusing to delete a tracked cache: " + relative)
                targets.append(path)
                directories.remove(name)
    reclaimed = sum(p.stat().st_size for t in targets for p in t.rglob("*") if p.is_file() and not p.is_symlink())
    for path in targets:
        shutil.rmtree(path)
    return dict(removed_directories=[str(p.relative_to(root)) for p in targets], reclaimed_bytes=reclaimed)


def validate_release(release, bundle):
    """Bind the delivery decision to the exact frozen model and completed customer check."""
    if (release.get("status") != "ready" or release.get("selected_policy") != bundle["selected_policy"]
            or release.get("bundle_sha256") != bundle["bundle_sha256"]
            or release.get("code_sha256") != bundle["code_sha256"]
            or release.get("customer_verification_passed") is not True
            or release.get("holdout_reused") is not False
            or release.get("new_final_evaluation") is not False):
        raise ValueError("Release decision does not match the verified frozen program and customer checks")


def check_project(root=ROOT):
    from safebench.experiment import verify_bundle
    root = Path(root)
    settings = json.loads((root / "configs/delivery.json").read_text())
    raw = json.loads((root / settings["bundle"]).read_text())
    bundle = verify_bundle(raw)
    lock = json.loads((root / "benchmark/FINAL_EVALUATION_LOCK.json").read_text())
    if lock["bundle_sha256"] != raw["bundle_sha256"]:
        raise ValueError("Final lock and frozen model differ")
    release = json.loads((root / settings["release_decision"]).read_text())
    validate_release(release, raw)
    required = ["README.md", "LOG.md", "docs/SOLUTION_GUIDE.md", "docs/SUBMISSION.md",
                "requirements.lock.txt", "pyproject.toml", ".gitignore", ".gitattributes"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    evidence = [settings["development_run"] + "/scenario_metrics.csv",
                settings["final_run"] + "/decisions.csv.gz",
                settings["final_run"] + "/scenario_metrics.csv"]
    missing = [p for p in evidence if not (root / p).is_file()]
    return dict(passed=True, selected_policy=bundle["selected_policy"], frozen_integrity_code_environment=True,
                customer_mapping={p["profile_id"]: p["portfolio"] for p in bundle["frozen_suitability"]},
                full_evidence_available=not missing, missing_large_evidence=missing,
                next_step="可打包；打包会再核验全部依赖。" if not missing else
                          "Git紧凑证据可阅读/测试；打包或重建讲解前，用make restore PACKAGE=<可信ZIP>恢复大证据。",
                final_evaluation_run=False)


def main():
    parser = argparse.ArgumentParser(description="项目检查、固定源码打包、证据恢复与缓存清理")
    parser.add_argument("action", choices=["check", "package", "verify", "restore", "clean", "guide"])
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/delivery.json").read_text())
    archive = (args.archive or ROOT / settings["archive"]).resolve()
    if args.action == "clean":
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0") if (ROOT / ".git").exists() else []
        result = clean_caches(ROOT, tracked)
    elif args.action == "check":
        result = check_project()
    elif args.action == "verify":
        result = inspect_archive(archive)
        result.pop("manifest")
    elif args.action == "restore":
        result = restore_evidence(archive)
    elif args.action == "guide":
        from build_solution_guide import build
        destination = ROOT / "docs/SOLUTION_GUIDE.md"
        destination.write_text(build(ROOT / settings["research_guide_run"]))
        result = dict(guide=str(destination), saved_results_only=True, market_data_read=False)
    else:
        from package_submission import build
        check_project()
        archive.relative_to(ROOT / "dist")
        if archive.exists():
            parser.error("目标ZIP已存在，未覆盖；用make verify-package核验，或指定PACKAGE=dist/<新文件名>.zip")
        build(ROOT / settings["development_run"], ROOT / settings["bundle"], ROOT / settings["final_run"],
              archive, [ROOT / p for p in settings["research_runs"]], ROOT / settings["release_decision"])
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
