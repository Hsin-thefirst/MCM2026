"""Generated archives verify safe evidence restore and non-destructive cache cleanup."""
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from project_tools import inspect_archive, restore_evidence, clean_caches, validate_release


def archive(path, files):
    manifest = dict(files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()})
    with zipfile.ZipFile(path, "w") as z:
        for name, value in files.items():
            z.writestr(name, value)
        z.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest))


class ProjectToolsTests(unittest.TestCase):
    def test_release_must_match_exact_frozen_bundle_and_safeguards(self):
        bundle = dict(selected_policy="fixture", bundle_sha256="bundle-a", code_sha256="code-a")
        release = dict(bundle, status="ready", customer_verification_passed=True,
                       holdout_reused=False, new_final_evaluation=False)
        validate_release(release, bundle)
        for field, value in [("bundle_sha256", "bundle-b"), ("code_sha256", "code-b"),
                             ("selected_policy", "other"), ("status", "pending"),
                             ("customer_verification_passed", False), ("holdout_reused", True),
                             ("new_final_evaluation", True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_release(dict(release, **{field: value}), bundle)
        missing = dict(release)
        missing.pop("holdout_reused")
        with self.assertRaises(ValueError):
            validate_release(missing, bundle)

    def test_archive_checks_hashes_and_external_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "source.zip"
            archive(p, {"README.md": b"generated"})
            self.assertTrue(inspect_archive(p)["all_checks_passed"])
            p.with_suffix(".verification.json").write_text(json.dumps(dict(sha256="0" * 64, size_bytes=p.stat().st_size)))
            with self.assertRaises(ValueError):
                inspect_archive(p)
            p.with_suffix(".verification.json").unlink()
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("README.md", b"tampered")
                z.writestr("PACKAGE_MANIFEST.json", json.dumps(dict(files={"README.md": hashlib.sha256(b"original").hexdigest()})))
            with self.assertRaises(ValueError):
                inspect_archive(p)

    def test_archive_rejects_traversal_unlisted_members_and_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unsafe.zip"
            for name in ["../escape", "/absolute", "a\\b", "C:/escape", "a/../escape"]:
                with self.subTest(name=name):
                    archive(p, {name: b"generated"})
                    with self.assertRaises(ValueError):
                        inspect_archive(p)
            archive(p, {"README.md": b"generated"})
            with zipfile.ZipFile(p, "a") as z:
                z.writestr("unexpected", b"generated")
            with self.assertRaises(ValueError):
                inspect_archive(p)
            with zipfile.ZipFile(p, "w") as z:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, "outside")
                z.writestr("PACKAGE_MANIFEST.json", json.dumps(dict(files={"link": hashlib.sha256(b"outside").hexdigest()})))
            with self.assertRaises(ValueError):
                inspect_archive(p)

    def test_restore_preflights_all_conflicts_and_never_replaces_code(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            (root / "benchmark").mkdir()
            files = {"benchmark/submission_frozen_v1.json": b"generated bundle",
                     "benchmark/FINAL_EVALUATION_LOCK.json": b"generated lock",
                     "benchmark/outputs/fixture/a.csv": b"first",
                     "benchmark/outputs/fixture/b.csv": b"second", "README.md": b"old archive code"}
            for name in ["benchmark/submission_frozen_v1.json", "benchmark/FINAL_EVALUATION_LOCK.json"]:
                (root / name).write_bytes(files[name])
            p = Path(td) / "source.zip"
            archive(p, files)
            destination = root / "benchmark/outputs/fixture"
            destination.mkdir(parents=True)
            (destination / "b.csv").write_bytes(b"local data")
            with self.assertRaises(ValueError):
                restore_evidence(p, root)
            self.assertFalse((destination / "a.csv").exists())
            (destination / "b.csv").write_bytes(b"second")
            (root / "README.md").write_bytes(b"current source")
            result = restore_evidence(p, root)
            self.assertEqual(result["restored_files"], 1)
            self.assertEqual((root / "README.md").read_bytes(), b"current source")
            self.assertEqual(restore_evidence(p, root)["restored_files"], 0)
            malformed = dict(files)
            malformed["benchmark/outputs/collision"] = b"a file"
            malformed["benchmark/outputs/collision/nested.csv"] = b"child"
            archive(p, malformed)
            with self.assertRaises(ValueError):
                restore_evidence(p, root)
            self.assertFalse((root / "benchmark/outputs/collision").exists())
            archive(p, files)
            (root / "benchmark/FINAL_EVALUATION_LOCK.json").write_bytes(b"another lock")
            with self.assertRaises(ValueError):
                restore_evidence(p, root)

    def test_clean_preserves_results_environment_tracked_files_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            for name in ["scripts/__pycache__/temporary.pyc", "benchmark/outputs/run/__pycache__/evidence",
                         ".venv/lib/__pycache__/dependency", "dist/__pycache__/evidence"]:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"generated")
            outside = Path(td) / "outside/__pycache__"
            outside.mkdir(parents=True)
            (outside / "keep").write_bytes(b"generated")
            (root / "linked").symlink_to(outside.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                clean_caches(root, ["scripts/__pycache__/temporary.pyc"])
            result = clean_caches(root)
            self.assertEqual(result["removed_directories"], ["scripts/__pycache__"])
            self.assertFalse((root / "scripts/__pycache__").exists())
            self.assertTrue((root / "benchmark/outputs/run/__pycache__/evidence").exists())
            self.assertTrue((root / ".venv/lib/__pycache__/dependency").exists())
            self.assertTrue((outside / "keep").exists())
