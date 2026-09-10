"""Build provenance prevents silent toolchain/configuration cache reuse."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import builds
from harness.runner import Run, contract
from harness.storage import FOUNDATION, digest, write_json


class Builds(unittest.TestCase):
    def test_inherited_build_overrides_are_removed(self):
        overrides = {
            key: "unwanted"
            for key in (
                "RUSTUP_TOOLCHAIN",
                "RUSTFLAGS",
                "RUSTC",
                "RUSTC_WRAPPER",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_PROFILE_RELEASE_LTO",
                "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER",
                "CC",
                "CFLAGS",
                "ORT_LIB_LOCATION",
                "SDKROOT",
                "MACOSX_DEPLOYMENT_TARGET",
            )
        }
        with patch.dict(os.environ, overrides):
            env = builds.build_environment(Path("isolated-cargo"))
        self.assertFalse(set(overrides) & env.keys())
        self.assertEqual(env["CARGO_HOME"], "isolated-cargo")

    def test_hidden_cargo_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".cargo").mkdir()
            (root / ".cargo/config.toml").write_text(
                '[build]\nrustflags = ["-Ctarget-cpu=native"]\n'
            )
            with self.assertRaisesRegex(ValueError, "Cargo config"):
                builds.reject_cargo_configs(root / "source", root / "cargo-home")

    def test_cache_reuse_requires_matching_compiler_linker_and_build_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "Cargo.toml").write_text('[package]\nname="fixture"\n')
            settings = {
                "target": "test-target",
                "command": ["build"],
                "rustc": "v1",
                "linker": "v1",
                "profile": "v1",
            }
            calls = []

            def invoke(args, *, cwd, env):
                calls.append(args)
                binary = (
                    Path(env["CARGO_TARGET_DIR"]) / "test-target/release/visiongrep"
                )
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"binary")

            def prepare(*args):
                return {}, Path("cargo"), settings.copy()

            with patch("harness.builds.prepare", side_effect=prepare):
                binary, _, _ = builds.build("commit", root / "cache", invoke, source)
                self.assertEqual(
                    builds.build("commit", root / "cache", invoke, source)[0], binary
                )
                self.assertEqual(len(calls), 1)
                for key in ("rustc", "linker", "profile"):
                    settings[key] = "v2"
                    updated, _, _ = builds.build(
                        "commit", root / "cache", invoke, source
                    )
                    self.assertNotEqual(binary, updated)
                    binary = updated
                self.assertEqual(len(calls), 4)
                binary.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "corrupt"):
                    builds.build("commit", root / "cache", invoke, source)

    def test_build_environment_change_rejects_foundation_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            write_json(corpus, {"images": []})
            binary = root / "visiongrep"
            binary.write_bytes(b"binary")
            profile = {"name": "local-quick", "cloud": False}
            config = {
                "directory": str(root / "run"),
                "profile": profile,
                "max_seconds": 60,
                "hourly_budget_usd": 0,
                "cache": str(root / "cache"),
                "corpus": str(corpus),
                "corpus_sha256": digest(corpus),
                "mode": "compare",
                "candidate": FOUNDATION,
                "baseline": str(root / "baseline.json"),
            }
            env = {
                key: "same"
                for key in (
                    "system",
                    "release",
                    "machine",
                    "cpu_count",
                    "python",
                    "ami",
                    "hardware",
                )
            }
            env["hardware"] = {"model name": "same"}
            with (
                patch("harness.runner.environment", return_value=env),
                patch("harness.runner.platform.system", return_value="Darwin"),
                patch("harness.runner.verify"),
                patch(
                    "harness.runner.build",
                    return_value=(binary, FOUNDATION, {"rustc": "new"}),
                ),
            ):
                old_contract = contract(config, profile, {}) | {
                    "build_environment": {"rustc": "old"}
                }
                write_json(root / "baseline.json", {"contract": old_contract})
                run = Run(config)
                with (
                    patch.object(run, "progress"),
                    self.assertRaisesRegex(ValueError, "recalibration"),
                ):
                    run.perform()
