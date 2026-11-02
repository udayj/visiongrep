"""Independent on-disk state for each scenario and binary; resets are outside timing."""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from .assets import stage_images, stage_models
from . import behavior
from .storage import BENCHMARKS, digest, read_json, write_json

SCENARIOS = (
    "index_absent",
    "reindex",
    "no_cache",
    "novel_text",
    "cached_text",
    "external_image_first",
    "external_image_repeated",
    "indexed_image",
    "modified_query_image",
    "added_1pct",
    "modified_1pct",
    "deleted_1pct",
    "renamed_1pct",
    "read_only",
    "persistent_text",
    "persistent_updates",
)
CALIBRATION = ("novel_text", "cached_text", "no_cache")


def checkpoint(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def warm_files(paths) -> None:
    for path in paths:
        with path.open("rb") as source:
            while source.read(1024 * 1024):
                pass


class Scenario:
    def __init__(
        self,
        root: Path,
        binary: Path,
        cache: Path,
        corpus: dict,
        name: str,
        invoke,
        *,
        graph_cache_role: str | None = None,
    ):
        self.root, self.binary, self.name, self.invoke = root, binary, name, invoke
        self.images = root / "images"
        self.index = root / "index.db"
        self.rows = corpus["images"]
        self.cache = cache
        self.count = len(self.rows)
        self.changed = max(1, math.ceil(self.count * 0.01))
        stage_images(cache, self.rows, self.images)
        self.environment = os.environ.copy()
        for key in list(self.environment):
            if key.startswith(("OMP_", "MKL_", "ORT_")) or key == "RAYON_NUM_THREADS":
                del self.environment[key]
        self.environment["XDG_CACHE_HOME"] = str(root / "cache")
        stage_models(cache, root / "cache")
        self.external = root / "external.jpg"
        shutil.copyfile(self.images / self.rows[0]["file_name"], self.external)
        self.text = "a bicycle near water"
        # Warmup builds fast artifact verification sidecars before any samples.
        warmup = self.execute("warmup", self.text, measured=False)
        checkpoint(self.index)
        shutil.copyfile(self.index, root / "expected.db")
        self.graph_cache_role = graph_cache_role
        self.graph_cache_identity = None
        self.graph_cache_metadata = None
        self.graph_cache_setup = None
        if graph_cache_role is not None:
            self.graph_cache_setup = self.prepare_graph_cache(warmup)
        if name == "added_1pct":
            for row in self.rows[: self.changed]:
                (self.images / row["file_name"]).unlink()
            self.execute("prepare-additions", self.text, measured=False)
        checkpoint(self.index)
        shutil.copyfile(self.index, root / "seed.db")
        self.original_stats = {
            row["file_name"]: (self.images / row["file_name"]).stat().st_mtime_ns
            for row in self.rows
            if (self.images / row["file_name"]).exists()
        }

    def optimized_graph_identity(self, *, include_content: bool) -> list[dict]:
        models = self.root / "cache/visiongrep/models"
        result = []
        for path in sorted((models / "optimized").glob("*.onnx")):
            identity = {
                "path": str(path.relative_to(models)),
                "device": path.stat().st_dev,
                "inode": path.stat().st_ino,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            if include_content:
                identity["sha256"] = digest(path)
            result.append(identity)
        return result

    def verify_graph_cache(self, *, include_content: bool = False) -> dict:
        actual = self.optimized_graph_identity(include_content=include_content)
        if self.graph_cache_role == "foundation":
            if actual:
                raise ValueError("source baseline unexpectedly produced optimized graphs")
            return {"source_graphs_absent": True}
        expected = (
            self.graph_cache_identity if include_content else self.graph_cache_metadata
        )
        if actual != expected:
            raise ValueError("candidate optimized graph identities or mtimes changed")
        return {
            "candidate_graphs_unchanged": True,
            "content_hashes_checked": include_content,
        }

    def prepare_graph_cache(self, warmup: dict) -> dict:
        session_ms = sum(
            phase["elapsed_ms"]
            for phase in warmup["timing"]["phases"]
            if phase["phase"] == "model_session_construction"
        )
        if self.graph_cache_role == "foundation":
            proof = self.verify_graph_cache()
            return {
                "role": "source_baseline",
                "setup_warmup_wall_ms": warmup["wall_ms"],
                "setup_model_session_construction_ms": session_ms,
                "artifact_bytes": 0,
                "artifacts": [],
                **proof,
            }
        if self.graph_cache_role != "candidate":
            raise ValueError(f"unknown graph-cache role: {self.graph_cache_role}")

        identity = self.optimized_graph_identity(include_content=True)
        if len(identity) != 2 or any(item["size"] <= 0 for item in identity):
            raise ValueError("candidate setup must produce two nonempty optimized graphs")
        self.graph_cache_identity = identity
        self.graph_cache_metadata = [
            {key: value for key, value in item.items() if key != "sha256"}
            for item in identity
        ]
        proofs = []
        for kind, options in (
            ("novel_text", {"query": self.text + " cache load proof"}),
            ("external_image", {"query": self.text, "image": self.external}),
        ):
            result = self.execute(
                f"graph-cache-{kind}-proof",
                options.pop("query"),
                measured=False,
                **options,
            )
            self.verify_graph_cache(include_content=True)
            proofs.append(
                {
                    "kind": kind,
                    "wall_ms": result["wall_ms"],
                    "exit_code": result["exit_code"],
                    "artifacts_unchanged": True,
                }
            )
        return {
            "role": "optimized_candidate",
            "setup_warmup_wall_ms": warmup["wall_ms"],
            "setup_model_session_construction_ms": session_ms,
            "setup_timing_scope": (
                "untimed seed search; the model-session phase includes source loading, "
                "optimization, and graph serialization"
            ),
            "artifact_bytes": sum(item["size"] for item in identity),
            "artifacts": identity,
            "cached_load_proofs": proofs,
        }

    def execute(
        self,
        label: str,
        query: str,
        *,
        measured: bool,
        image: Path | None = None,
        no_cache: bool = False,
        reindex: bool = False,
        top: int = 10,
    ):
        output = self.root / "observations" / label
        output.mkdir(parents=True)
        args = [str(self.binary)]
        args += ["--image", str(image)] if image else [query]
        args += [
            str(self.images),
            "--top",
            str(top),
            "--threshold",
            "-1",
            "--json",
            "--quiet",
            "--timing",
            "--timing-file",
            str(output / "phases.json"),
        ]
        args += ["--no-cache"] if no_cache else ["--index-path", str(self.index)]
        if reindex:
            args.append("--reindex")
        config = {
            "command": args,
            "stdout": str(output / "results.json"),
            "stderr": str(output / "stderr.log"),
        }
        if self.environment.get("BENCH_AMI"):
            config["storage_quiescence_path"] = str(self.root)
        write_json(output / "invocation.json", config)
        metrics = json.loads(
            self.invoke(
                [
                    sys.executable,
                    str(BENCHMARKS / "harness/measurement.py"),
                    str(output / "invocation.json"),
                ],
                env=self.environment,
            )
        )
        if metrics["exit_code"] not in (0, 1):
            raise RuntimeError(
                f"CLI failed in {label}: {(output / 'stderr.log').read_text()[-3000:]}"
            )
        metrics["timing"] = read_json(output / "phases.json")
        if any(
            p["phase"] == "artifact_download" and p["invocations"]
            for p in metrics["timing"]["phases"]
        ):
            raise ValueError(
                "benchmark invoked artifact download; inputs were not fully staged"
            )
        results = read_json(output / "results.json")
        if not isinstance(results, list):
            raise ValueError("CLI result is not an array")
        if metrics["exit_code"] != (0 if results else 1):
            raise ValueError("CLI exit code disagrees with its results")
        metrics["results"] = [
            {
                "path": str(Path(item["path"]).relative_to(self.images)),
                "score": item["score"],
            }
            for item in results
        ]
        if measured:
            metrics["behavior"] = behavior.check(self, metrics, image, no_cache, top)
            if self.index.exists():
                checkpoint(self.index)
                metrics["index_bytes"] = self.index.stat().st_size
                image_count = sum(path.is_file() for path in self.images.iterdir())
                metrics["index_bytes_per_image"] = metrics["index_bytes"] / image_count
                metrics["index_sidecar_bytes"] = sum(
                    p.stat().st_size
                    for p in self.root.glob("index.db-*")
                    if p.is_file()
                )
            if self.name in ("index_absent", "reindex", "no_cache"):
                metrics["images_per_second"] = self.count * 1000 / metrics["wall_ms"]
        return metrics

    def sample(self, index: int):
        name = self.name
        self.images.chmod(0o755)
        # SQLite root identity is preserved by restoring to the same image and index paths.
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(self.index) + suffix).unlink(missing_ok=True)
        if name not in ("index_absent", "no_cache"):
            shutil.copyfile(self.root / "seed.db", self.index)
        for row in self.rows[: self.changed]:
            path = self.images / row["file_name"]
            path.with_name("renamed-" + path.name).unlink(missing_ok=True)
            if not path.exists() or name in ("modified_1pct", "modified_query_image", "persistent_updates"):
                shutil.copyfile(self.cache / "objects" / row["sha256"], path)
            if row["file_name"] in self.original_stats:
                stamp = self.original_stats[row["file_name"]]
                os.utime(path, ns=(stamp, stamp))
        selected = self.images / self.rows[0]["file_name"]
        if name in ("modified_1pct", "modified_query_image"):
            # Replace pixels with another fixed source image, retaining the original path.
            targets = (
                self.rows[: self.changed] if name == "modified_1pct" else self.rows[:1]
            )
            for row in targets:
                replacement = self.rows[-1]
                shutil.copyfile(
                    self.cache / "objects" / replacement["sha256"],
                    self.images / row["file_name"],
                )
                stamp = self.original_stats[row["file_name"]] + 1_000_000_000
                os.utime(self.images / row["file_name"], ns=(stamp, stamp))
        if name in ("deleted_1pct", "renamed_1pct"):
            for row in self.rows[: self.changed]:
                path = self.images / row["file_name"]
                if name == "deleted_1pct":
                    path.unlink()
                else:
                    path.rename(path.with_name("renamed-" + path.name))
        image = None
        if name in ("indexed_image", "modified_query_image"):
            image = selected
        elif name.startswith("external_image"):
            image = self.external
        if name == "external_image_repeated":
            self.execute(
                f"external-prepare-{index}", self.text, measured=False, image=image
            )
        if name == "read_only":
            for path in self.images.iterdir():
                path.chmod(0o444)
            self.images.chmod(0o555)
        warm_files(p for p in self.images.iterdir() if p.is_file())
        if self.graph_cache_role == "candidate":
            warm_files(
                (self.root / "cache/visiongrep/models/optimized").glob("*.onnx")
            )
        else:
            warm_files((self.root / "cache/visiongrep/models").glob("*.onnx"))
        if self.index.exists():
            warm_files([self.index])
        if name.startswith("persistent_"):
            from .persistent import sample

            result = sample(self, index)
            if self.graph_cache_role is not None:
                result["graph_cache_validation"] = self.verify_graph_cache()
            return result
        query = (
            f"a novel query {index}: bicycle near water"
            if name == "novel_text"
            else self.text
        )
        result = self.execute(
            f"sample-{index:04d}",
            query,
            measured=True,
            image=image,
            no_cache=name == "no_cache",
            reindex=name == "reindex",
        )
        if self.graph_cache_role is not None:
            result["graph_cache_validation"] = self.verify_graph_cache()
        return result

    def cleanup(self):
        self.images.chmod(0o755)
        for path in self.images.iterdir():
            path.chmod(0o644)
