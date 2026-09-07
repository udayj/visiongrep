"""Prepare checksum-addressed model/image caches; measured runs never download."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .storage import BENCHMARKS, digest, outside_repository, read_json, write_json


def fetch(url: str, destination: Path, expected: str) -> None:
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"cached artifact has wrong checksum: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, output)
        if digest(Path(name)) != expected:
            raise ValueError(f"download checksum mismatch: {url}")
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def prepare(cache: Path, corpus: Path, import_models: Path | None = None) -> None:
    cache = outside_repository(cache)
    for artifact in read_json(BENCHMARKS / "profiles/artifacts.json"):
        destination = cache / "objects" / artifact["sha256"]
        source = import_models / artifact["name"] if import_models else None
        if source and source.is_file() and not destination.exists():
            if digest(source) != artifact["sha256"]:
                raise ValueError(f"incorrect model: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        fetch(artifact["url"], destination, artifact["sha256"])
    manifest = read_json(corpus)
    lock_path = cache / "corpora" / (digest(corpus) + ".json")
    if lock_path.exists():
        manifest = read_json(lock_path)
    known_images = read_json(BENCHMARKS / "corpora/image-checksums.json")

    def acquire(row):
        if row["file_name"] in known_images:
            known = known_images[row["file_name"]]
            if "sha256" in row and row["sha256"] != known["sha256"]:
                raise ValueError(
                    "prepared image differs from the pinned source checksum"
                )
            row = row | known
        receipt = (
            cache
            / "downloads"
            / (hashlib.sha256(row["url"].encode()).hexdigest() + ".json")
        )
        if "sha256" not in row and receipt.exists():
            row = row | read_json(receipt)
        if "sha256" in row:
            fetch(row["url"], cache / "objects" / row["sha256"], row["sha256"])
            return row
        # First preparation freezes source bytes; subsequent preparations require this lock.
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            path = Path(temporary) / "image"
            with (
                urllib.request.urlopen(row["url"], timeout=120) as response,
                path.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            checksum = digest(path)
            destination = cache / "objects" / checksum
            if not destination.exists():
                path.replace(destination)
            identity = {"sha256": checksum, "size": destination.stat().st_size}
            write_json(receipt, identity)
            return row | identity

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        manifest["images"] = list(pool.map(acquire, manifest["images"]))
    manifest["selection_sha256"] = digest(corpus)
    write_json(lock_path, manifest)


def verify(cache: Path, corpus: dict) -> None:
    for row in [*read_json(BENCHMARKS / "profiles/artifacts.json"), *corpus["images"]]:
        path = cache / "objects" / row["sha256"]
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"missing/corrupt artifact {path}; run prepare first")


def stage_models(cache: Path, destination: Path) -> None:
    models = destination / "visiongrep/models"
    models.mkdir(parents=True)
    for row in read_json(BENCHMARKS / "profiles/artifacts.json"):
        # Copies isolate candidate writes and verification sidecars from shared cache objects.
        shutil.copyfile(cache / "objects" / row["sha256"], models / row["name"])


def stage_images(cache: Path, rows: list[dict], destination: Path) -> None:
    destination.mkdir(parents=True)
    for row in rows:
        name = row["file_name"]
        if Path(name).name != name or name.startswith("."):
            raise ValueError(f"invalid corpus filename: {name}")
        shutil.copyfile(cache / "objects" / row["sha256"], destination / name)
