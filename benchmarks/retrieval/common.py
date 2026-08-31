"""Shared manifest handling for the isolated retrieval benchmark tools."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

SUPPORTED_CORPUS_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(corpus_path: Path, query_path: Path) -> list[dict[str, Any]]:
    corpus = read_json(corpus_path)
    query_manifest = read_json(query_path)
    expected = query_manifest["corpus"]
    for key in ("dataset", "dataset_revision", "split", "row_start", "row_count"):
        if corpus[key] != expected[key]:
            raise RuntimeError(
                f"corpus/query manifest mismatch for {key}: {corpus[key]!r} != {expected[key]!r}"
            )

    rows = {image["row"]: image for image in corpus["images"]}
    generated = query_manifest["generated_positive_queries"]
    start = generated["rows"]["start"]
    count = generated["rows"]["count"]
    caption_index = generated["caption_index"]
    queries = []
    for row_index in range(start, start + count):
        image = rows[row_index]
        queries.append(
            {
                "id": f"{generated['id_prefix']}-{row_index:06d}-{caption_index}",
                "query": image["captions"][caption_index],
                "relevant": [image["file_name"]],
                "tags": generated["tags"],
                "kind": "positive",
            }
        )
    for query in query_manifest["curated_positive_queries"]:
        queries.append(query | {"kind": "positive"})
    for query in query_manifest["absent_queries"]:
        queries.append(query | {"kind": "absent", "relevant": []})
    return queries


def environment_metadata() -> dict[str, Any]:
    total_memory = None
    try:
        total_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        pass
    packages = {}
    for distribution in ("numpy", "onnxruntime", "Pillow", "coremltools"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory,
        "python": platform.python_version(),
        "packages": packages,
    }


def corpus_image_count(path: Path) -> int:
    return sum(
        entry.is_file() and entry.suffix.lower() in SUPPORTED_CORPUS_SUFFIXES
        for entry in path.iterdir()
    )


def peak_rss_kib(raw_value: int) -> int:
    return raw_value // 1024 if sys.platform == "darwin" else raw_value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
