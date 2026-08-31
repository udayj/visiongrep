#!/usr/bin/env python3
"""Materialize a pinned, non-redistributed COCO Caption 2017 benchmark corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "lmms-lab-encoder/COCO-Caption2017"
REVISION = "3bdd5827e243cc3084ac69a1111e69c3ab9193ff"
SPLIT = "val"
ROWS_PER_REQUEST = 100
LICENSES = {
    1: ("Attribution-NonCommercial-ShareAlike License", "http://creativecommons.org/licenses/by-nc-sa/2.0/"),
    2: ("Attribution-NonCommercial License", "http://creativecommons.org/licenses/by-nc/2.0/"),
    3: ("Attribution-NonCommercial-NoDerivs License", "http://creativecommons.org/licenses/by-nc-nd/2.0/"),
    4: ("Attribution License", "http://creativecommons.org/licenses/by/2.0/"),
    5: ("Attribution-ShareAlike License", "http://creativecommons.org/licenses/by-sa/2.0/"),
    6: ("Attribution-NoDerivs License", "http://creativecommons.org/licenses/by-nd/2.0/"),
    7: ("No known copyright restrictions", "http://flickr.com/commons/usage/"),
    8: ("United States Government Work", "http://www.usa.gov/copyright.shtml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", default=500, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="write the pinned row metadata without downloading image bytes",
    )
    return parser.parse_args()


def retry_delays() -> tuple[int, ...]:
    return (0, 1, 2, 4, 8)


def get_json(url: str) -> dict[str, Any]:
    for attempt, delay in enumerate(retry_delays()):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, urllib.error.HTTPError) as error:
            if attempt + 1 == len(retry_delays()):
                raise RuntimeError(f"failed to fetch {url} after bounded retries") from error
    raise AssertionError("retry loop exhausted without returning or raising")


def require_revision() -> None:
    metadata = get_json(f"https://huggingface.co/api/datasets/{DATASET}")
    if metadata.get("sha") != REVISION:
        raise RuntimeError(
            f"dataset revision changed: expected {REVISION}, got {metadata.get('sha')}"
        )


def fetch_rows(count: int) -> list[dict[str, Any]]:
    rows = []
    for offset in range(0, count, ROWS_PER_REQUEST):
        length = min(ROWS_PER_REQUEST, count - offset)
        parameters = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": SPLIT,
                "offset": offset,
                "length": length,
                "revision": REVISION,
            }
        )
        response = get_json(f"https://datasets-server.huggingface.co/rows?{parameters}")
        for wrapped in response["rows"]:
            row = wrapped["row"]
            license_name, license_url = LICENSES[row["license"]]
            rows.append(
                {
                    "row": wrapped["row_idx"],
                    "file_name": row["file_name"],
                    "image_url": row["image"]["src"],
                    "coco_url": row["coco_url"],
                    "captions": row["answer"],
                    "license_id": row["license"],
                    "license_name": license_name,
                    "license_url": license_url,
                    "width": row["width"],
                    "height": row["height"],
                }
            )
    if len(rows) != count:
        raise RuntimeError(f"expected {count} rows, received {len(rows)}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_image(row: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    destination = image_dir / row["file_name"]
    if not destination.is_file():
        for attempt, delay in enumerate(retry_delays()):
            if delay:
                time.sleep(delay)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=image_dir
            )
            try:
                with os.fdopen(file_descriptor, "wb") as output:
                    with urllib.request.urlopen(row["image_url"], timeout=120) as response:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, destination)
                break
            except (OSError, urllib.error.HTTPError) as error:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                if attempt + 1 == len(retry_delays()):
                    raise RuntimeError(
                        f"failed to download row {row['row']} after bounded retries"
                    ) from error
    return {
        key: value for key, value in row.items() if key != "image_url"
    } | {"size": destination.stat().st_size, "sha256": sha256(destination)}


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.count > 5000:
        raise ValueError("--count must be between 1 and 5000")
    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")

    require_revision()
    rows = fetch_rows(args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        corpus = [{key: value for key, value in row.items() if key != "image_url"} for row in rows]
    else:
        image_dir = args.output / "images"
        image_dir.mkdir(exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            corpus = list(executor.map(lambda row: download_image(row, image_dir), rows))

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": SPLIT,
        "row_start": 0,
        "row_count": args.count,
        "images_downloaded": not args.metadata_only,
        "annotation_license": "CC BY 4.0",
        "annotation_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "images": corpus,
    }
    (args.output / "corpus.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
