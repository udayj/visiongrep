#!/usr/bin/env python3
"""Generate small CC0 benchmark fixtures for screenshot and text-heavy retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def editor_fixture(path: Path) -> None:
    image = Image.new("RGB", (640, 400), "#1e1e1e")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 639, 31), fill="#323233")
    draw.text((14, 10), "main.rs - VisionGrep", fill="#eeeeee")
    lines = [
        "fn main() {",
        "    let query = \"red bicycle\";",
        "    println!(\"search: {query}\");",
        "}",
    ]
    colors = ["#dcdcaa", "#9cdcfe", "#ce9178", "#d7ba7d"]
    for index, line in enumerate(lines):
        draw.text((48, 74 + index * 34), line, fill=colors[index])
        draw.text((12, 74 + index * 34), str(index + 1), fill="#858585")
    image.save(path, format="PNG", optimize=False)


def settings_fixture(path: Path) -> None:
    image = Image.new("RGB", (640, 400), "#f2f2f2")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 639, 54), fill="#2d6cdf")
    draw.text((24, 20), "Application Settings", fill="white")
    draw.text((48, 91), "Privacy", fill="#222222")
    draw.text((48, 132), "Keep image search on this device", fill="#333333")
    draw.rounded_rectangle((500, 121, 564, 147), radius=13, fill="#2d6cdf")
    draw.ellipse((540, 124, 560, 144), fill="white")
    draw.text((48, 190), "Index location", fill="#333333")
    draw.rectangle((48, 218, 580, 258), outline="#999999", width=2)
    draw.text((62, 233), "/Volumes/Photos/search.db", fill="#555555")
    image.save(path, format="PNG", optimize=False)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("synthetic-rust-editor.png", editor_fixture),
        ("synthetic-settings-dialog.png", settings_fixture),
    ]
    report = []
    for name, generate in fixtures:
        path = args.output / name
        generate(path)
        report.append(
            {
                "file_name": name,
                "sha256": sha256(path),
                "license": "CC0-1.0",
                "generator": "Pillow 12.3.0",
            }
        )
    print(json.dumps({"schema_version": 1, "fixtures": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
