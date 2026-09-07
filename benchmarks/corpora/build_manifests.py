#!/usr/bin/env python3
"""Freeze COCO selections and leave-category-out query galleries from official annotations."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def stable(row):
    return hashlib.sha256(f"visiongrep-coco-v1:{row['id']}".encode()).digest()


def write(path, value):
    # One image/query per line keeps large frozen manifests reviewable in Git.
    fields = []
    for key, item in sorted(value.items()):
        if key in ("images", "queries"):
            encoded = (
                "[\n"
                + ",\n".join("    " + json.dumps(row, sort_keys=True) for row in item)
                + "\n  ]"
            )
        else:
            encoded = json.dumps(item, sort_keys=True)
        fields.append("  " + json.dumps(key) + ": " + encoded)
    path.write_text("{\n" + ",\n".join(fields) + "\n}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    with zipfile.ZipFile(args.annotations) as archive:
        instances = json.loads(archive.read("annotations/instances_val2017.json"))
        captions = json.loads(archive.read("annotations/captions_val2017.json"))
        train = json.loads(archive.read("annotations/captions_train2017.json"))
    labels = {row["id"]: set() for row in instances["images"]}
    for row in instances["annotations"]:
        labels[row["image_id"]].add(row["category_id"])
    ordered = sorted(instances["images"], key=stable)
    selected = {}
    # Ensure every concept has present examples; selection never uses model scores.
    for category in instances["categories"]:
        matches = [row for row in ordered if category["id"] in labels[row["id"]]]
        for row in matches[:3]:
            selected[row["id"]] = row
    for row in ordered:
        if len(selected) >= 500:
            break
        selected[row["id"]] = row
    small = sorted(selected.values(), key=stable)
    large = small + [row for row in ordered if row["id"] not in selected]
    large += sorted(train["images"], key=stable)[: 10000 - len(large)]
    by_caption = {}
    for row in captions["annotations"]:
        by_caption.setdefault(row["image_id"], []).append(row)
    licenses = {row["id"]: row for row in instances["licenses"]}
    archive_hash = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    source = {
        "url": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "sha256": archive_hash,
        "annotation_license": "CC BY 4.0",
    }

    def image(row):
        split = "val2017" if row["id"] in labels else "train2017"
        return {
            "id": row["id"],
            "file_name": row["file_name"],
            "url": f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{row['file_name']}",
            "width": row["width"],
            "height": row["height"],
            "license": licenses[row["license"]],
            "category_ids": sorted(labels.get(row["id"], [])),
        }

    for count, rows in [(500, small), (10000, large)]:
        write(
            args.output / f"coco-{count}.json",
            {
                "schema_version": 1,
                "name": f"coco-{count}-v1",
                "source": source,
                "selection": "SHA256 order; local subset covers all 80 categories",
                "images": [image(row) for row in rows],
            },
        )
    queries = []
    for row in small[:200]:
        caption = min(by_caption[row["id"]], key=lambda item: item["id"])
        queries.append(
            {
                "id": f"caption-{caption['id']}",
                "intent": f"caption-{row['id']}",
                "query": caption["caption"],
                "relevant": [row["file_name"]],
                "kind": "caption",
            }
        )
    for category in sorted(instances["categories"], key=lambda item: item["id"]):
        relevant = [
            row["file_name"] for row in small if category["id"] in labels[row["id"]]
        ]
        for index, template in enumerate(
            ["a photo containing {}", "an image showing {}", "find {} in the picture"]
        ):
            queries.append(
                {
                    "id": f"category-{category['id']}-{index}",
                    "intent": f"category-{category['id']}",
                    "query": template.format(category["name"]),
                    "kind": "category",
                    "relevant": relevant,
                    "absent_gallery": "all corpus images except relevant",
                    "category": category["name"],
                    "category_id": category["id"],
                }
            )
    write(
        args.output / "quality-500.json",
        {
            "schema_version": 1,
            "source": source,
            "corpus": "coco-500-v1",
            "threshold": 0.25,
            "category_intents": 80,
            "absence_evidence": "COCO instance annotations; not exhaustively human-reviewed semantic absence",
            "review_status": "annotation-defined",
            "queries": queries,
        },
    )
    print(
        f"Frozen {len(small)} / {len(large)} images; {len(queries)} queries, 80 absent intents"
    )


if __name__ == "__main__":
    main()
