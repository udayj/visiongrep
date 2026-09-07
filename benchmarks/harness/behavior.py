"""Untimed checks of scenario results and persisted image state."""

import math
import os
import sqlite3
from contextlib import closing

from .quality import embeddings


def vectors_equal(left, right):
    return (
        len(left) == len(right) == 512
        and all(
            math.isfinite(x) and math.isfinite(y) and abs(x - y) <= 1e-4
            for x, y in zip(left, right)
        )
        and all(
            abs(math.sqrt(sum(x * x for x in v)) - 1) <= 0.001 for v in (left, right)
        )
    )


def check(scenario, observation, image, no_cache, top):
    reasons = []
    paths = {p.name: p for p in scenario.images.iterdir() if p.is_file()}
    eligible = set(paths)
    if image is not None and image.parent == scenario.images:
        eligible.discard(image.name)
    results = observation["results"]
    names = [row["path"] for row in results]
    if (
        len(names) != min(top, len(eligible))
        or len(set(names)) != len(names)
        or not set(names) <= eligible
        or any(not math.isfinite(row["score"]) for row in results)
    ):
        reasons.append("incorrect result count, membership, or scores")
    if no_cache:
        if scenario.index.exists():
            reasons.append("no-cache invocation persisted an index")
    elif not scenario.index.is_file():
        reasons.append("index missing")
    else:
        actual = embeddings(scenario.index)
        expected = embeddings(scenario.root / "expected.db")
        if actual.keys() != {os.fsencode(name) for name in paths}:
            reasons.append("index membership does not match the mutated corpus")
        else:
            modified = set()
            if scenario.name in ("modified_1pct", "modified_query_image"):
                count = scenario.changed if scenario.name == "modified_1pct" else 1
                modified = {row["file_name"] for row in scenario.rows[:count]}
            for name in paths:
                source = name.removeprefix("renamed-")
                if name in modified:
                    source = scenario.rows[-1]["file_name"]
                if os.fsencode(source) not in expected or not vectors_equal(
                    actual[os.fsencode(name)], expected[os.fsencode(source)]
                ):
                    reasons.append(
                        "index embeddings do not reflect the expected pixels"
                    )
                    break
        with closing(sqlite3.connect(f"file:{scenario.index}?mode=ro", uri=True)) as db:
            for key, stamp, size in db.execute(
                "SELECT path, mtime_ns, size FROM images"
            ):
                path = paths.get(os.fsdecode(key))
                if path is None or (stamp, size) != (
                    path.stat().st_mtime_ns,
                    path.stat().st_size,
                ):
                    reasons.append("index file metadata is stale")
                    break
    return {"passed": not reasons, "reasons": reasons}


def compare(reference, candidate, reference_index, candidate_index):
    reasons = []
    left, right = reference["results"], candidate["results"]
    if reference["exit_code"] != candidate["exit_code"]:
        reasons.append("scenario exit codes differ")
    if [row["path"] for row in left] != [row["path"] for row in right]:
        reasons.append("scenario rankings differ")
    elif any(
        not math.isfinite(b["score"]) or abs(a["score"] - b["score"]) > 1e-4
        for a, b in zip(left, right)
    ):
        reasons.append("scenario scores differ beyond tolerance")
    if reference_index.exists() != candidate_index.exists():
        reasons.append("scenario index presence differs")
    elif reference_index.exists():
        a, b = embeddings(reference_index), embeddings(candidate_index)
        if a.keys() != b.keys() or any(not vectors_equal(a[key], b[key]) for key in a):
            reasons.append("scenario index membership or embeddings differ")
    return {"passed": not reasons, "reasons": reasons}
