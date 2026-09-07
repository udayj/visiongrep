"""Scenario guards reject fast but incorrect image and mutation implementations."""

import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import behavior
from harness.scenarios import SCENARIOS


def vector(number):
    return struct.pack("<512f", *(float(i == number) for i in range(512)))


class Behavior(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        images = root / "images"
        images.mkdir()
        for i in range(4):
            (images / f"{i}.jpg").write_bytes(bytes([i]))
        self.scenario = SimpleNamespace(
            root=root,
            images=images,
            index=root / "index.db",
            rows=[{"file_name": f"{i}.jpg"} for i in range(4)],
            changed=1,
            name="novel_text",
        )
        with closing(sqlite3.connect(root / "expected.db")) as db, db:
            db.execute(
                "CREATE TABLE images (path BLOB PRIMARY KEY, mtime_ns INTEGER, size INTEGER, embedding BLOB)"
            )
            for i in range(4):
                stat = (images / f"{i}.jpg").stat()
                db.execute(
                    "INSERT INTO images VALUES (?, ?, ?, ?)",
                    (f"{i}.jpg".encode(), stat.st_mtime_ns, stat.st_size, vector(i)),
                )
        shutil.copyfile(root / "expected.db", self.scenario.index)

    def observation(self, image=None):
        names = sorted(p.name for p in self.scenario.images.iterdir() if p != image)
        return {
            "exit_code": 0,
            "results": [
                {"path": name, "score": 0.9 - i * 0.1} for i, name in enumerate(names)
            ],
        }

    def check(self, result=None, image=None, no_cache=False):
        return behavior.check(
            self.scenario, result or self.observation(image), image, no_cache, 10
        )

    def test_image_queries_reject_empty_results_and_self_matches(self):
        for name in (
            "external_image_first",
            "external_image_repeated",
            "indexed_image",
            "modified_query_image",
        ):
            self.scenario.name = name
            image = (
                self.scenario.images / "0.jpg"
                if name in ("indexed_image", "modified_query_image")
                else self.scenario.root / "external.jpg"
            )
            result = {"exit_code": 1, "results": []}
            self.assertFalse(self.check(result, image)["passed"], name)
        self.scenario.name = "indexed_image"
        image = self.scenario.images / "0.jpg"
        self.assertTrue(self.check(image=image)["passed"])
        self.assertFalse(self.check(self.observation(), image)["passed"])

    def test_every_mutation_rejects_stale_state_and_accepts_updated_state(self):
        for name in (
            "added_1pct",
            "deleted_1pct",
            "renamed_1pct",
            "modified_1pct",
            "modified_query_image",
        ):
            with self.subTest(name=name):
                # Reset using a fresh fixture so each mutation is independent.
                self.setUp()
                scenario = self.scenario
                scenario.name = name
                path = scenario.images / "0.jpg"
                image = path if name == "modified_query_image" else None
                with closing(sqlite3.connect(scenario.index)) as db, db:
                    if name == "added_1pct":
                        db.execute("DELETE FROM images WHERE path = ?", (b"0.jpg",))
                    elif name == "deleted_1pct":
                        path.unlink()
                    elif name == "renamed_1pct":
                        path.rename(path.with_name("renamed-0.jpg"))
                    else:
                        path.write_bytes(bytes([3]))
                        os.utime(path, ns=(1234567890, 1234567890))
                self.assertFalse(self.check(image=image)["passed"])
                with closing(sqlite3.connect(scenario.index)) as db, db:
                    db.execute("DELETE FROM images WHERE path = ?", (b"0.jpg",))
                    if name != "deleted_1pct":
                        if name == "renamed_1pct":
                            path = scenario.images / "renamed-0.jpg"
                        stat = path.stat()
                        db.execute(
                            "INSERT INTO images VALUES (?, ?, ?, ?)",
                            (
                                os.fsencode(path.name),
                                stat.st_mtime_ns,
                                stat.st_size,
                                vector(3 if name.startswith("modified") else 0),
                            ),
                        )
                self.assertTrue(
                    self.check(image=image)["passed"], self.check(image=image)
                )

    def test_updated_metadata_cannot_hide_stale_embeddings(self):
        self.scenario.name = "modified_1pct"
        path = self.scenario.images / "0.jpg"
        path.write_bytes(bytes([3]))
        with closing(sqlite3.connect(self.scenario.index)) as db, db:
            db.execute(
                "UPDATE images SET mtime_ns = ? WHERE path = ?",
                (path.stat().st_mtime_ns, b"0.jpg"),
            )
        self.assertIn(
            "index embeddings do not reflect the expected pixels",
            self.check()["reasons"],
        )

    def test_pair_comparison_checks_rankings_scores_and_all_embeddings(self):
        reference = self.observation()
        candidate_index = self.scenario.root / "candidate.db"
        shutil.copyfile(self.scenario.index, candidate_index)

        def compare(candidate):
            return behavior.compare(
                reference, candidate, self.scenario.index, candidate_index
            )

        self.assertTrue(compare(reference)["passed"])
        self.assertFalse(compare({"exit_code": 1, "results": []})["passed"])
        self.assertFalse(
            compare(reference | {"results": list(reversed(reference["results"]))})[
                "passed"
            ]
        )
        changed = [row | {"score": row["score"] + 0.01} for row in reference["results"]]
        self.assertFalse(compare(reference | {"results": changed})["passed"])
        with closing(sqlite3.connect(candidate_index)) as db, db:
            db.execute(
                "UPDATE images SET embedding = ? WHERE path = ?", (vector(1), b"0.jpg")
            )
        self.assertFalse(compare(reference)["passed"])

    def test_no_cache_rejects_persisted_index(self):
        self.assertFalse(self.check(no_cache=True)["passed"])
        self.scenario.index.unlink()
        self.assertTrue(self.check(no_cache=True)["passed"])

    def test_non_mutating_scenarios_accept_expected_index(self):
        for name in SCENARIOS:
            if name in ("modified_1pct", "modified_query_image"):
                continue
            self.scenario.name = name
            self.assertTrue(self.check()["passed"], name)
