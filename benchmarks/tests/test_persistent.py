"""Exercise real JSONL pipes and mutations without model inference."""

import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.persistent import exchange, run
from harness import behavior
from harness.storage import BENCHMARKS


class PersistentSequences(unittest.TestCase):
    def test_comparison_checks_intermediate_responses(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent.db"
            reference = {"results": [], "exit_code": 0, "requests": [
                {"id": 0, "results": [{"path": "a.jpg", "score": 0.5}]},
                {"id": 1, "results": []},
            ]}
            candidate = reference | {"requests": [
                {"id": 0, "results": [{"path": "b.jpg", "score": 0.5}]},
                {"id": 1, "results": []},
            ]}
            self.assertFalse(behavior.compare(reference, candidate, missing, missing)["passed"])

    def test_measurement_helper_imports_package_and_reports_child_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            for name in ("index.db", "expected.db"):
                with closing(sqlite3.connect(root / name)) as db, db:
                    db.execute("CREATE TABLE images (path BLOB, embedding BLOB, mtime_ns INTEGER, size INTEGER)")
            script = "import json,sys\nfor line in sys.stdin:\n print(json.dumps({'id':json.loads(line)['id'],'results':[]}),flush=True)\n"
            config = {
                "command": [sys.executable, "-c", script],
                "stdout": str(root / "responses.jsonl"), "stderr": str(root / "stderr"),
                "persistent": {"root": str(root), "cache": str(root), "rows": [],
                               "name": "persistent_text", "changed": 0,
                               "original_stats": {}, "query": "query"},
            }
            path = root / "invocation.json"
            path.write_text(json.dumps(config))
            metrics = json.loads(subprocess.check_output(
                [sys.executable, str(BENCHMARKS / "harness/measurement.py"), str(path)],
                text=True, timeout=10,
            ))
            self.assertTrue(metrics["behavior"]["passed"])
            self.assertGreater(metrics["peak_rss_bytes"], 0)
            self.assertEqual(len(metrics["requests"]), 5)

    def test_requests_share_process_and_mutations_precede_refresh(self):
        for name in ("persistent_text", "persistent_updates"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "images").mkdir()
                (root / "objects").mkdir()
                rows = [{"file_name": f"{i}.jpg", "sha256": str(i)} for i in range(2)]
                for i in range(2):
                    (root / "images" / f"{i}.jpg").write_bytes(bytes([i]))
                    (root / "objects" / str(i)).write_bytes(bytes([i]))
                script = root / "server.py"
                script.write_text('''import json, sys
for number, line in enumerate(sys.stdin):
    request = json.loads(line)
    assert request['id'] == number
    print(json.dumps({'id': number, 'results': []}), flush=True)
''')
                pixels = []

                def check(*args):
                    pixels.append((root / "images/0.jpg").read_bytes())
                    return {"passed": True, "reasons": []}

                config = {"command": [sys.executable, str(script)], "persistent": {
                    "root": str(root), "cache": str(root), "rows": rows,
                    "name": name, "changed": 1, "original_stats": {"0.jpg": 1},
                    "query": "query",
                }}
                with patch("harness.persistent.behavior.check", side_effect=check):
                    output = io.BytesIO()
                    with (root / "stderr").open("wb") as err:
                        metrics = run(config, output, err)
                self.assertEqual(metrics["exit_code"], 0)
                self.assertTrue(metrics["behavior"]["passed"])
                self.assertEqual(len(output.getvalue().splitlines()), 5)
                self.assertGreater(metrics["first_response_ms"], 0)
                expected_pixels = [bytes([0])] * 5 if name == "persistent_text" else [
                    bytes([i]) for i in (0, 1, 0, 1, 1)
                ]
                self.assertEqual(pixels, expected_pixels)
                times = [row["wall_ms"] for row in metrics["requests"]]
                expected = sorted(times[1:4])[1] if name == "persistent_text" else sum(times[2:4]) / 2
                self.assertEqual(metrics["wall_ms"], expected)

    def test_wrong_id_error_and_early_eof_are_rejected(self):
        for response in ({"id": 3, "results": []}, {"id": 0, "error": "bad"}, None):
            script = "import sys; sys.stdin.readline(); "
            if response is not None:
                script += f"print({json.dumps(response)!r}, flush=True)"
            with subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, bufsize=0) as child:
                with self.assertRaises(ValueError):
                    exchange(child, {"id": 0, "query": "query"})
                child.stdin.close()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
