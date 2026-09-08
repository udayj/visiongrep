"""Portable JSON and self-contained HTML reports, including incomplete runs."""

import html
import json
from pathlib import Path

from .storage import read_json


def render(directory: Path) -> Path:
    report = read_json(directory / "report.json")
    rows = []
    for name, item in report.get("comparisons", {}).items():
        lo, hi = item["interval"]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{item['improvement']:.1%}</td>"
            f"<td>{lo:.1%} to {hi:.1%}</td><td>{item['pairs']}</td></tr>"
        )
    value = """<!doctype html><meta charset="utf-8"><title>VisionGrep benchmark</title>
<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#19232b}
table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}
pre{white-space:pre-wrap;background:#f4f6f8;padding:20px}h1{font-size:28px}</style>"""
    value += f"<h1>VisionGrep: {html.escape(report['verdict'])}</h1>"
    value += "<p>" + html.escape("; ".join(report.get("reasons", []))) + "</p>"
    estimates = report.get("timing_screen", {}).get("estimates", {})
    if estimates:
        value += "<h2>Local median uncertainty and batch stability</h2>"
        value += "<p>Approximate bootstrap intervals; local screening only. Raw CV is diagnostic.</p>"
        value += "<table><tr><th>Scenario / binary</th><th>Median ms</th><th>95% interval ms</th><th>Max batch deviation</th><th>Raw CV</th></tr>"
        for name, roles in estimates.items():
            for role, info in roles.items():
                lo, hi = info["median_interval_ms"]
                value += (
                    f"<tr><td>{html.escape(name)} / {html.escape(role)}</td>"
                    f"<td>{info['median_ms']:.3f}</td><td>{lo:.3f} to {hi:.3f}</td>"
                    f"<td>{info['max_relative_batch_deviation']:.1%}</td>"
                    f"<td>{info['raw_summaries'][0]['cv']:.1%}</td></tr>"
                )
        value += "</table>"
    value += (
        "<table><tr><th>Scenario</th><th>Improvement</th><th>Interval</th><th>Pairs</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    value += (
        "<details><summary>All metrics and provenance</summary><pre>"
        + html.escape(json.dumps(report, indent=2))
        + "</pre></details>"
    )
    path = directory / "report.html"
    path.write_text(value, encoding="utf-8")
    return path
