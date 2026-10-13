"""Read best hyperparameters and reported numbers from the paper's appendix tables
(labels tab:2wiki, tab:Hot, tab:Mu), so experiments use exactly the published
configurations and the reproduction check can compare against them."""
from __future__ import annotations

import re

TABLE_LABELS = {"2wiki": "tab:2wiki", "hotpotqa": "tab:Hot", "musique": "tab:Mu"}
METRIC_COLS = ["R@3", "R@5", "FR@3", "FR@5"]


def _params(cell):
    p = {}
    for key, pat, cast in (("h", r"\bh:\s*(\d+)", int), ("d", r"\bd:\s*(\d+)", int),
                           ("b", r"\bb:\s*(\d+)", int), ("hops", r"hops:\s*(\d+)", int),
                           ("alpha", r"alpha\$?\s*:\s*([0-9.]+)", float)):
        m = re.search(pat, cell)
        if m:
            p[key] = cast(m.group(1))
    return p


def parse_tables(tex_path):
    """-> {dataset: [ {retriever, mode, type, params, R@3, R@5, FR@3, FR@5}, ... ]}"""
    text = open(tex_path, encoding="utf-8").read()
    out = {}
    for ds, label in TABLE_LABELS.items():
        end = text.find("\\label{%s}" % label)
        if end == -1:
            continue
        start = text.rfind("\\begin{tabular}", 0, end)
        rows = []
        for line in text[start:end].splitlines():
            line = line.strip()
            if "&" not in line or not line.endswith("\\\\") or line.startswith("\\textbf"):
                continue
            cells = [c.strip() for c in line[:-2].split("&")]
            if len(cells) != 8:
                continue
            try:
                vals = [float(c) for c in cells[4:]]
            except ValueError:
                continue
            rows.append({"retriever": cells[0], "mode": cells[1],
                         "type": None if cells[2] == "-" else cells[2],
                         "params": _params(cells[3]),
                         **dict(zip(METRIC_COLS, vals))})
        out[ds] = rows
    return out


def find_row(rows, retriever, mode, etype=None):
    for r in rows:
        if r["retriever"] == retriever and r["mode"] == mode and \
                (mode == "Baseline" or r["type"] == etype):
            return r
    return None
