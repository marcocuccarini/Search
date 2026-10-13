"""FR@5 bar plots (Figure fr5_all) with the four graph rankers: Base, BFS, NSP, PPR, GCS.

Query-aware graph, both edge types, configuration selected on validation (same as the
paper's tables). Reads revision_results/experiments/ (run_experiments.py) and
revision_results/similar_methods/ (similar_methods.py).

    python revision/plot_fr5.py
    python revision/plot_fr5.py --out acl-style-files-master/img

Writes Plot2Wiki, PlotHotPot, PlotMusi (.png and .pdf), the file names used in the paper.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "revision_results")
RET = ["BM25", "GTE-base", "GTE-large", "BGE-base", "BGE-large", "BGE-M3", "E5-base", "E5-large"]
LAB = ["BM25", "GTE-b", "GTE-l", "BGE-b", "BGE-l", "BGE-M3", "E5-b", "E5-l"]
# Base is a neutral gray; BFS orange and NSP green as in the paper's text; PPR blue, GCS pink
SERIES = [("Base", "#9a9a92"), ("BFS", "#eb6834"), ("NSP", "#1baf7a"), ("PPR", "#2a78d6"), ("GCS", "#e87ba4")]
FILES = {"2wiki": "Plot2Wiki", "hotpotqa": "PlotHotPot", "musique": "PlotMusi"}


def values(ds):
    vals = {k: [] for k, _ in SERIES}
    for ret in RET:
        E = {(r["variant"], r["mode"], r["type"]): float(r["FullRec@5"])
             for r in csv.DictReader(open(os.path.join(OUT, "experiments", f"{ds}_{ret}.csv")))}
        S = {(r["method"], r["type"]): float(r["FullRec@5"])
             for r in csv.DictReader(open(os.path.join(OUT, "similar_methods", f"{ds}_{ret}.csv")))}
        vals["Base"].append(E[("none", "none", "-")])
        vals["BFS"].append(E[("aware", "bfs", "both")])
        vals["NSP"].append(E[("aware", "ge", "both")])
        vals["PPR"].append(S[("ppr", "both")])
        vals["GCS"].append(S[("gcs", "both")])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "plots"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 7})
    for ds, fn in FILES.items():
        vals = values(ds)
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        x, w = np.arange(len(RET)), 0.16
        for i, (k, c) in enumerate(SERIES):
            ax.bar(x + (i - 2) * w, vals[k], w * 0.88, color=c, label=k, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(LAB, rotation=35, ha="right")
        ax.set_ylabel("FR@5")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#8a8a85")
            ax.spines[s].set_linewidth(0.6)
        ax.tick_params(colors="#444444", width=0.6, length=2)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False,
                  handlelength=1, columnspacing=0.8, handletextpad=0.4)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.out, f"{fn}.{ext}"), dpi=300)
        plt.close(fig)
        print(f"{ds:9s} mean FR@5 " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in vals.items()))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
