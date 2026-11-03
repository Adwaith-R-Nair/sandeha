"""Draw every figure from the saved results. A result that wasn't run is said to be missing.

    uv run python scripts/make_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
ACCURACY_POLICIES = {
    "ensemble": ("Ensemble, no hand-off", SLOTS[0]),
    "single_model": ("Single model", SLOTS[1]),
    "persistence": ("Persistence", SLOTS[3]),
    "coarse_solver": ("Coarse solver", SLOTS[2]),
    "solver_only": ("Solver only", SLOTS[6]),
}
HANDOFF_POLICIES = {
    "doubt": ("Doubt-based", SLOTS[0]),
    "fixed": ("Fixed schedule", SLOTS[1]),
    "random": ("Random", SLOTS[2]),
    "oracle": ("Oracle", SLOTS[3]),
}
SIGNALS = {
    "chosen": ("Ensemble doubt", SLOTS[0]),
    "no_ensemble": ("Single-model divergence", SLOTS[1]),
    "independent_secondary": ("Independent rollouts (secondary)", SLOTS[2]),
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _labelled_groups(result: dict) -> list[tuple[str, dict]]:
    found = []
    for split, measured in result["sets"].items():
        for group, values in measured["groups"].items():
            found.append((split if group == "all" else f"{split} {group}", values))
    return found


def _save(figure, out: Path, name: str) -> None:
    figure.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(out / f"{name}.png")


def _panels(count: int, height: float = 2.8, width: float = 3.4, sharey: bool = True, sharex: bool = True):
    columns = min(count, 3) or 1
    rows = (count + columns - 1) // columns or 1
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), sharex=sharex, sharey=sharey, squeeze=False)
    flat = list(axes.ravel())
    for extra in flat[count:]:
        extra.set_visible(False)
    return figure, flat


def _legend(figure, handles, title: str) -> None:
    figure.suptitle(title, fontsize=11, x=0.01, ha="left", color=INK)
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.0),
        ncols=min(len(handles), 4),
    )
    figure.tight_layout()


def _dots(ax, rows: list[tuple[str, dict]], names: dict) -> list:
    """Each row a group, with a dot and interval for each named figure, nudged so that none overlap."""
    handles = {}
    for position, (_, values) in enumerate(rows):
        for offset, (name, (label, colour)) in enumerate(names.items()):
            item = values.get(name)
            if not item or item[0] is None:
                continue
            y = position + (offset - (len(names) - 1) / 2) * 0.18
            low, high = item[1] if item[1] else (item[0], item[0])
            ax.plot([low, high], [y, y], color=colour, solid_capstyle="round", linewidth=2)
            handles[name] = ax.plot(item[0], y, "o", color=colour, markersize=6, markeredgecolor=SURFACE, label=label)[0]
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    return list(handles.values())


def accuracy_figure(result: dict, out: Path) -> None:
    groups = _labelled_groups(result)
    figure, axes = _panels(len(groups))
    handles = {}
    for ax, (label, values) in zip(axes, groups, strict=False):
        for name, (title, colour) in ACCURACY_POLICIES.items():
            curve = values["policies"][name]["error_by_time"]
            handles[name] = ax.plot(range(1, len(curve) + 1), curve, color=colour, label=title)[0]
        ax.set_title(label, loc="left")
        ax.set_xlabel("Rollout time (frames)")
        ax.set_ylabel("Per-frame error")
    _legend(figure, list(handles.values()), "Error along the rollout, mean over scenes (capped at 10)")
    _save(figure, out, "accuracy_error_by_time")


def doubt_evidence_figure(result: dict, out: Path) -> None:
    rows = []
    for label, values in _labelled_groups(result):
        window = values["primary_window"]
        chosen = window["aggregations"][values["aggregation"]]["evidence"]
        rows.append(
            (
                label,
                {
                    "chosen": (chosen["value"], chosen["ci95"]),
                    "no_ensemble": (window["no_ensemble"]["evidence"]["value"], window["no_ensemble"]["evidence"]["ci95"]),
                    "independent_secondary": (
                        window["independent_secondary"]["evidence"]["value"],
                        window["independent_secondary"]["evidence"]["ci95"],
                    ),
                },
            )
        )
    figure, ax = plt.subplots(figsize=(6.5, 0.55 * len(rows) + 1.2))
    handles = _dots(ax, rows, SIGNALS)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Within-time Spearman correlation with error, 95% interval")
    _legend(figure, handles, "Does the signal rank frames by error at a fixed rollout time?")
    _save(figure, out, "doubt_evidence")


def _lines_by_group(result: dict, out: Path, name: str, title: str, xlabel: str, ylabel: str, read) -> None:
    groups = _labelled_groups(result)
    figure, ax = plt.subplots(figsize=(6.5, 3.6))
    handles = []
    for i, (label, values) in enumerate(groups):
        x, y = read(values)
        points = [(a, b) for a, b in zip(x, y, strict=True) if b is not None]
        if points:
            handles.append(ax.plot(*zip(*points, strict=True), color=SLOTS[i % len(SLOTS)], marker="o", markersize=4, label=label)[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _legend(figure, handles, title)
    _save(figure, out, name)


def doubt_figures(result: dict, out: Path) -> None:
    doubt_evidence_figure(result, out)

    def chosen(values):
        return values["primary_window"]["aggregations"][values["aggregation"]]

    def calibration(values):
        bins = chosen(values)["calibration_not_evidence"]
        return range(1, len(bins) + 1), [item["mean_error"] for item in bins]

    def coverage(values):
        curve = chosen(values)["risk_coverage_not_evidence"]["within_time"]
        return curve["achieved"], curve["mean_error"]

    def median(values):
        series = values["median_doubt_by_time"]
        return range(1, len(series) + 1), series

    _lines_by_group(
        result, out, "doubt_calibration", "Calibration: error by frozen doubt bin (not evidence by itself)",
        "Doubt bin, frozen on validation", "Mean per-frame error", calibration,
    )
    _lines_by_group(
        result, out, "doubt_risk_coverage", "Risk-coverage within rollout times (not evidence by itself)",
        "Achieved coverage", "Mean per-frame error of frames kept", coverage,
    )
    _lines_by_group(
        result, out, "doubt_by_time", "Median doubt along the rollout",
        "Rollout time (frames)", "Median doubt", median,
    )


def handoff_figures(result: dict, out: Path) -> None:
    rows = []
    for label, values in _labelled_groups(result):
        row = {"doubt": (values["frozen"]["set_score"], values["frozen"]["set_score_ci"])}
        for kind in ("fixed", "random"):
            runs = values[kind]
            if runs:
                nearest = min(runs, key=lambda run: abs(run["achieved_share"] - values["budget"]))
                row[kind] = (nearest["set_score"], nearest["set_score_ci"])
        if values["oracle"].get("run"):
            row["oracle"] = (values["oracle"]["set_score"], values["oracle"]["set_score_ci"])
        rows.append((f"{label} ({100 * values['budget']:.1f}% solver)", row))
    figure, ax = plt.subplots(figsize=(6.5, 0.65 * len(rows) + 1.2))
    handles = _dots(ax, rows, HANDOFF_POLICIES)
    ax.set_xlabel("Set score (mean per-frame error, frames 21 to 70), 95% interval")
    _legend(figure, handles, "Hand-off at the budget the frozen threshold used")
    _save(figure, out, "handoff_comparison")

    sets = list(result["sets"].items())
    figure, axes = _panels(len(sets), sharey=False, sharex=False)
    handles = {}
    for ax, (split, measured) in zip(axes, sets, strict=False):
        by_policy = {}
        for point in measured["sweep"]:
            group = next(iter(point["groups"].values())) if "pooled" not in point["groups"] else point["groups"]["pooled"]
            by_policy.setdefault(point["policy"], []).append((group["ms_per_frame"], group["set_score"]))
        for policy, points in by_policy.items():
            title, colour = HANDOFF_POLICIES.get(policy, (policy.replace("_", " ").capitalize(), MUTED))
            if policy in HANDOFF_POLICIES:
                ordered = sorted(points)
                handles[policy] = ax.plot(*zip(*ordered, strict=True), color=colour, marker="o", markersize=4, label=title)[0]
            else:
                ax.plot(*zip(*points, strict=True), "D", color=MUTED, markersize=5)
                ax.annotate(title, points[0], xytext=(4, 4), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_title(split, loc="left")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Milliseconds per frame, every member included")
        ax.set_ylabel("Set score")
    _legend(figure, list(handles.values()), "Descriptive sweeps on each set: they choose nothing")
    _save(figure, out, "handoff_sweep")


def ablation_figure(result: dict, out: Path) -> None:
    rows = []
    for label, values in _labelled_groups(result):
        row = {"full_member": (values["full_member"]["set_score"], values["full_member"]["set_score_ci"])}
        for name, run in values["reduced_trainings"].items():
            row[name] = (run["set_score"], run["set_score_ci"])
        for size, run in values["ensemble_sizes"].items():
            row[f"size_{size}"] = (run["set_score"], run["set_score_ci"])
        rows.append((label, row))
    names = {}
    for _, row in rows:
        for name in row:
            names.setdefault(name, (name.replace("_", " ").capitalize(), SLOTS[len(names) % len(SLOTS)]))
    figure, ax = plt.subplots(figsize=(6.5, 0.8 * len(rows) + 1.4))
    handles = _dots(ax, rows, names)
    ax.set_xlabel("Set score (mean per-frame error, frames 21 to 70), 95% interval")
    _legend(figure, handles, "Ablations, for comparison only")
    _save(figure, out, "ablation")


DRAWN = {
    "accuracy": accuracy_figure,
    "doubt": doubt_figures,
    "handoff": handoff_figures,
    "ablation": ablation_figure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, draw in DRAWN.items():
        path = args.results / f"exp_{name}.json"
        if not path.exists():
            print(f"{path} is missing, so its figures aren't drawn.")
            continue
        draw(json.loads(path.read_text()), args.out)


if __name__ == "__main__":
    main()
