"""Write the numbers in docs/RESULTS.md from the saved results, refusing any without provenance.

    uv run python scripts/render_results.py

Only the block between the two markers is replaced, so the written analysis around it stays. No figure
is ever typed: this block is all generated.
"""

import argparse
import json
import sys
from pathlib import Path

from sandeha.eval.experiments import EXPERIMENTS, MissingProvenance, check_provenance

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def _number(value, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _interval(pair) -> str:
    return "n/a" if not pair else f"[{pair[0]:.3f}, {pair[1]:.3f}]"


def _share(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return [*lines, ""]


def _groups(result: dict):
    for split, measured in result["sets"].items():
        for group, values in measured["groups"].items():
            yield (split if group == "all" else f"{split} {group}"), values


def accuracy_lines(result: dict) -> list[str]:
    lines = ["## Accuracy", ""]
    for split, measured in result["sets"].items():
        lines.append(f"Reference solver on `{split}`: {_number(measured['reference_ms_per_frame'], 1)} ms per frame.")
    lines.append("")
    rows = []
    for label, values in _groups(result):
        for name, run in values["policies"].items():
            vorticity = run["secondary"]["primary_window"]["vorticity_correlation"]["mean"]
            rows.append(
                [
                    label,
                    name,
                    _number(run["set_score"]),
                    _interval(run["set_score_ci"]),
                    _share(run["blown_up_share"]),
                    _number(run["usable_frames"]["mean"], 1),
                    _number(vorticity),
                    _number(run["ms_per_frame"], 1),
                ]
            )
    lines += _table(
        ["Set", "Policy", "Set score", "95% interval", "Blown up", "Usable frames", "Vorticity corr.", "ms/frame"], rows
    )
    late = [[label, *values["late_obstacles"].values()] for label, values in _groups(result)]
    lines += ["Scenes by when a late obstacle changes the solid:", ""]
    lines += _table(["Set", "Inside the window", "After it", "None"], late)
    contact = [[label, values["no_contact_scenes"]["count"], values["no_contact_scenes"]["of"]] for label, values in _groups(result) if "no_contact_scenes" in values]
    if contact:
        lines += ["Scenes with no contact by the window's end, by the coarse rule on the stored reference:", ""]
        lines += _table(["Set", "No contact", "Of"], contact)
    return lines


def _doubt_row(label: str, signal: str, figures: dict) -> list:
    evidence = figures["evidence"]
    return [
        label,
        signal,
        _number(evidence["value"]),
        _interval(evidence["ci95"]),
        evidence["undefined_times"],
        figures["non_finite_frames"],
        _number(figures["within_time_auroc"]["value"]),
        figures["within_time_auroc"]["excluded_times"],
        _number(figures["pooled_spearman_not_evidence"]),
    ]


def doubt_lines(result: dict) -> list[str]:
    lines = ["## Doubt", "", "Only the within-time Spearman counts as evidence. Pooled figures do not by themselves.", ""]
    headers = ["Set", "Signal", "Within-time Spearman", "95% interval", "Undefined times", "Non-finite frames", "Within-time AUROC", "Times excluded", "Pooled Spearman"]
    for window, title in (("primary_window", "Frames 21 to 70"), ("long_window", "Frames 21 to 120 (secondary, not evidence)")):
        rows = []
        for label, values in _groups(result):
            measured = values[window]
            for name, figures in measured["aggregations"].items():
                marker = " (chosen)" if name == values["aggregation"] else ""
                rows.append(_doubt_row(label, f"{name}{marker}", figures))
            rows.append(_doubt_row(label, "no ensemble", measured["no_ensemble"]))
            rows.append(_doubt_row(label, "independent (secondary)", measured["independent_secondary"]))
        lines += [f"### {title}", "", *_table(headers, rows)]

    rise = result.get("rise_on_unfamiliar", {})
    if "statement" in rise:
        lines += [rise["statement"], ""]
    else:
        rows = []
        if "test_b" in rise:
            rows.append(["test_b", _number(rise["test_b"]["value"]), _interval(rise["test_b"]["ci95"])])
        for side, found in rise.get("test_c", {}).items():
            rows.append([f"test_c {side}", _number(found["value"]), _interval(found["ci95"])])
        lines += ["### Rise on unfamiliar sets, against Test A", "", *_table(["Set", "Chance of higher doubt", "95% interval"], rows)]
    return lines


def _baseline_row(label: str, kind: str, run: dict) -> list:
    return [
        label,
        kind,
        _number(run["set_score"]),
        _share(run["achieved_share"]),
        run["status"],
        run["forced_handoffs"],
        _share(run["blown_up_share"]),
        _number(run["difference"]),
        _interval(run["difference_ci"]),
        "yes" if run["beats"] else "no",
    ]


def handoff_lines(result: dict) -> list[str]:
    lines = ["## Hand-off", ""]
    choice = next(iter(result["sets"].values()))["choice"]
    lines += [
        f"Frozen on validation: kind `{choice['kind']}`, aggregation `{choice['aggregation']}`, "
        f"`handoff_frames` {choice['handoff_frames']}, `tau` {choice['tau']}.",
        "",
    ]
    rows, oracles, statements = [], [], []
    for label, values in _groups(result):
        frozen = values["frozen"]
        rows.append(
            [label, "doubt", _number(frozen["set_score"]), _share(frozen["achieved_share"]), "budget",
             frozen["forced_handoffs"], _share(frozen["blown_up_share"]), "", "", ""]
        )
        if "statement" in values:
            statements.append(f"- `{label}`: {values['statement']}")
        for kind in ("fixed", "random"):
            for run in values[kind]:
                rows.append(_baseline_row(label, f"{kind} ({run['role']})", run))
        oracle = values["oracle"]
        if oracle.get("run"):
            oracles.append(
                [label, _number(oracle["set_score"]), _share(oracle["achieved_share"]),
                 _number(oracle["difference_from_doubt"]), _interval(oracle["difference_from_doubt_ci"]),
                 "yes" if oracle.get("unmatched") else "no"]
            )
        else:
            statements.append(f"- `{label}`: {oracle['statement']}")
    lines += _table(
        ["Set", "Policy", "Set score", "Solver share", "Match", "Forced", "Blown up", "Doubt less baseline", "95% interval", "Doubt beats it"],
        rows,
    )
    lines += [*statements, ""] if statements else []
    if oracles:
        lines += ["### Oracle bound (descriptive)", ""]
        lines += _table(["Set", "Set score", "Solver share", "Doubt less oracle", "95% interval", "Unmatched"], oracles)
    return lines


def ablation_lines(result: dict) -> list[str]:
    lines = ["## Ablations", "", *[f"{note}" for note in result["notes"]], ""]
    rows = []
    for label, values in _groups(result):
        base = values["full_member"]
        rows.append([label, "full member", _number(base["set_score"]), _interval(base["set_score_ci"]), _share(base["blown_up_share"]), ""])
        for name, run in values["reduced_trainings"].items():
            rows.append([label, name, _number(run["set_score"]), _interval(run["set_score_ci"]), _share(run["blown_up_share"]), ""])
        for size, run in values["ensemble_sizes"].items():
            rows.append([label, f"{size} members", _number(run["set_score"]), _interval(run["set_score_ci"]), _share(run["blown_up_share"]), _number(run["evidence"]["value"])])
    return lines + _table(["Set", "Run", "Set score", "95% interval", "Blown up", "Within-time Spearman"], rows)


RENDERED = {"accuracy": accuracy_lines, "doubt": doubt_lines, "handoff": handoff_lines, "ablation": ablation_lines}


def provenance_lines(results: dict) -> list[str]:
    rows = [
        [f"`exp_{name}.json`", f"`{result['git_commit'][:12]}`", "yes" if result.get("git_dirty") else "no",
         f"`{result['data_manifest_sha256'][:12]}`", len(result["checkpoint_sha256"]), "yes" if result.get("final") else "no"]
        for name, result in results.items()
    ]
    lines = ["## Provenance", "", *_table(["File", "Commit", "Changed tree", "Data manifest", "Checkpoints", "Final"], rows)]
    if any(result.get("dry_run") for result in results.values()):
        lines = ["**Dry run: small random models on tiny data. These numbers measure nothing.**", "", *lines]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args()

    results = {}
    for name in EXPERIMENTS:
        path = args.results / f"exp_{name}.json"
        if not path.exists():
            print(f"{path} is missing, so it isn't rendered.")
            continue
        results[name] = json.loads(path.read_text())
        try:
            check_provenance(results[name])
        except MissingProvenance as error:
            sys.exit(f"Refusing {path}: {error}")
    if not results:
        sys.exit(f"No results in {args.results}, so nothing is rendered.")

    block = provenance_lines(results)
    for name, result in results.items():
        block += RENDERED[name](result)
    generated = "\n".join([BEGIN, "", *block, END]) + "\n"

    existing = args.out.read_text() if args.out.exists() else "# Results\n\n"
    if BEGIN in existing and END in existing:
        head, rest = existing.split(BEGIN, 1)
        tail = rest.split(END, 1)[1].lstrip("\n")
        text = head + generated + tail
    else:
        text = existing.rstrip("\n") + "\n\n" + generated
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
