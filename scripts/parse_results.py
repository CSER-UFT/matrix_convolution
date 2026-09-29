#!/usr/bin/env python3
"""Consolida os arquivos brutos do perf sem substituir as evidências originais."""

import csv
import math
import re
import sys
from pathlib import Path

import pandas as pd

EVENT_COLUMNS = {
    "instructions": "instructions",
    "cycles": "cycles",
    "cache-references": "cache_references",
    "cache-misses": "cache_misses",
    "branches": "branches",
    "branch-misses": "branch_misses",
    "L1-dcache-load-misses": "l1_dcache_load_misses",
    "LLC-load-misses": "llc_load_misses",
}


def parse_perf(path: Path):
    values = {}
    elapsed = None
    if not path.exists():
        raise ValueError(f"arquivo ausente: {path}")
    with path.open(encoding="utf-8", errors="replace") as file:
        for fields in csv.reader(file, delimiter=";"):
            if len(fields) < 3:
                continue
            raw, event = fields[0].strip(), fields[2].strip()
            if event == "seconds time elapsed":
                try:
                    elapsed = float(raw)
                except ValueError as error:
                    raise ValueError(f"tempo total invalido em {path}: {raw}") from error
                continue
            base = event.rsplit(":", 1)[0] if event.endswith(":u") else event
            if base == "duration_time":
                elapsed = float(raw) / 1e9
                continue
            if base not in EVENT_COLUMNS:
                continue
            if raw.startswith("<"):
                raise ValueError(f"contador invalido em {path}: {event}={raw}")
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f"contador invalido em {path}: {event}={raw}") from error
            if len(fields) >= 5 and fields[4].strip():
                try:
                    running = float(fields[4])
                    if running < 99.0:
                        raise ValueError(f"multiplexacao em {path}: {event}={running}%")
                except ValueError as error:
                    if "multiplexacao" in str(error):
                        raise
            values[EVENT_COLUMNS[base]] = value
    return values, elapsed


def parse_output(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    checksum_match = re.search(r"^Checksum\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    if not checksum_match:
        raise ValueError(f"saida incompleta: {path}")
    return float(checksum_match.group(1))


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Uso: parse_results.py <diretorio_da_campanha>")
    campaign = Path(sys.argv[1]).resolve()
    manifest = pd.read_csv(campaign / "manifest.csv", keep_default_na=False)
    rows = []
    for record in manifest.to_dict("records"):
        p1, total_time1 = parse_perf(Path(record["p1_file"]))
        p2, total_time2 = parse_perf(Path(record["p2_file"]))
        checksum1 = parse_output(Path(record["p1_output"]))
        checksum2 = parse_output(Path(record["p2_output"]))
        if total_time1 is None:
            raise ValueError(f"tempo total ausente: {record['p1_file']}")
        if total_time2 is None:
            total_time2 = total_time1
        tolerance = 1e-9 * max(abs(checksum1), 1.0) + 1e-9
        if abs(checksum1 - checksum2) > tolerance:
            raise ValueError(f"checksum mudou entre passagens: {record['run_id']}")
        row = dict(record)
        row.update(p1)
        row.update(p2)
        row["time_p1"] = total_time1
        row["time_p2"] = total_time2
        row["time_sec"] = (total_time1 + total_time2) / 2.0
        row["checksum"] = checksum1
        row["ipc"] = row.get("instructions", 0) / row.get("cycles", math.nan)
        row["cache_miss_rate"] = row.get("cache_misses", 0) / row.get("cache_references", math.nan)
        row["branch_miss_rate"] = row.get("branch_misses", 0) / row.get("branches", math.nan)
        if record["operation"].startswith("conv"):
            interior = max(int(record["N"]) - int(record["parameter"]) + 1, 0) ** 2
            terms = int(record["parameter"]) ** 2 if record["technique"] == "full" else ((int(record["parameter"]) + 1) // 2) ** 2
        else:
            interior = int(record["N"]) ** 2
            terms = int(record["N"]) if record["technique"] == "full" else (int(record["N"]) + 1) // 2
        operations_per_term = 3 if record["technique"] == "skip_k" else 2
        row["estimated_operations"] = operations_per_term * interior * terms
        row["gops"] = row["estimated_operations"] / row["time_sec"] / 1e9
        rows.append(row)

    raw = pd.DataFrame(rows)
    keys = ["operation", "program", "precision", "technique", "N", "dist", "parameter", "input", "aux_input"]
    counts = raw.groupby(keys).size()
    if counts.nunique() != 1:
        raise ValueError(f"quantidades diferentes de repeticoes: {counts.to_dict()}")
    raw.to_csv(campaign / "results_raw.csv", index=False)

    metrics = [c for c in [
        "time_sec", "time_p1", "time_p2", "cycles", "instructions", "ipc",
        "cache_references", "cache_misses", "cache_miss_rate", "branches",
        "branch_misses", "branch_miss_rate", "l1_dcache_load_misses",
        "llc_load_misses", "checksum",
        "estimated_operations", "gops",
    ] if c in raw.columns]
    grouped = raw.groupby(keys)[metrics]
    summary = grouped.agg(["mean", "median", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["repetitions"] = grouped.size().values
    for metric in metrics:
        mean, std = f"{metric}_mean", f"{metric}_std"
        if mean in summary and std in summary:
            summary[f"{metric}_cv"] = summary[std] / summary[mean].abs().replace(0, math.nan)
    summary.to_csv(campaign / "results_summary.csv", index=False)

    comparison_rows = []
    case_keys = ["operation", "N", "dist", "parameter", "repetition"]
    for case, subset in raw.groupby(case_keys):
        programs = subset.set_index("program")
        operation = case[0]
        prefix = operation
        skip = "skip_k" if operation == "gemm" else "skip_kernel"
        definitions = [
            ("precision", f"{prefix}_double_full", f"{prefix}_float_full"),
            ("approximation_double", f"{prefix}_double_full", f"{prefix}_double_{skip}"),
            ("approximation_float", f"{prefix}_float_full", f"{prefix}_float_{skip}"),
        ]
        for name, reference, candidate in definitions:
            if reference not in programs.index or candidate not in programs.index:
                raise ValueError(f"par ausente em {case}: {reference}/{candidate}")
            ref, cand = programs.loc[reference], programs.loc[candidate]
            comparison_rows.append({
                **dict(zip(case_keys, case)), "comparison": name,
                "reference_program": reference, "candidate_program": candidate,
                "reference_input": ref["input"], "candidate_input": cand["input"],
                "reference_aux_input": ref["aux_input"], "candidate_aux_input": cand["aux_input"],
                "time_speedup": ref["time_sec"] / cand["time_sec"],
                "cycles_speedup": ref["cycles"] / cand["cycles"],
                "instruction_ratio": cand["instructions"] / ref["instructions"],
                "cache_miss_rate_reference": ref["cache_miss_rate"],
                "cache_miss_rate_candidate": cand["cache_miss_rate"],
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(campaign / "comparisons_raw.csv", index=False)
    comparison_keys = case_keys[:-1] + ["comparison", "reference_program", "candidate_program"]
    comparison_metrics = ["time_speedup", "cycles_speedup", "instruction_ratio",
                          "cache_miss_rate_reference", "cache_miss_rate_candidate"]
    comp_group = comparisons.groupby(comparison_keys)[comparison_metrics]
    comp_summary = comp_group.agg(["mean", "median", "std"])
    comp_summary.columns = [f"{metric}_{stat}" for metric, stat in comp_summary.columns]
    comp_summary = comp_summary.reset_index()
    comp_summary["repetitions"] = comp_group.size().values
    comp_summary.to_csv(campaign / "comparisons_summary.csv", index=False)
    print(f"Dados brutos: {campaign / 'results_raw.csv'} ({len(raw)} linhas)")
    print(f"Resumo: {campaign / 'results_summary.csv'} ({len(summary)} configuracoes)")
    print(f"Comparacoes pareadas: {campaign / 'comparisons_raw.csv'} ({len(comparisons)} linhas)")


if __name__ == "__main__":
    main()
