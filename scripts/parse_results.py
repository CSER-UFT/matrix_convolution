#!/usr/bin/env python3
"""Consolida os arquivos brutos do perf e as saídas dos programas.

Mudanças em relação à versão anterior:
  * O tempo vem do próprio programa (TimeMedian, medido só em volta do kernel),
    não do duration_time do perf, que media o programa inteiro.
  * Os contadores do perf cobrem a mesma região (perf stat --control) e são
    divididos por Reps: todas as colunas de contadores são POR EXECUÇÃO do kernel.
  * time_p1 e time_p2 agora são medições independentes (uma por passada).
  * Novo eixo "step" (grau de aproximação) e nova coluna "kernel_type".
  * Comparações: precision, approximation_double, approximation_float e
    combined (float aproximado contra double completo).
  * Resumos com média geométrica dos speedups, além de média, mediana e desvio.
  * Energia (passada 3, RAPL em modo sistema): energia do pacote e da DRAM por
    execução do kernel, potência média, energia acima da potência ociosa e EDP.
"""

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
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
    "power/energy-pkg/": "energy_pkg_j",
    "power/energy-ram/": "energy_ram_j",
}
OUTPUT_FIELDS = {
    "Reps": int, "TimeMedian": float, "TimeMin": float, "TimeMean": float,
    "KeptTerms": int, "Checksum": float,
}
KEYS = ["operation", "program", "precision", "technique", "step", "N", "dist",
        "parameter", "kernel_type", "input", "aux_input"]


def parse_perf(path: Path):
    values = {}
    if not path.exists():
        raise ValueError(f"arquivo ausente: {path}")
    with path.open(encoding="utf-8", errors="replace") as file:
        for fields in csv.reader(file, delimiter=";"):
            if len(fields) < 3:
                continue
            raw, event = fields[0].strip(), fields[2].strip()
            base = event.rsplit(":", 1)[0] if event.endswith(":u") else event
            if base not in EVENT_COLUMNS:
                continue
            if raw.startswith("<"):
                raise ValueError(f"contador invalido em {path}: {event}={raw}")
            if len(fields) >= 5 and fields[4].strip():
                running = float(fields[4])
                if running < 99.0:
                    raise ValueError(f"multiplexacao em {path}: {event}={running}%")
            values[EVENT_COLUMNS[base]] = float(raw)
    return values


def parse_output(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    for name, kind in OUTPUT_FIELDS.items():
        match = re.search(rf"^{name}\s+(\S+)\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"saida incompleta ({name} ausente): {path}")
        result[name] = kind(match.group(1))
    return result


def idle_power(campaign: Path):
    """Potência ociosa média (W) do pacote e da DRAM, das medições de início e fim."""
    seconds = None
    for env in sorted(campaign.glob("environment*.txt")):
        match = re.search(r"^idle_seconds=(\S+)$", env.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            seconds = float(match.group(1))
            break
    files = sorted(campaign.glob("idle_*.perf"))
    if not seconds or not files:
        return {}
    samples = [parse_perf(path) for path in files]
    return {f"idle_{key.replace('_j', '_w')}": float(np.mean([s[key] for s in samples if key in s])) / seconds
            for key in ("energy_pkg_j", "energy_ram_j") if any(key in s for s in samples)}


def estimated_operations(operation: str, n: int, parameter: int, step: int) -> int:
    """Multiplicações e somas efetivamente executadas pelo kernel."""
    if operation.startswith("conv"):
        width = max(n - parameter + 1, 0)
        kept = math.ceil(parameter / step) ** 2
        return 2 * width * width * kept
    return 2 * n * n * math.ceil(n / step)


def geometric_mean(series):
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.log(values).mean())) if len(values) else math.nan


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Uso: parse_results.py <diretorio_da_campanha>")
    campaign = Path(sys.argv[1]).resolve()
    manifest = pd.read_csv(campaign / "manifest.csv", keep_default_na=False)
    idle = idle_power(campaign)
    if idle:
        print("Potencia ociosa: " + ", ".join(f"{k}={v:.2f} W" for k, v in idle.items()))
    rows = []
    for record in manifest.to_dict("records"):
        p1 = parse_perf(campaign / record["p1_file"])
        p2 = parse_perf(campaign / record["p2_file"]) if record["p2_file"] else {}
        o1 = parse_output(campaign / record["p1_output"])
        o2 = parse_output(campaign / record["p2_output"])
        p3 = parse_perf(campaign / record["p3_file"]) if record.get("p3_file") else {}
        o3 = parse_output(campaign / record["p3_output"]) if record.get("p3_output") else o1
        tolerance = 1e-9 * max(abs(o1["Checksum"]), 1.0)
        if abs(o1["Checksum"] - o2["Checksum"]) > tolerance:
            raise ValueError(f"checksum mudou entre passagens: {record['run_id']}")
        row = dict(record)
        for name, value in p1.items():
            row[name] = value / o1["Reps"]
        for name, value in p2.items():
            row[name] = value / o2["Reps"]
        for name, value in p3.items():
            row[name] = value / o3["Reps"]
        row["reps_p1"], row["reps_p2"], row["reps_p3"] = o1["Reps"], o2["Reps"], o3["Reps"]
        row["time_p1"], row["time_p2"], row["time_p3"] = o1["TimeMedian"], o2["TimeMedian"], o3["TimeMedian"]
        row["time_sec"] = 0.5 * (o1["TimeMedian"] + o2["TimeMedian"])
        row["time_min"] = min(o1["TimeMin"], o2["TimeMin"])
        row["kept_terms"] = o1["KeptTerms"]
        row["checksum"] = o1["Checksum"]
        row["ipc"] = row.get("instructions", math.nan) / row.get("cycles", math.nan)
        row["cache_miss_rate"] = row.get("cache_misses", math.nan) / row.get("cache_references", math.nan)
        row["branch_miss_rate"] = row.get("branch_misses", math.nan) / row.get("branches", math.nan)
        row["effective_ghz"] = row.get("cycles", math.nan) / row["time_sec"] / 1e9
        if "energy_pkg_j" in row:
            # Energia por execução do kernel; a potência usa o tempo médio da passada 3.
            time_p3 = o3["TimeMean"]
            row["energy_j"] = row["energy_pkg_j"] + row.get("energy_ram_j", 0.0)
            row["power_pkg_w"] = row["energy_pkg_j"] / time_p3
            row["power_w"] = row["energy_j"] / time_p3
            idle_w = idle.get("idle_energy_pkg_w", math.nan) + idle.get("idle_energy_ram_w", 0.0)
            row["energy_dynamic_j"] = row["energy_j"] - idle_w * time_p3
            row["edp_js"] = row["energy_j"] * row["time_sec"]
        n, parameter, step = int(record["N"]), int(record["parameter"]), int(record["step"])
        row["estimated_operations"] = estimated_operations(record["operation"], n, parameter, step)
        row["full_operations"] = estimated_operations(record["operation"], n, parameter, 1)
        row["gops"] = row["estimated_operations"] / row["time_sec"] / 1e9
        row["effective_gops"] = row["full_operations"] / row["time_sec"] / 1e9
        rows.append(row)

    raw = pd.DataFrame(rows)
    counts = raw.groupby(KEYS).size()
    if counts.nunique() != 1:
        raise ValueError(f"quantidades diferentes de repeticoes: {counts.to_dict()}")
    raw.to_csv(campaign / "results_raw.csv", index=False)

    metrics = [c for c in [
        "time_sec", "time_p1", "time_p2", "time_min", "cycles", "instructions", "ipc",
        "cache_references", "cache_misses", "cache_miss_rate", "branches",
        "branch_misses", "branch_miss_rate", "l1_dcache_load_misses",
        "llc_load_misses", "effective_ghz", "energy_pkg_j", "energy_ram_j", "energy_j",
        "energy_dynamic_j", "power_pkg_w", "power_w", "edp_js", "checksum",
        "estimated_operations", "gops", "effective_gops",
    ] if c in raw.columns]
    grouped = raw.groupby(KEYS)[metrics]
    summary = grouped.agg(["mean", "median", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["repetitions"] = grouped.size().values
    for metric in metrics:
        mean, std = f"{metric}_mean", f"{metric}_std"
        summary[f"{metric}_cv"] = summary[std] / summary[mean].abs().replace(0, math.nan)
    summary.to_csv(campaign / "results_summary.csv", index=False)

    # Comparações pareadas pela repetição: os programas de uma mesma repetição
    # rodam em sequência, em ordem rotacionada, e por isso são vizinhos no tempo.
    comparison_rows = []
    case_keys = ["operation", "N", "dist", "parameter", "kernel_type", "repetition"]
    for case, subset in raw.groupby(case_keys):
        operation = case[0]
        table = subset.set_index(["precision", "step"])
        steps = sorted(s for s in subset["step"].unique() if s > 1)
        definitions = [("precision", ("double", 1), ("float", 1))]
        for step in steps:
            definitions += [
                ("approximation_double", ("double", 1), ("double", step)),
                ("approximation_float", ("float", 1), ("float", step)),
                ("combined", ("double", 1), ("float", step)),
            ]
        for name, reference, candidate in definitions:
            if reference not in table.index or candidate not in table.index:
                raise ValueError(f"par ausente em {case}: {reference}/{candidate}")
            ref, cand = table.loc[reference], table.loc[candidate]
            comparison_rows.append({
                **dict(zip(case_keys, case)), "comparison": name, "step": candidate[1],
                "reference_program": ref["program"], "candidate_program": cand["program"],
                "reference_step": reference[1], "candidate_step": candidate[1],
                "time_speedup": ref["time_sec"] / cand["time_sec"],
                "cycles_speedup": ref["cycles"] / cand["cycles"],
                "instruction_ratio": cand["instructions"] / ref["instructions"],
                "energy_saving": ref.get("energy_j", math.nan) / cand.get("energy_j", math.nan),
                "edp_saving": ref.get("edp_js", math.nan) / cand.get("edp_js", math.nan),
                "cache_miss_rate_reference": ref["cache_miss_rate"],
                "cache_miss_rate_candidate": cand["cache_miss_rate"],
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(campaign / "comparisons_raw.csv", index=False)
    comparison_keys = case_keys[:-1] + ["comparison", "step", "reference_program", "candidate_program"]
    comparison_metrics = ["time_speedup", "cycles_speedup", "instruction_ratio",
                          "energy_saving", "edp_saving",
                          "cache_miss_rate_reference", "cache_miss_rate_candidate"]
    comp_group = comparisons.groupby(comparison_keys)[comparison_metrics]
    comp_summary = comp_group.agg(["mean", "median", "std"])
    comp_summary.columns = [f"{metric}_{stat}" for metric, stat in comp_summary.columns]
    for metric in ["time_speedup", "cycles_speedup", "energy_saving", "edp_saving"]:
        comp_summary[f"{metric}_gmean"] = comparisons.groupby(comparison_keys)[metric].apply(geometric_mean)
    comp_summary = comp_summary.reset_index()
    comp_summary["repetitions"] = comp_group.size().values
    comp_summary.to_csv(campaign / "comparisons_summary.csv", index=False)
    print(f"Dados brutos: {campaign / 'results_raw.csv'} ({len(raw)} linhas)")
    print(f"Resumo: {campaign / 'results_summary.csv'} ({len(summary)} configuracoes)")
    print(f"Comparacoes pareadas: {campaign / 'comparisons_raw.csv'} ({len(comparisons)} linhas)")


if __name__ == "__main__":
    main()
