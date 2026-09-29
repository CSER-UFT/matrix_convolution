#!/usr/bin/env python3
"""Executa uma saída por configuração e calcula erros fora do perf."""

import csv
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_dump(path: Path):
    with path.open("rb") as file:
        count_data = file.read(8)
        if len(count_data) != 8:
            raise ValueError(f"dump invalido: {path}")
        count = struct.unpack("<Q", count_data)[0]
        data = np.fromfile(file, dtype="<f8", count=count)
        trailing = file.read(1)
    if len(data) != count or trailing:
        raise ValueError(f"dump incompleto: {path}")
    return data


def metrics(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("saidas com tamanhos diferentes")
    difference = np.abs(candidate - reference)
    denominator = np.maximum(np.abs(reference), 1e-12)
    return (float(difference.mean()), float((difference / denominator).mean()),
            float(np.sqrt(np.mean(difference * difference))), float(difference.max()))


def run_dump(root: Path, program: str, input_path: str, aux: str, parameter: int, destination: Path, operation: str):
    command = [str(root / "bin" / program), input_path]
    command.append(str(parameter) if operation == "gemm" else aux)
    command.extend(["--dump", str(destination)])
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Uso: collect_accuracy.py <campanha> <--quick|--full>")
    campaign = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(campaign / "manifest.csv", keep_default_na=False)
    temp = campaign / "accuracy_tmp"
    temp.mkdir(exist_ok=True)
    rows = []
    try:
        case_keys = ["operation", "N", "dist", "parameter"]
        for case_values, subset in manifest.groupby(case_keys):
            case = dict(zip(case_keys, case_values))
            operation = case["operation"]
            prefix = operation
            programs = {
                "double_full": f"{prefix}_double_full",
                "float_full": f"{prefix}_float_full",
                "double_skip": f"{prefix}_double_{'skip_k' if operation == 'gemm' else 'skip_kernel'}",
                "float_skip": f"{prefix}_float_{'skip_k' if operation == 'gemm' else 'skip_kernel'}",
            }
            dumps = {}
            for label, program in programs.items():
                matches = subset[subset["program"] == program]
                if matches.empty:
                    raise ValueError(f"programa ausente no manifesto: {program} em {case}")
                record = matches.iloc[0]
                input_path = str(record["input"])
                aux = "" if pd.isna(record["aux_input"]) else str(record["aux_input"])
                destination = temp / f"{label}.bin"
                run_dump(root, program, input_path, aux, int(case["parameter"]), destination, operation)
                dumps[label] = read_dump(destination)
            comparisons = [
                ("precision", "float_full", "double_full"),
                ("approximation_double", "double_skip", "double_full"),
                ("approximation_float", "float_skip", "float_full"),
            ]
            for comparison, candidate, reference in comparisons:
                abs_mean, rel_mean, rmse, maximum = metrics(dumps[reference], dumps[candidate])
                rows.append({
                    "operation": operation, "N": case["N"], "dist": case["dist"],
                    "parameter": case["parameter"], "comparison": comparison,
                    "reference_program": programs[reference], "candidate_program": programs[candidate],
                    "error_abs_mean": abs_mean, "error_rel_mean": rel_mean,
                    "rmse": rmse, "error_max": maximum,
                })
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    with (campaign / "accuracy.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Erros numericos: {campaign / 'accuracy.csv'} ({len(rows)} comparacoes)")


if __name__ == "__main__":
    main()
