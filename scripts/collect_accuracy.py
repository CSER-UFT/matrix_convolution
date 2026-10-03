#!/usr/bin/env python3
"""Executa uma saída por configuração (fora do perf) e calcula os erros numéricos.

Mudanças em relação à versão anterior:
  * Métrica principal: erro relativo pela norma, ||C - R||_F / ||R||_F. O erro
    relativo médio elemento a elemento explode quando a referência tem valores
    perto de zero (distribuição normal, kernel Sobel); ele continua no CSV como
    error_rel_mean_elementwise só para comparação com a campanha anterior.
  * PSNR em dB, com a amplitude (máximo menos mínimo) da referência.
  * Na convolução, as métricas usam só o interior (a borda de largura K/2 não é
    calculada e vale zero nas duas saídas).
  * Comparações com eixo step e a comparação combined (float aproximado contra
    double completo).
"""

import csv
import math
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


def interior(values, operation: str, n: int, parameter: int):
    matrix = values.reshape(n, n)
    if operation.startswith("conv"):
        off = parameter // 2
        matrix = matrix[off:n - off, off:n - off]
    return matrix.ravel()


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("saidas com tamanhos diferentes")
    difference = candidate - reference
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    norm_ref = float(np.linalg.norm(reference))
    amplitude = float(reference.max() - reference.min())
    return {
        "error_rel_norm": float(np.linalg.norm(difference)) / norm_ref if norm_ref > 0 else math.nan,
        "error_abs_mean": float(absolute.mean()),
        "error_rel_mean_elementwise": float((absolute / np.maximum(np.abs(reference), 1e-12)).mean()),
        "rmse": rmse,
        "error_max": float(absolute.max()),
        "psnr_db": 20.0 * math.log10(amplitude / rmse) if rmse > 0 and amplitude > 0 else math.inf,
    }


def run_dump(root: Path, binary: str, input_path: str, aux: str, parameter: int,
             step: int, destination: Path, operation: str):
    command = [str(root / "bin" / binary), input_path,
               str(parameter) if operation == "gemm" else aux,
               "--step", str(step), "--reps", "1", "--dump", str(destination)]
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Uso: collect_accuracy.py <campanha> [--quick|--full]")
    campaign = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(campaign / "manifest.csv", keep_default_na=False)
    temp = campaign / "accuracy_tmp"
    temp.mkdir(exist_ok=True)
    rows = []
    try:
        case_keys = ["operation", "N", "dist", "parameter", "kernel_type"]
        for case_values, subset in manifest.groupby(case_keys):
            case = dict(zip(case_keys, case_values))
            operation, n, parameter = case["operation"], int(case["N"]), int(case["parameter"])
            dumps, names = {}, {}
            variants = subset.drop_duplicates(["program", "step"])
            for record in variants.to_dict("records"):
                key = (record["precision"], int(record["step"]))
                destination = temp / f"{record['program']}_s{key[1]}.bin"
                run_dump(root, record["program"], record["input"], record["aux_input"],
                         parameter, key[1], destination, operation)
                dumps[key] = interior(read_dump(destination), operation, n, parameter)
                names[key] = record["program"]
            steps = sorted({step for _, step in dumps if step > 1})
            definitions = [("precision", ("double", 1), ("float", 1))]
            for step in steps:
                definitions += [
                    ("approximation_double", ("double", 1), ("double", step)),
                    ("approximation_float", ("float", 1), ("float", step)),
                    ("combined", ("double", 1), ("float", step)),
                ]
            for comparison, reference, candidate in definitions:
                rows.append({
                    **case, "comparison": comparison, "step": candidate[1],
                    "reference_program": names[reference], "candidate_program": names[candidate],
                    "reference_step": reference[1], "candidate_step": candidate[1],
                    **metrics(dumps[reference], dumps[candidate]),
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
