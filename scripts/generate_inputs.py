#!/usr/bin/env python3
"""Gera uma vez as entradas binárias fixas e o manifesto SHA256 da campanha.

Mudanças em relação à versão anterior:
  * GEMM com mais de uma distribuição (gemm_n{N}_d{dist}.bin). Com valores todos
    positivos o erro relativo do skip_k fica artificialmente pequeno; a
    distribuição 1 (normal, média zero) mostra o caso desfavorável.
  * Três tipos de kernel de convolução (kernel_{tipo}_k{K}.bin):
      rand   pesos aleatórios positivos com soma 1 (o kernel da campanha anterior)
      gauss  suavização gaussiana (sigma no padrão do OpenCV), soma 1
      sobel  derivada horizontal de Sobel generalizada, soma zero, norma L1 = 1
    O campo "dist" do cabeçalho do kernel guarda o tipo (0, 1, 2).
"""

import argparse
import hashlib
import math
import struct
from pathlib import Path

MAGIC = b"TCC2I01\0"
HEADER = struct.Struct("<8sIIIiIIQ")
MASK = (1 << 64) - 1
ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
PRECISIONS = {
    "float": (1, "<f"),
    "double": (2, "<d"),
}
KERNEL_TYPES = {"rand": 0, "gauss": 1, "sobel": 2}

PROFILES = {
    "quick": {"sizes": [64, 128], "conv_dists": [0, 1], "gemm_dists": [0, 1], "kernels": [3, 5]},
    "full": {"sizes": [512, 1024, 2048], "conv_dists": [0, 1, 2], "gemm_dists": [0, 1], "kernels": [3, 5, 7]},
}


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return (z ^ (z >> 31)) & MASK

    def uniform(self) -> float:
        return ((self.next_u64() >> 11) + 0.5) * (1.0 / (1 << 53))


def values(prng: SplitMix64, count: int, dist: int):
    if dist == 0:  # uniforme em (0, 1)
        for _ in range(count):
            yield prng.uniform()
    elif dist == 1:  # normal padrão (Box Muller)
        produced = 0
        while produced < count:
            u1, u2 = prng.uniform(), prng.uniform()
            radius = math.sqrt(-2.0 * math.log(u1))
            for value in (radius * math.cos(2.0 * math.pi * u2),
                          radius * math.sin(2.0 * math.pi * u2)):
                if produced < count:
                    yield value
                    produced += 1
    elif dist == 2:  # exponencial de média 1
        for _ in range(count):
            yield -math.log1p(-prng.uniform())
    else:
        raise ValueError(dist)


def write_file(path: Path, kind: int, n: int, dist: int, seed: int,
               stream, count: int, data_type: int, value_format: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as file:
        file.write(HEADER.pack(MAGIC, 2, kind, n, dist, seed, data_type, count))
        buffer = bytearray()
        written = 0
        for value in stream:
            buffer.extend(struct.pack(value_format, float(value)))
            written += 1
            if len(buffer) >= 1024 * 1024:
                file.write(buffer)
                buffer.clear()
        file.write(buffer)
    if written != count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{path}: esperados {count}, gerados {written}")
    temporary.replace(path)


def generate_matrix(n, dist, precision, data_type, value_format):
    seed = 12346 + dist
    path = INPUTS / precision / f"matrix_n{n}_d{dist}.bin"
    write_file(path, 1, n, dist, seed, values(SplitMix64(seed), n * n, dist),
               n * n, data_type, value_format)


def generate_gemm(n, dist, precision, data_type, value_format):
    seed = 22346 + dist
    path = INPUTS / precision / f"gemm_n{n}_d{dist}.bin"
    write_file(path, 2, n, dist, seed, values(SplitMix64(seed), 2 * n * n, dist),
               2 * n * n, data_type, value_format)


def binomial(m):
    return [math.comb(m, i) for i in range(m + 1)]


def convolve(x, y):
    out = [0.0] * (len(x) + len(y) - 1)
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            out[i + j] += a * b
    return out


def kernel_values(kind: str, k: int, seed: int):
    if kind == "rand":
        raw = list(values(SplitMix64(seed), k * k, 0))
        total = sum(raw)
        return [v / total for v in raw]
    if kind == "gauss":
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        g = [math.exp(-((i - k // 2) ** 2) / (2 * sigma * sigma)) for i in range(k)]
        raw = [gi * gj for gi in g for gj in g]
        total = sum(raw)
        return [v / total for v in raw]
    if kind == "sobel":
        smooth = binomial(k - 1)
        derivative = convolve([-1.0, 0.0, 1.0], binomial(k - 3))
        raw = [si * dj for si in smooth for dj in derivative]
        l1 = sum(abs(v) for v in raw)
        return [v / l1 for v in raw]
    raise ValueError(kind)


def generate_kernel(kind, k, precision, data_type, value_format):
    seed = 32346 + k
    data = kernel_values(kind, k, seed)
    write_file(INPUTS / precision / f"kernel_{kind}_k{k}.bin", 3, k, KERNEL_TYPES[kind],
               seed, data, k * k, data_type, value_format)


def write_manifest():
    lines = []
    for path in sorted(INPUTS.glob("*/*.bin")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(INPUTS).as_posix()}\n")
    (INPUTS / "SHA256SUMS").write_text("".join(lines), encoding="ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("quick", "full", "all"), nargs="?", default="all")
    args = parser.parse_args()
    selected = ["quick", "full"] if args.profile == "all" else [args.profile]
    INPUTS.mkdir(parents=True, exist_ok=True)
    for precision, (data_type, value_format) in PRECISIONS.items():
        for name in selected:
            profile = PROFILES[name]
            for n in profile["sizes"]:
                for dist in profile["conv_dists"]:
                    generate_matrix(n, dist, precision, data_type, value_format)
                for dist in profile["gemm_dists"]:
                    generate_gemm(n, dist, precision, data_type, value_format)
            for k in profile["kernels"]:
                for kind in KERNEL_TYPES:
                    generate_kernel(kind, k, precision, data_type, value_format)
    write_manifest()
    print(f"Entradas fixas geradas em {INPUTS}")


if __name__ == "__main__":
    main()
