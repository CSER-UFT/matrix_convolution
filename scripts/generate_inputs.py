#!/usr/bin/env python3
"""Gera uma vez as entradas binárias fixas e o manifesto SHA-256 da campanha."""

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
    if dist == 0:
        for _ in range(count):
            yield prng.uniform()
    elif dist == 1:
        produced = 0
        while produced < count:
            u1, u2 = prng.uniform(), prng.uniform()
            radius = math.sqrt(-2.0 * math.log(u1))
            for value in (radius * math.cos(2.0 * math.pi * u2),
                          radius * math.sin(2.0 * math.pi * u2)):
                if produced < count:
                    yield value
                    produced += 1
    elif dist == 2:
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


def generate_matrix(n: int, dist: int, precision: str, data_type: int, value_format: str):
    seed = 12346 + dist
    path = INPUTS / precision / f"matrix_n{n}_d{dist}.bin"
    write_file(path, 1, n, dist, seed, values(SplitMix64(seed), n * n, dist),
               n * n, data_type, value_format)


def generate_gemm(n: int, precision: str, data_type: int, value_format: str):
    seed = 22346
    path = INPUTS / precision / f"gemm_n{n}.bin"
    write_file(path, 2, n, -1, seed, values(SplitMix64(seed), 2 * n * n, 0),
               2 * n * n, data_type, value_format)


def generate_kernel(k: int, precision: str, data_type: int, value_format: str):
    seed = 32346 + k
    raw = list(values(SplitMix64(seed), k * k, 0))
    total = sum(raw)
    normalized = (value / total for value in raw)
    write_file(INPUTS / precision / f"kernel_k{k}.bin", 3, k, -1, seed,
               normalized, k * k, data_type, value_format)


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
    INPUTS.mkdir(parents=True, exist_ok=True)
    sizes = [64] if args.profile == "quick" else [512, 1024, 2048]
    if args.profile == "all":
        sizes = [64, 512, 1024, 2048]
    distributions = [0] if args.profile == "quick" else [0, 1, 2]
    kernels = [3] if args.profile == "quick" else [3, 5, 7]
    for precision, (data_type, value_format) in PRECISIONS.items():
        for n in sizes:
            for dist in distributions:
                generate_matrix(n, dist, precision, data_type, value_format)
            generate_gemm(n, precision, data_type, value_format)
        for k in kernels:
            generate_kernel(k, precision, data_type, value_format)
    write_manifest()
    print(f"Entradas fixas geradas em {INPUTS}")


if __name__ == "__main__":
    main()
