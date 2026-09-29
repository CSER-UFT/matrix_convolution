#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f Makefile || ! -f program/matrix_bench.c || ! -f scripts/run_benchmark.sh ]]; then
  echo "Erro: estrutura de TCC2_melhorado nao encontrada em $ROOT." >&2
  exit 1
fi

make clean

find inputs -type f -name '*.bin' -delete
rm -f -- inputs/SHA256SUMS

find results -mindepth 1 -maxdepth 1 -type d \
  \( -name 'quick_*' -o -name 'full_*' \) -exec rm -rf -- {} +

find scripts -type d -name '__pycache__' -prune -exec rm -rf -- {} +
find scripts -type f -name '*.pyc' -delete

mkdir -p results
touch results/.gitkeep

echo "Arquivos gerados removidos. Fontes, documentos, requirements e .venv foram preservados."
echo "Para recomecar: gere as entradas, compile, valide e execute o teste piloto."
