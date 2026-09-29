#!/usr/bin/env bash
set -euo pipefail

cd "${_CONDOR_SCRATCH_DIR:-$PWD}"

RESULT_ARCHIVE="condor_full_results.tar.gz"
RESULT_SHA256="condor_full_results.tar.gz.sha256"

# Garante que os arquivos esperados pelo Condor sempre existam.
: > "$RESULT_ARCHIVE"
: > "$RESULT_SHA256"

package_results() {
    local original_status=$?
    local full_dir=""
    local full_name=""
    local tar_status=0
    local sha_status=0

    trap - EXIT
    set +e

    full_dir=$(find results -maxdepth 1 -type d -name 'full_*' 2>/dev/null |
        sort | tail -n1)

    if [[ -n "$full_dir" ]]; then
        full_name=$(basename "$full_dir")
        echo "[INFO] Compactando campanha: $full_name"
        tar -czf "$RESULT_ARCHIVE" -C results "$full_name"
        tar_status=$?
    elif [[ -d results ]]; then
        echo "[AVISO] Campanha full não identificada; salvando results completo." >&2
        tar -czf "$RESULT_ARCHIVE" results
        tar_status=$?
    else
        echo "[AVISO] Nenhuma pasta de resultados foi criada." >&2
        tar -czf "$RESULT_ARCHIVE" --files-from /dev/null
        tar_status=$?
    fi

    if [[ $tar_status -eq 0 ]]; then
        sha256sum "$RESULT_ARCHIVE" > "$RESULT_SHA256"
        sha_status=$?
    else
        echo "[ERRO] Falha ao compactar os resultados." >&2
        : > "$RESULT_ARCHIVE"
        printf 'ERRO: compactacao dos resultados falhou\n' > "$RESULT_SHA256"
        original_status=90
    fi

    if [[ $sha_status -ne 0 ]]; then
        echo "[ERRO] Falha ao calcular SHA256." >&2
        original_status=91
    fi

    echo "[INFO] Fim: $(date --iso-8601=seconds)"
    echo "[INFO] Status original: $original_status"
    exit "$original_status"
}

trap package_results EXIT

tar -xzf job_input.tar.gz

echo "[INFO] Inicio: $(date --iso-8601=seconds)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Diretorio: $(pwd)"

python3 -c 'import numpy, pandas, matplotlib'

chmod +x scripts/run_benchmark.sh

echo "[INFO] Verificando entradas..."
(cd inputs && sha256sum -c SHA256SUMS >/dev/null)

echo "[INFO] Iniciando campanha completa..."
./scripts/run_benchmark.sh --full

FULL_DIR=$(find results -maxdepth 1 -type d -name 'full_*' | sort | tail -n1)

if [[ -z "$FULL_DIR" ]]; then
    echo "[ERRO] Pasta da campanha completa não encontrada."
    exit 1
fi

echo "[INFO] Campanha concluída: $(basename "$FULL_DIR")"
