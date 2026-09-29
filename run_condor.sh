#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p condor/logs

echo "[INFO] Verificando manifesto das entradas..."
(cd inputs && sha256sum -c SHA256SUMS >/dev/null)

echo "[INFO] Compilando..."
make clean
make

echo "[INFO] Validando os 12 programas..."
VALIDATION_OUTPUT=$(make validate)
printf '%s\n' "$VALIDATION_OUTPUT"

OK_COUNT=$(printf '%s\n' "$VALIDATION_OUTPUT" | grep -c '^OK:' || true)

if [[ "$OK_COUNT" -ne 12 ]]; then
    echo "[ERRO] Esperadas 12 validacoes OK; encontradas: $OK_COUNT"
    exit 1
fi

echo "[INFO] Criando pacote temporario do job..."
tar -czf job_input.tar.gz \
    Makefile requirements.txt bin inputs program scripts

echo "[INFO] Enviando campanha completa ao HTCondor..."

if SUBMIT_OUTPUT=$(condor_submit condor/perf.sub 2>&1); then
    printf '%s\n' "$SUBMIT_OUTPUT"
else
    printf '%s\n' "$SUBMIT_OUTPUT" >&2
    echo "[ERRO] Submissao falhou. Nenhuma confirmacao de job recebido." >&2
    exit 1
fi

CLUSTER_ID=$(printf '%s\n' "$SUBMIT_OUTPUT" |
    sed -n 's/.*submitted to cluster \([0-9][0-9]*\)\..*/\1/p')

if [[ -z "$CLUSTER_ID" ]]; then
    echo "[AVISO] Submissao retornou sucesso, mas o ID nao foi identificado."
    echo "[INFO] Consulte condor_q antes de tentar submeter novamente."
    exit 1
fi

echo "[INFO] Cluster ID: $CLUSTER_ID"
echo "[INFO] Acompanhar: condor_q $CLUSTER_ID"
echo "[INFO] Log geral: condor/logs/${CLUSTER_ID}.log"
echo "[INFO] Saida: condor/logs/${CLUSTER_ID}_0.out"
echo "[INFO] Erros: condor/logs/${CLUSTER_ID}_0.err"
