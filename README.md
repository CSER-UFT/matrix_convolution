# TCC2 — convolução, GEMM e computação aproximada

Esta branch contém a continuação experimental do projeto desenvolvido no TCC1. A versão original permanece preservada na branch `main`; a branch `tcc2` separa precisão numérica, aproximação e organização da memória para permitir comparações controladas.

O projeto avalia:

- convolução 2D com armazenamento contíguo (`conv_linear`);
- convolução 2D com alocação por linhas (`conv_malloc`);
- multiplicação de matrizes em blocos (`gemm`);
- precisões nativas `float` e `double`;
- versões completas e aproximadas (`skip_kernel` e `skip_k`);
- tempo, ciclos, instruções, IPC e contadores de cache e desvios;
- erro absoluto médio, erro relativo médio, RMSE e erro máximo.

## Organização

```text
program/                         código C e validadores
scripts/                         entradas, execução, análise e gráficos
condor/                          submissão e execução pelo HTCondor
inputs/README.md                 formato das entradas fixas
results/full_20260922_095004/    CSVs consolidados da campanha no Saruê
results/figures_full_20260922/   figuras em PDF e PNG
docs/                            auditoria e instruções dos gráficos
Makefile                         compilação, validação e gráficos
run_condor.sh                    preparação e submissão da campanha completa
```

Entradas binárias, executáveis, arquivos brutos do `perf`, ambientes virtuais e pacotes compactados não são versionados. As entradas são determinísticas e podem ser geradas novamente.

## Requisitos

Ambiente utilizado na campanha publicada:

- Ubuntu Server 20.04;
- GCC 9.4.0;
- perf 5.4.291;
- Python 3.8.10;
- NumPy, pandas, Matplotlib e SciPy nas versões de `requirements.txt`.

Preparação do Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Entradas, compilação e validação

```bash
python3 scripts/generate_inputs.py all
make clean
make
make validate
```

São gerados doze executáveis: três operações, duas precisões e duas técnicas. `float` e `double` usam arquivos de entrada próprios, sem conversão das matrizes durante a medição.

## Execução local

Teste rápido da infraestrutura:

```bash
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh --quick
```

Campanha completa:

```bash
./scripts/run_benchmark.sh --full
```

O perfil completo usa matrizes de ordem 512, 1024 e 2048, kernels 3, 5 e 7, blocos 8, 16, 32, 64 e 128 e 50 repetições. O `perf` mede o programa inteiro, incluindo leitura, organização dos dados, alocação, cálculo, checksum e liberação.

## Execução pelo HTCondor

Depois de gerar as entradas:

```bash
chmod +x run_condor.sh condor/run_full_job.sh scripts/run_benchmark.sh
./run_condor.sh
```

O script recompila, executa os doze validadores, cria `job_input.tar.gz`, submete o job e informa o identificador do cluster. O pacote temporário não é versionado.

## Resultados publicados

A campanha `full_20260922_095004` contém:

- 276 configurações;
- 50 repetições por configuração;
- 13.800 registros consolidados;
- 10.350 comparações pareadas;
- 207 comparações de erro numérico;
- 37 figuras em PDF e PNG.

Os CSVs publicados permitem reproduzir a análise sem os 82 mil arquivos brutos. A auditoria completa está em `docs/AUDITORIA_CAMPANHA_FULL_20260922.md`.

A CPU operou com governador `powersave`. Essa condição foi registrada e aceita para a campanha. Como o tempo apresentou maior variação, ciclos são usados como principal métrica de desempenho e o tempo observado é apresentado com intervalos de confiança.

## Gerar novamente os gráficos

```bash
make plots
```

As figuras são gravadas em `results/figures_full_20260922`. O índice nessa pasta informa a pergunta, os filtros e a fonte de cada gráfico.

Para analisar outro pacote produzido pelo Condor:

```bash
python3 scripts/prepare_analysis_data.py caminho/do/pacote.tar.gz
python3 scripts/generate_plots.py --clean
```

## Limpeza

`make clean` remove somente os executáveis de `bin/`.

Para remover executáveis, entradas e campanhas locais geradas:

```bash
chmod +x scripts/reset_generated.sh
./scripts/reset_generated.sh
```

Os resultados consolidados versionados e as figuras publicadas não são removidos pelo script de reset.
