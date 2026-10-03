# Pipeline de medição do TCC2 (revisão de outubro de 2026)

Máquina alvo: sarue, HP ProLiant DL360e Gen8, 1 × Xeon E5-2407 v2 (4 núcleos,
sem Hyper-Threading, sem Turbo Boost, AVX sem FMA, L3 de 10 MB), 32 GB.

## O que mudou e por quê

| Problema na campanha de 22/09 | Correção |
|---|---|
| `perf` media o programa inteiro (leitura, cópias, alocação, checksum). Nas convoluções isso era da mesma ordem do kernel. | O programa mede só o kernel com `clock_gettime` e liga e desliga o `perf` em volta do lote de repetições (`perf stat --delay=-1 --control`). Tempo, contadores e energia cobrem a mesma região. |
| Uma execução do kernel por processo, ~10 ms nas convoluções, CV ~18%. | O kernel repete até somar `TCC2_MIN_TIME` (0,2 s, mínimo 3 repetições), com aquecimento descartado; o programa informa a mediana. Contadores e energia são divididos por `Reps`. |
| `time_p2` era cópia de `time_p1`. | Cada passada tem seu próprio tempo, medido pelo programa. |
| GEMM em ordem i, j, k: `b` lida por coluna e soma serial (~1 GFLOP/s, `float` igual a `double`). | Ordem i, k, j dentro dos blocos: laço interno contíguo e vetorizado. Mesma ordem de acumulação, mesmo arredondamento. |
| Convolução com redução no laço de 3 a 7 iterações. | Laço interno percorre as colunas da saída (vetorizado). |
| `skip_k` multiplicava pela escala em todo termo. | Escala aplicada uma vez por `a[i][k]`. |
| Grau de aproximação fixo no código. | `--step S` em tempo de execução; completa e aproximada usam o mesmo binário. |
| `skip_kernel` falhava com kernels de soma zero. | Reescala pela norma L1 (igual à anterior para kernels positivos). |
| Kernel aleatório, GEMM só com valores positivos. | Kernels `rand`, `gauss` e `sobel`; GEMM com distribuições 0 (uniforme) e 1 (normal). |
| Erro relativo médio explodia com referência perto de zero. | Métrica principal `error_rel_norm` = ‖C − R‖ / ‖R‖, mais PSNR; borda da convolução excluída. |
| Sem erro combinado contra a referência. | Comparação `combined`: `float` aproximado contra `double` completo. |
| Energia citada como motivação, mas não medida. | Terceira passada do `perf` com RAPL (pacote e DRAM), potência ociosa no início e no fim, energia por execução, energia acima do ocioso e EDP. |
| Governor powersave aceito com aviso; CPU 0 por padrão. | Exige `performance` (ou `TCC2_ALLOW_GOVERNOR=1`, registrado); usa o último núcleo permitido; registra turbo, slot do Condor e carga da máquina. |
| Binário compilado fora do nó de execução. | `run_benchmark.sh` recompila e valida no próprio nó. |
| Falha no meio perdia a campanha. | `--resume <diretório>`. |
| Gráficos com legendas dentro dos eixos, tapando dados. | `generate_plots.py` reescrito: legenda única abaixo dos painéis, fora da área de desenho; cores validadas para daltonismo; marcador e estilo de linha redundantes à cor. |

## Preparação do sarue (administrador)

```bash
sudo cpupower frequency-set -g performance
sudo sysctl -w kernel.nmi_watchdog=0
sudo sysctl -w kernel.perf_event_paranoid=0   # energia RAPL exige 0
perf --version                                # precisa ser 5.11 ou mais novo
```

Rodar em horário ocioso: o RAPL mede o pacote inteiro, então qualquer outro
processo no servidor entra na energia medida. O `environment.txt` registra o
`loadavg` e as medições de potência ociosa ficam em `idle_*.perf`.

## Como rodar

```bash
python3 scripts/generate_inputs.py full      # uma vez
make validate                                 # testes contra implementação ingênua
make vecreport                                # documenta quais laços vetorizaram
scripts/run_benchmark.sh --quick              # teste rápido (N = 64 e 128)
condor_submit condor/campanha_full.sub        # campanha completa, 9 a 12 horas
```

Depois da campanha:

```bash
make plots DATA=results/full_AAAAMMDD_HHMMSS FIGURES=results/figuras_full
```

(ou, com o pacote `.tar.gz` da campanha, `prepare_analysis_data.py` e depois
`make plots`, que lê `results/analysis_data` por padrão).

Variáveis úteis: `TCC2_REPETITIONS`, `TCC2_MIN_TIME`, `TCC2_MIN_REPS`,
`TCC2_ENERGY=0` (desliga a energia), `TCC2_ENERGY_MIN_TIME`,
`TCC2_IDLE_SECONDS`, `TCC2_CPU`. Todas descritas no topo do `run_benchmark.sh`.

## Interface dos programas

```
bin/conv_linear_{double,float} <matriz.bin> <kernel.bin> [--step S] [--reps R | --min-time T] [--dump arq]
bin/conv_malloc_{double,float} <matriz.bin> <kernel.bin> [...]
bin/gemm_{double,float}        <gemm.bin>   <bloco>      [...]
```

`--step 1` é a versão completa. A saída tem `Reps`, `TimeMedian`, `TimeMin`,
`TimeMean`, `KeptTerms` e `Checksum`.

## CSVs gerados

* `results_raw.csv` / `results_summary.csv`: colunas `step`, `kernel_type`,
  `time_min`, `effective_ghz`, `full_operations`, `effective_gops` e, com
  energia, `energy_pkg_j`, `energy_ram_j`, `energy_j`, `energy_dynamic_j`,
  `power_w`, `power_pkg_w`, `edp_js`. Contadores e energia são por execução do
  kernel.
* `comparisons_*.csv`: comparações `precision`, `approximation_double`,
  `approximation_float` e `combined`, com coluna `step`, `time_speedup`,
  `cycles_speedup`, `energy_saving` e `edp_saving`; resumo com média geométrica
  (`*_gmean`).
* `accuracy.csv`: `error_rel_norm` (principal), `psnr_db`, `rmse`, `error_max`,
  `error_abs_mean` e `error_rel_mean_elementwise` (só para comparar com a
  campanha anterior).

## Gráficos

`generate_plots.py` gera 28 figuras em 9 categorias (desempenho, comparações,
blocos da GEMM, qualidade numérica, compromisso entre erro e tempo, energia,
microarquitetura, reprodutibilidade e diagnóstico da frequência), com índice em
`INDICE_GRAFICOS.md`. As figuras que não mostram todas as combinações usam o
recorte definido no topo do script (`CONV_DIST`, `CONV_KERNEL`, `GEMM_DIST`).

A figura `09_diagnosticos/frequencia_efetiva` serve de controle: no E5-2407 v2,
sem turbo e com governor `performance`, a frequência efetiva deve ficar perto
de 2,4 GHz em todas as execuções.
