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
| Governor e núcleo podiam variar entre campanhas. | Usa `powersave` como condição fixa e prende o processo à CPU 3; registra governador, frequência, turbo, slot do Condor e carga da máquina. |
| Binário compilado fora do nó de execução. | `run_benchmark.sh` recompila e valida no próprio nó. |
| Falha no meio perdia a campanha. | `--resume <diretório>`. |
| Gráficos com legendas dentro dos eixos, tapando dados. | `generate_plots.py` reescrito: legenda única abaixo dos painéis, fora da área de desenho; cores validadas para daltonismo; marcador e estilo de linha redundantes à cor. |

## Verificação do servidor antes da campanha

Execute os comandos desta seção na raiz do projeto. Não submeta a campanha
completa enquanto o teste `--quick` não terminar corretamente.

### 1. Ferramentas e versões

```bash
command -v gcc make python3 perf taskset sha256sum mkfifo condor_submit
gcc --version | head -n1
perf --version
condor_version | head -n1
python3 -c "import numpy, pandas, matplotlib, scipy; print('Dependencias Python: OK')"
```

Todos os caminhos e a confirmação das dependências Python precisam ser
exibidos. Se a importação falhar, prepare o ambiente antes de continuar:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Esta versão do pipeline requer um
`perf` que aceite `--delay=-1` e `--control` (referência: `perf` 5.11 ou mais
novo). Confira se as opções aparecem:

```bash
perf stat -h 2>&1 | grep -E -- '--control|--delay'
```

Confirme também o acesso aos contadores obrigatórios de instruções e ciclos:

```bash
perf stat -e instructions:u,cycles:u -- true
```

O `run_benchmark.sh` faz uma prova adicional com a sintaxe exata de
`--delay=-1 --control fd:...` e interrompe imediatamente se ela não for
suportada.

### 2. Governador, permissões e NMI watchdog

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
taskset -c 3 true && echo "CPU 3 permitida: OK"
cat /proc/sys/kernel/perf_event_paranoid
cat /proc/sys/kernel/nmi_watchdog
```

Condição esperada para a campanha definitiva:

```text
governador: powersave
CPU usada pelo benchmark: 3
kernel.perf_event_paranoid: 0
kernel.nmi_watchdog: 0
```

O governador já é `powersave` no Saruê e não precisa ser alterado. Se
necessário, o administrador configura somente as permissões dos contadores:

```bash
sudo sysctl -w kernel.perf_event_paranoid=0
sudo sysctl -w kernel.nmi_watchdog=0
```

O script confirma que a CPU 3 está em `powersave` e interrompe a campanha se a
condição for diferente. O valor zero de `perf_event_paranoid` permite a medição
de energia em modo sistema. O NMI watchdog desligado libera um contador de
hardware e reduz o risco de multiplexação.

A afinidade com a CPU 3 é necessária para impedir que o sistema migre o
processo entre núcleos durante uma medição. Isso reduz uma fonte de variação e
mantém todos os programas sob a mesma condição. A variável `TCC2_CPU` só é
necessária se o projeto for executado em outra máquina onde a CPU 3 não esteja
disponível.

### 3. Energia RAPL

Liste os eventos de energia reconhecidos:

```bash
perf list | grep -E 'energy-(pkg|ram)'
```

Teste cada domínio disponível:

```bash
perf stat -a -e power/energy-pkg/ -- sleep 1
perf stat -a -e power/energy-ram/ -- sleep 1
```

O script precisa encontrar pelo menos um evento RAPL. O Xeon do sarue deve
oferecer energia do pacote e pode oferecer energia da DRAM. A medição fica
sempre ativada pelo próprio código: não é necessário definir nenhuma variável
antes do `--quick` ou do `--full`.

### 4. Entradas, compilação e validação

Use `all`, e não `full`, para gerar tanto as entradas do teste rápido
(`N = 64` e `128`) quanto as da campanha completa (`N = 512`, `1024` e `2048`):

```bash
python3 scripts/generate_inputs.py all
(cd inputs && sha256sum -c SHA256SUMS)
chmod +x scripts/run_benchmark.sh
make clean
make validate
make vecreport
```

Os seis validadores precisam terminar com `OK:`. O relatório de vetorização é
gravado em `bin/vecreport.txt`.

## Teste rápido obrigatório

Execute o pipeline completo com tamanhos pequenos antes de usar o Condor:

```bash
timeout 30m scripts/run_benchmark.sh --quick
```

O teste foi concluído somente se aparecer `Campanha concluida:` e o diretório
mais recente contiver manifesto, ambiente e CSVs consolidados:

```bash
QUICK_DIR=$(find results -maxdepth 1 -type d -name 'quick_*' | sort | tail -n1)
echo "$QUICK_DIR"
test -n "$QUICK_DIR"
test -f "$QUICK_DIR/manifest.csv"
test -f "$QUICK_DIR/environment.txt"
test -f "$QUICK_DIR/results_raw.csv"
test -f "$QUICK_DIR/results_summary.csv"
test -f "$QUICK_DIR/comparisons_raw.csv"
test -f "$QUICK_DIR/comparisons_summary.csv"
test -f "$QUICK_DIR/accuracy.csv"
grep -E '^cpu=|perf version|cpu_governor|perf_event_paranoid|p3_events|perf_scope' \
  "$QUICK_DIR/environment.txt"
```

O código de saída `124` do comando `timeout` indica que o teste excedeu 30
minutos ou ficou bloqueado. Nesse caso, não execute o `full` antes de examinar a
última mensagem e os arquivos `.err` da campanha.

### Diagnóstico de incompatibilidades

| Mensagem ou comportamento | Causa provável | Ação |
|---|---|---|
| `nao aceita '--delay=-1 --control'` | `perf` antigo ou sem suporte ao protocolo de controle | Disponibilizar um `perf` compatível; o Condor não corrige essa limitação. |
| `evento obrigatorio indisponivel` | permissão do `perf` ou contador indisponível | Conferir `perf_event_paranoid` e testar `instructions:u` e `cycles:u` manualmente. |
| `nenhum evento RAPL disponivel` | permissão, módulo ou domínio de energia ausente | Conferir `perf list`, testar `energy-pkg`/`energy-ram` e consultar o administrador. |
| `a campanha exige 'powersave'` | A CPU 3 está usando outro governador | Não execute a campanha até o servidor voltar à condição padrão `powersave`. |
| `CPU 3 nao permitida` | O job não recebeu acesso à CPU 3 | Conferir a alocação do Condor; em outra máquina, escolher uma CPU fixa com `TCC2_CPU`. |
| `entrada ausente` | foram geradas somente entradas `full` ou o manifesto está incompleto | Executar `python3 scripts/generate_inputs.py all` e validar `SHA256SUMS`. |
| `contador ausente ou multiplexado` | NMI watchdog, eventos em excesso ou disputa por contadores | Desligar o NMI watchdog e repetir o `--quick`. |
| teste sem avanço até o `timeout` | falha na confirmação entre o programa e `perf --control` | Não iniciar o `full`; guardar terminal e `.err` para análise. |

## Campanha completa pelo Condor

Somente depois que o `--quick` e as verificações acima passarem:

```bash
condor_submit condor/campanha_full.sub
condor_q
```

Os arquivos de acompanhamento são gravados em `condor/campanha_<Cluster>.out`,
`.err` e `.log`. O Condor evita que a campanha dependa da sessão SSH, mas não
impede processos interativos de outros usuários. Rode em horário ocioso: o
RAPL mede o pacote inteiro e qualquer outra carga entra na energia observada.

Para retomar uma campanha interrompida, edite temporariamente `arguments` em
`condor/campanha_full.sub` conforme o exemplo comentado no próprio arquivo.

Depois da campanha:

```bash
make plots DATA=results/full_AAAAMMDD_HHMMSS FIGURES=results/figuras_full
```

(ou, com o pacote `.tar.gz` da campanha, `prepare_analysis_data.py` e depois
`make plots`, que lê `results/analysis_data` por padrão).

Variáveis úteis: `TCC2_REPETITIONS`, `TCC2_MIN_TIME`, `TCC2_MIN_REPS`,
`TCC2_ENERGY_MIN_TIME`, `TCC2_IDLE_SECONDS` e `TCC2_CPU`. Todas descritas no
topo do `run_benchmark.sh`. A energia já fica ativada e a CPU padrão já é 3.

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
ela mostra a estabilidade e a variação da frequência efetiva com o governador
`powersave`, que deve permanecer igual em toda a campanha.
