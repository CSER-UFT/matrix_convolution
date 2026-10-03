#!/usr/bin/env bash
# Campanha de medição do TCC2: convolução 2D e GEMM blocada, completas e aproximadas.
#
# O que mudou em relação à campanha de 22/09:
#   * Só o kernel é medido. O programa cronometra a região de interesse e liga e
#     desliga o perf pelo modo --control, então tempo e contadores cobrem a mesma
#     região (antes: o programa inteiro, com leitura, alocação e checksum).
#   * Cada execução repete o kernel até somar TCC2_MIN_TIME segundos (mínimo de
#     TCC2_MIN_REPS repetições, aquecimento descartado) e informa a mediana.
#   * Energia: uma terceira passada do perf, em modo sistema (-a), lê os
#     contadores RAPL do pacote e da DRAM durante o mesmo lote de repetições.
#     A potência ociosa é medida no início e no fim da campanha.
#   * Novos eixos: grau de aproximação (step), tipo de kernel (rand, gauss, sobel)
#     e distribuição das matrizes do GEMM.
#   * Os binários são recompilados e validados no nó que executa a campanha
#     (-march=native corresponde à CPU usada de fato).
#   * O governor precisa ser "performance"; o estado do turbo, o núcleo, os
#     irmãos de SMT e o slot do Condor ficam registrados em environment.txt.
#   * O núcleo padrão é o último permitido (a CPU 0 costuma atender interrupções).
#   * A campanha pode ser retomada com --resume <diretório>.
#
# Uso:
#   scripts/run_benchmark.sh [--quick|--full] [--resume results/full_AAAAMMDD_HHMMSS]
#
# Permissões: os eventos :u exigem perf_event_paranoid <= 2; a energia (modo
# sistema) exige perf_event_paranoid <= 0 ou CAP_PERFMON. Ajuste antes da
# campanha (ver condor/campanha_full.sub).
#
# Variáveis de ambiente opcionais:
#   TCC2_CPU                núcleo usado (padrão: último núcleo permitido)
#   TCC2_REPETITIONS        execuções independentes por configuração
#   TCC2_MIN_TIME           segundos de kernel por execução nas passadas 1 e 2 (padrão 0.2)
#   TCC2_MIN_REPS           mínimo de repetições medidas por execução (padrão 3)
#   TCC2_ENERGY=0           desliga a passada de energia (fica registrado)
#   TCC2_ENERGY_MIN_TIME    segundos de kernel na passada de energia (padrão 1.0)
#   TCC2_IDLE_SECONDS       duração de cada medição de potência ociosa (padrão 20)
#   TCC2_ALLOW_GOVERNOR=1   aceita governor diferente de performance (fica registrado)
#   TCC2_MARCH              alvo do -march (padrão native)
#   TCC2_PAUSE_SECONDS      pausa entre execuções (padrão 0.05)
set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() { echo "Uso: $0 [--quick|--full] [--resume DIRETORIO_DA_CAMPANHA]" >&2; }

profile=--quick
resume=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick|--full) profile="$1"; shift ;;
    --resume) [[ $# -ge 2 ]] || { usage; exit 2; }; resume="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

case "$profile" in
  --quick)
    sizes=(64 128); conv_dists=(0 1); gemm_dists=(0 1); kernels=(3 5); kernel_types=(rand gauss sobel)
    conv_steps=(1 2); blocks=(8 16); gemm_steps=(1 2); repetitions=3; min_time=0.01; energy_min_time=0.05 ;;
  --full)
    sizes=(512 1024 2048); conv_dists=(0 1 2); gemm_dists=(0 1); kernels=(3 5 7); kernel_types=(rand gauss sobel)
    conv_steps=(1 2 3); blocks=(8 16 32 64 128); gemm_steps=(1 2 4); repetitions=10; min_time=0.2; energy_min_time=1.0 ;;
esac
repetitions="${TCC2_REPETITIONS:-$repetitions}"
min_time="${TCC2_MIN_TIME:-$min_time}"
min_reps="${TCC2_MIN_REPS:-3}"
energy_enabled="${TCC2_ENERGY:-1}"
energy_min_time="${TCC2_ENERGY_MIN_TIME:-$energy_min_time}"
idle_seconds="${TCC2_IDLE_SECONDS:-20}"
pause="${TCC2_PAUSE_SECONDS:-0.05}"
march="${TCC2_MARCH:-native}"

layouts=(conv_linear conv_malloc)
precisions=(double float)

# ---------------------------------------------------------------------------
# Ferramentas, compilação e entradas
# ---------------------------------------------------------------------------
for tool in perf taskset sha256sum python3 make gcc mkfifo; do
  command -v "$tool" >/dev/null || { echo "Erro: $tool nao encontrado." >&2; exit 1; }
done

echo "Compilando e validando no no de execucao ($(hostname))..."
make -B MARCH="$march" all >/dev/null
make MARCH="$march" validate

[[ -f inputs/SHA256SUMS ]] || { echo "Erro: gere as entradas com scripts/generate_inputs.py." >&2; exit 1; }
(cd inputs && sha256sum --quiet -c SHA256SUMS)

required=()
for precision in "${precisions[@]}"; do
  for n in "${sizes[@]}"; do
    for dist in "${conv_dists[@]}"; do required+=("inputs/$precision/matrix_n${n}_d${dist}.bin"); done
    for dist in "${gemm_dists[@]}"; do required+=("inputs/$precision/gemm_n${n}_d${dist}.bin"); done
  done
  for k in "${kernels[@]}"; do
    for type in "${kernel_types[@]}"; do required+=("inputs/$precision/kernel_${type}_k${k}.bin"); done
  done
done
for input in "${required[@]}"; do [[ -f "$input" ]] || { echo "Erro: entrada ausente: $input" >&2; exit 1; }; done

# ---------------------------------------------------------------------------
# Núcleo, governor, turbo, ASLR e NMI watchdog
# ---------------------------------------------------------------------------
expand_cpus() { # "0-3,8" -> "0 1 2 3 8"
  local part out=()
  IFS=, read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then out+=($(seq "${part%-*}" "${part#*-}")); else out+=("$part"); fi
  done
  echo "${out[@]}"
}
allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
read -ra allowed_cpus <<< "$(expand_cpus "$allowed")"
cpu="${TCC2_CPU:-${allowed_cpus[-1]}}"
[[ "$cpu" =~ ^[0-9]+$ ]] || { echo "Erro: CPU invalida: $cpu" >&2; exit 1; }
taskset -c "$cpu" true || { echo "Erro: CPU $cpu nao permitida para esta tarefa." >&2; exit 1; }

sysfs="/sys/devices/system/cpu"
read_or_unknown() { [[ -r "$1" ]] && cat "$1" || echo unknown; }
governor=$(read_or_unknown "$sysfs/cpu$cpu/cpufreq/scaling_governor")
if [[ "$governor" != performance && "${TCC2_ALLOW_GOVERNOR:-0}" != 1 ]]; then
  echo "Erro: governor da CPU $cpu e '$governor'. Use 'performance' ou defina TCC2_ALLOW_GOVERNOR=1" >&2
  echo "      (a condicao ficara registrada e devera ser discutida no texto)." >&2
  exit 1
fi
if [[ -r "$sysfs/intel_pstate/no_turbo" ]]; then
  turbo_state="intel_pstate/no_turbo=$(cat "$sysfs/intel_pstate/no_turbo")"
elif [[ -r "$sysfs/cpufreq/boost" ]]; then
  turbo_state="cpufreq/boost=$(cat "$sysfs/cpufreq/boost")"
else
  turbo_state=unknown
fi
smt_siblings=$(read_or_unknown "$sysfs/cpu$cpu/topology/thread_siblings_list")

arch=()
if command -v setarch >/dev/null && setarch "$(uname -m)" -R true 2>/dev/null; then
  arch=(setarch "$(uname -m)" -R)
else
  echo "Aviso: ASLR nao pode ser desativado; a condicao sera registrada." >&2
fi

nmi_original=$(read_or_unknown /proc/sys/kernel/nmi_watchdog)
nmi_changed=0
restore_nmi() {
  if [[ "$nmi_changed" == 1 ]]; then sudo -n sysctl -q -w "kernel.nmi_watchdog=$nmi_original" >/dev/null || true; fi
}
trap restore_nmi EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$nmi_original" == 1 ]] && command -v sudo >/dev/null; then
  if sudo -n sysctl -q -w kernel.nmi_watchdog=0 >/dev/null 2>&1; then nmi_changed=1
  else echo "Aviso: NMI watchdog permaneceu ativo; a multiplexacao sera verificada." >&2
  fi
fi

# ---------------------------------------------------------------------------
# Eventos do perf e suporte ao modo --control
# ---------------------------------------------------------------------------
P1=(instructions:u cycles:u cache-references:u cache-misses:u)
P2_CANDIDATES=(branches:u branch-misses:u L1-dcache-load-misses:u LLC-load-misses:u)
probe_event() { perf stat -e "$1" -- true >/dev/null 2>&1; }
for event in "${P1[@]}"; do
  probe_event "$event" || { echo "Erro: evento obrigatorio indisponivel: $event (veja perf_event_paranoid)" >&2; exit 1; }
done
P2=()
for event in "${P2_CANDIDATES[@]}"; do
  if probe_event "$event"; then P2+=("$event"); else echo "Aviso: evento estendido indisponivel: $event" >&2; fi
done
join_by_comma() { local IFS=,; echo "$*"; }
p1_events=$(join_by_comma "${P1[@]}")
p2_events=$(join_by_comma "${P2[@]}")

# Energia RAPL (modo sistema). Os dois domínios existem no Xeon E5 v2.
P3=()
if [[ "$energy_enabled" == 1 ]]; then
  for event in power/energy-pkg/ power/energy-ram/; do
    if perf stat -a -e "$event" -- true >/dev/null 2>&1; then P3+=("$event")
    else echo "Aviso: evento de energia indisponivel: $event" >&2; fi
  done
  if [[ ${#P3[@]} -eq 0 ]]; then
    echo "Erro: nenhum evento RAPL disponivel. Ajuste kernel.perf_event_paranoid para 0" >&2
    echo "      (ou rode com TCC2_ENERGY=0, ficando a energia como limitacao declarada)." >&2
    exit 1
  fi
fi
p3_events=$(join_by_comma "${P3[@]}")

probe_control() {
  local dir ctl ack ok=0
  dir=$(mktemp -d)
  mkfifo "$dir/ctl" "$dir/ack"
  exec {ctl}<>"$dir/ctl" {ack}<>"$dir/ack"
  perf stat --delay=-1 --control "fd:$ctl,$ack" -e instructions:u -o /dev/null -- true >/dev/null 2>&1 && ok=1
  exec {ctl}>&- {ack}>&-
  rm -rf "$dir"
  [[ "$ok" == 1 ]]
}
probe_control || { echo "Erro: este perf nao aceita '--delay=-1 --control fd:...' (requer perf 5.11 ou mais novo)." >&2; exit 1; }

# ---------------------------------------------------------------------------
# Diretório da campanha, manifesto e ambiente
# ---------------------------------------------------------------------------
header='run_id,operation,program,precision,technique,step,N,dist,parameter,kernel_type,input,aux_input,repetition,p1_file,p2_file,p3_file,p1_output,p2_output,p3_output'
declare -A done_runs=()
if [[ -n "$resume" ]]; then
  campaign="${resume%/}"
  [[ -f "$campaign/manifest.csv" ]] || { echo "Erro: $campaign/manifest.csv nao existe." >&2; exit 1; }
  while IFS=, read -r id _; do done_runs["$id"]=1; done < <(tail -n +2 "$campaign/manifest.csv")
  echo "Retomando $campaign (${#done_runs[@]} execucoes ja concluidas)."
else
  campaign="results/${profile#--}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$campaign/raw"
  echo "$header" > "$campaign/manifest.csv"
fi
raw="$campaign/raw"
manifest="$campaign/manifest.csv"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "resumed=$([[ -n "$resume" ]] && echo yes || echo no)"
  echo "hostname=$(hostname)"
  echo "condor_slot=${_CONDOR_SLOT:-none}"
  echo "condor_job_ad=${_CONDOR_JOB_AD:-none}"
  echo "profile=$profile"
  echo "repetitions=$repetitions"
  echo "min_time_per_execution=$min_time"
  echo "min_reps_per_execution=$min_reps"
  echo "energy_enabled=$energy_enabled"
  echo "energy_min_time_per_execution=$energy_min_time"
  echo "idle_seconds=$idle_seconds"
  echo "sizes=${sizes[*]}"
  echo "conv_dists=${conv_dists[*]} gemm_dists=${gemm_dists[*]}"
  echo "kernels=${kernels[*]} kernel_types=${kernel_types[*]} conv_steps=${conv_steps[*]}"
  echo "blocks=${blocks[*]} gemm_steps=${gemm_steps[*]}"
  echo "cpu=$cpu"
  echo "cpus_allowed=$allowed"
  echo "smt_siblings=$smt_siblings"
  echo "cpu_governor=$governor"
  echo "turbo=$turbo_state"
  echo "scaling_cur_freq_khz=$(read_or_unknown "$sysfs/cpu$cpu/cpufreq/scaling_cur_freq")"
  echo "scaling_max_freq_khz=$(read_or_unknown "$sysfs/cpu$cpu/cpufreq/scaling_max_freq")"
  echo "aslr_disabled=$([[ ${#arch[@]} -gt 0 ]] && echo yes || echo no)"
  echo "nmi_watchdog_original=$nmi_original"
  echo "nmi_watchdog_during_run=$(read_or_unknown /proc/sys/kernel/nmi_watchdog)"
  echo "perf_event_paranoid=$(read_or_unknown /proc/sys/kernel/perf_event_paranoid)"
  echo "loadavg=$(cat /proc/loadavg)"
  echo "perf_scope=kernel_only (perf stat --delay=-1 --control)"
  echo "p1_events=$p1_events"
  echo "p2_events=$p2_events"
  echo "p3_events=${p3_events:-desativada}"
  echo "input_manifest_sha256=$(sha256sum inputs/SHA256SUMS | awk '{print $1}')"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo none)"
  echo "--- compilador"
  make --no-print-directory MARCH="$march" compiler-info
  echo "--- binarios"
  (cd bin && sha256sum ./*)
  echo "--- sistema"
  uname -a
  perf --version
  lscpu
} > "$campaign/environment$([[ -n "$resume" ]] && echo "_resume_$(date +%Y%m%d_%H%M%S)").txt"
cp inputs/SHA256SUMS "$campaign/input_SHA256SUMS"
make --no-print-directory MARCH="$march" vecreport > "$campaign/vecreport.txt" 2>&1 || true

# ---------------------------------------------------------------------------
# Execução de uma passada do perf
# ---------------------------------------------------------------------------
check_perf() {
  local file="$1"
  ! grep -Eqi '<not counted>|<not supported>|not supported' "$file" || return 1
  awk -F';' 'NF >= 5 && $5 ~ /^[0-9.]+$/ && ($5 + 0) < 99.0 { bad=1 } END { exit bad }' "$file"
}

# run_pass eventos arquivo_perf stdout stderr escopo comando...
#   escopo: "task" (só o processo medido) ou "system" (-a, usado para RAPL)
run_pass() {
  local events="$1" perf_file="$2" stdout_file="$3" stderr_file="$4" scope="$5"; shift 5
  local dir ctl ack status scope_args=()
  [[ "$scope" == system ]] && scope_args=(-a)
  dir=$(mktemp -d "${TMPDIR:-/tmp}/tcc2ctl.XXXXXX")
  mkfifo "$dir/ctl" "$dir/ack"
  exec {ctl}<>"$dir/ctl" {ack}<>"$dir/ack"
  set +e
  TCC2_PERF_CTL_FD=$ctl TCC2_PERF_ACK_FD=$ack \
    taskset -c "$cpu" ${arch[@]+"${arch[@]}"} perf stat --no-big-num -x ';' \
      --delay=-1 --control "fd:$ctl,$ack" ${scope_args[@]+"${scope_args[@]}"} \
      -e "$events" -o "$perf_file" -- "$@" >"$stdout_file" 2>"$stderr_file"
  status=$?
  set -e
  exec {ctl}>&- {ack}>&-
  rm -rf "$dir"
  [[ $status -eq 0 ]] || { cat "$stderr_file" >&2; return "$status"; }
  check_perf "$perf_file" || { echo "Erro: contador ausente ou multiplexado em $perf_file" >&2; return 1; }
}

technique_of() { # operação, passo
  if [[ "$2" == 1 ]]; then echo full
  elif [[ "$1" == gemm ]]; then echo skip_k
  else echo skip_kernel; fi
}

# measure operação programa passo N dist parâmetro tipo_kernel entrada aux repetição
measure() {
  local operation="$1" program="$2" step="$3" n="$4" dist="$5" parameter="$6" ktype="$7"
  local input="$8" aux="$9" repetition="${10}"
  local id precision technique p1 p2 p3 o1 o2 o3 e1 e2 e3 second
  id="${operation}__${program}__s${step}__n${n}__d${dist}__p${parameter}__${ktype}__r$(printf '%03d' "$repetition")"
  [[ -z "${done_runs[$id]:-}" ]] || return 0
  precision=${program##*_}
  technique=$(technique_of "$operation" "$step")
  p1="raw/$id.p1"; p2="raw/$id.p2"; p3="raw/$id.p3"
  o1="raw/$id.p1.out"; o2="raw/$id.p2.out"; o3="raw/$id.p3.out"
  e1="$raw/$id.p1.err"; e2="$raw/$id.p2.err"; e3="$raw/$id.p3.err"
  if [[ "$operation" == gemm ]]; then second="$parameter"; else second="$aux"; fi
  local command=("bin/$program" "$input" "$second" --step "$step" --min-reps "$min_reps")
  run_pass "$p1_events" "$campaign/$p1" "$campaign/$o1" "$e1" task "${command[@]}" --min-time "$min_time"
  if [[ ${#P2[@]} -gt 0 ]]; then
    run_pass "$p2_events" "$campaign/$p2" "$campaign/$o2" "$e2" task "${command[@]}" --min-time "$min_time"
  else
    : > "$campaign/$p2"
    "${command[@]}" --min-time "$min_time" > "$campaign/$o2"
  fi
  if [[ ${#P3[@]} -gt 0 ]]; then
    run_pass "$p3_events" "$campaign/$p3" "$campaign/$o3" "$e3" system "${command[@]}" --min-time "$energy_min_time"
  else
    : > "$campaign/$p3"; cp "$campaign/$o1" "$campaign/$o3"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$id" "$operation" "$program" "$precision" "$technique" "$step" "$n" "$dist" "$parameter" "$ktype" \
    "$input" "$aux" "$repetition" "$p1" "$p2" "$p3" "$o1" "$o2" "$o3" >> "$manifest"
  sleep "$pause"
}

# Aquece o cache de páginas das entradas (o aquecimento do kernel é interno).
warmup() {
  taskset -c "$cpu" "$@" --reps 1 >/dev/null
}

# ---------------------------------------------------------------------------
# Laços da campanha. Em cada repetição, as variantes de uma configuração rodam
# em sequência com ordem rotacionada, para que tendências lentas da máquina não
# favoreçam sempre a mesma variante.
# ---------------------------------------------------------------------------
# Potência ociosa do pacote e da DRAM (máquina parada, perf em modo sistema).
measure_idle() {
  [[ ${#P3[@]} -gt 0 ]] || return 0
  taskset -c "$cpu" perf stat --no-big-num -x ';' -a -e "$p3_events" -o "$campaign/$1" -- sleep "$idle_seconds"
}

echo "Campanha: $campaign"
echo "CPU: $cpu | governor: $governor | turbo: $turbo_state | repeticoes: $repetitions | min_time: $min_time s"
echo "P1: $p1_events | P2: ${p2_events:-desativada} | P3 (energia): ${p3_events:-desativada}"
idle_tag=$(date +%Y%m%d_%H%M%S)
echo "Medindo potencia ociosa (${idle_seconds} s)..."
measure_idle "idle_start_$idle_tag.perf"

for n in "${sizes[@]}"; do
  for dist in "${conv_dists[@]}"; do
    matrix_name="matrix_n${n}_d${dist}.bin"
    for ktype in "${kernel_types[@]}"; do
      for k in "${kernels[@]}"; do
        kernel_name="kernel_${ktype}_k${k}.bin"
        variants=()
        for layout in "${layouts[@]}"; do
          for precision in "${precisions[@]}"; do
            for step in "${conv_steps[@]}"; do
              (( step == 1 || step < k )) && variants+=("${layout}_${precision}:$step")
            done
          done
        done
        for variant in "${variants[@]}"; do
          program=${variant%%:*}; precision=${program##*_}
          warmup "bin/$program" "inputs/$precision/$matrix_name" "inputs/$precision/$kernel_name"
        done
        count=${#variants[@]}
        for ((rep = 1; rep <= repetitions; rep++)); do
          for ((offset = 0; offset < count; offset++)); do
            variant="${variants[$(((offset + rep - 1) % count))]}"
            program=${variant%%:*}; step=${variant##*:}; precision=${program##*_}
            measure "${program%_*}" "$program" "$step" "$n" "$dist" "$k" "$ktype" \
              "inputs/$precision/$matrix_name" "inputs/$precision/$kernel_name" "$rep"
          done
        done
      done
    done
  done

  for dist in "${gemm_dists[@]}"; do
    gemm_name="gemm_n${n}_d${dist}.bin"
    for block in "${blocks[@]}"; do
      variants=()
      for precision in "${precisions[@]}"; do
        for step in "${gemm_steps[@]}"; do variants+=("gemm_${precision}:$step"); done
      done
      for precision in "${precisions[@]}"; do
        warmup "bin/gemm_$precision" "inputs/$precision/$gemm_name" "$block"
      done
      count=${#variants[@]}
      for ((rep = 1; rep <= repetitions; rep++)); do
        for ((offset = 0; offset < count; offset++)); do
          variant="${variants[$(((offset + rep - 1) % count))]}"
          program=${variant%%:*}; step=${variant##*:}; precision=${program##*_}
          measure gemm "$program" "$step" "$n" "$dist" "$block" none \
            "inputs/$precision/$gemm_name" "" "$rep"
        done
      done
    done
  done
done

echo "Medindo potencia ociosa final (${idle_seconds} s)..."
measure_idle "idle_end_$(date +%Y%m%d_%H%M%S).perf"

python3 scripts/parse_results.py "$campaign"
python3 scripts/collect_accuracy.py "$campaign" "$profile"
echo "Campanha concluida: $campaign"
