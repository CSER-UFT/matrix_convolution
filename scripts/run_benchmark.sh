#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

profile="${1:---quick}"
case "$profile" in
  --quick) sizes=(64); distributions=(0); kernels=(3); blocks=(8); repetitions=2 ;;
  --full) sizes=(512 1024 2048); distributions=(0 1 2); kernels=(3 5 7); blocks=(8 16 32 64 128); repetitions=50 ;;
  *) echo "Uso: $0 [--quick|--full]" >&2; exit 2 ;;
esac

conv_linear=(conv_linear_double_full conv_linear_float_full conv_linear_double_skip_kernel conv_linear_float_skip_kernel)
conv_malloc=(conv_malloc_double_full conv_malloc_float_full conv_malloc_double_skip_kernel conv_malloc_float_skip_kernel)
gemm=(gemm_double_full gemm_float_full gemm_double_skip_k gemm_float_skip_k)
all_programs=("${conv_linear[@]}" "${conv_malloc[@]}" "${gemm[@]}")

for tool in perf taskset sha256sum python3 sudo; do
  command -v "$tool" >/dev/null || { echo "Erro: $tool nao encontrado." >&2; exit 1; }
done
for program in "${all_programs[@]}"; do
  [[ -x "bin/$program" ]] || { echo "Erro: bin/$program ausente. Execute make." >&2; exit 1; }
done
[[ -f inputs/SHA256SUMS ]] || { echo "Erro: gere as entradas fixas antes da campanha." >&2; exit 1; }
(cd inputs && sha256sum -c SHA256SUMS)

required=()
for precision in float double; do
  for n in "${sizes[@]}"; do
    for dist in "${distributions[@]}"; do required+=("inputs/$precision/matrix_n${n}_d${dist}.bin"); done
    required+=("inputs/$precision/gemm_n${n}.bin")
  done
  for k in "${kernels[@]}"; do required+=("inputs/$precision/kernel_k${k}.bin"); done
done
for input in "${required[@]}"; do [[ -f "$input" ]] || { echo "Erro: entrada ausente: $input" >&2; exit 1; }; done

allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
first_allowed=${allowed%%,*}; first_allowed=${first_allowed%%-*}
cpu="${TCC2_CPU:-$first_allowed}"
[[ "$cpu" =~ ^[0-9]+$ ]] || { echo "Erro: CPU invalida: $cpu" >&2; exit 1; }
taskset -c "$cpu" true || { echo "Erro: CPU $cpu nao permitida para esta tarefa." >&2; exit 1; }
governor_file="/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"
governor=$([[ -r "$governor_file" ]] && cat "$governor_file" || echo unknown)
if [[ "$governor" != performance && "$governor" != unknown ]]; then
  echo "Aviso: governador da CPU $cpu e '$governor'; a condicao sera registrada." >&2
fi

arch=( )
if command -v setarch >/dev/null && setarch "$(uname -m)" -R true 2>/dev/null; then
  arch=(setarch "$(uname -m)" -R)
else
  echo "Aviso: ASLR nao pode ser desativado; a campanha registrara essa condicao." >&2
fi

nmi_original=$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null || echo unknown)
nmi_changed=0
restore_nmi() {
  if [[ "$nmi_changed" == 1 ]]; then sudo -n sysctl -q -w "kernel.nmi_watchdog=$nmi_original" >/dev/null || true; fi
}
cleanup_runtime() {
  restore_nmi
}
trap cleanup_runtime EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$nmi_original" == 1 ]]; then
  if sudo -n sysctl -q -w kernel.nmi_watchdog=0 >/dev/null; then nmi_changed=1
  else echo "Aviso: NMI watchdog permaneceu ativo; a multiplexacao sera verificada." >&2
  fi
fi

P1=(instructions:u cycles:u cache-references:u cache-misses:u)
P2_CANDIDATES=(branches:u branch-misses:u L1-dcache-load-misses:u LLC-load-misses:u)
probe_event() { perf stat -e "$1" -- true >/dev/null 2>&1; }
for event in "${P1[@]}"; do probe_event "$event" || { echo "Erro: evento obrigatorio indisponivel: $event" >&2; exit 1; }; done
P2=()
for event in "${P2_CANDIDATES[@]}"; do
  if probe_event "$event"; then P2+=("$event"); else echo "Aviso: evento estendido indisponivel: $event" >&2; fi
done

join_by_comma() { local IFS=,; echo "$*"; }
p1_events=$(join_by_comma "${P1[@]}"),duration_time
p2_events=$(join_by_comma "${P2[@]}")

stamp=$(date +%Y%m%d_%H%M%S)
campaign="results/${profile#--}_$stamp"
raw="$campaign/raw"
mkdir -p "$raw"
manifest="$campaign/manifest.csv"
echo 'run_id,operation,program,precision,technique,N,dist,parameter,input,aux_input,repetition,p1_file,p2_file,p1_output,p2_output' > "$manifest"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "profile=$profile"
  echo "repetitions=$repetitions"
  echo "cpu=$cpu"
  echo "cpus_allowed=$allowed"
  echo "cpu_governor=$governor"
  echo "aslr_disabled=$([[ ${#arch[@]} -gt 0 ]] && echo yes || echo no)"
  echo "nmi_watchdog_original=$nmi_original"
  echo "nmi_watchdog_during_run=$(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null || echo unknown)"
  echo "perf_scope=whole_program"
  echo "p1_events=$p1_events"
  echo "p2_events=$p2_events"
  echo "input_manifest_sha256=$(sha256sum inputs/SHA256SUMS | awk '{print $1}')"
  uname -a
  gcc --version | head -n1
  perf --version
  lscpu
} > "$campaign/environment.txt"
cp inputs/SHA256SUMS "$campaign/input_SHA256SUMS"

check_perf() {
  local file="$1"
  ! grep -Eqi '<not counted>|<not supported>|not supported' "$file" || return 1
  awk -F';' 'NF >= 5 && $5 ~ /^[0-9.]+$/ && ($5 + 0) < 99.0 { bad=1 } END { exit bad }' "$file"
}

run_pass() {
  local events="$1" perf_file="$2" stdout_file="$3" stderr_file="$4"; shift 4
  local status
  set +e
  taskset -c "$cpu" "${arch[@]}" perf stat --no-big-num -x ';' \
    -e "$events" -o "$perf_file" -- "$@" >"$stdout_file" 2>"$stderr_file"
  status=$?
  set -e
  [[ $status -eq 0 ]] || { cat "$stderr_file" >&2; return "$status"; }
  check_perf "$perf_file" || { echo "Erro: contador ausente ou multiplexado em $perf_file" >&2; return 1; }
}

describe_program() {
  local program="$1"
  precision=$([[ "$program" == *float* ]] && echo float || echo double)
  if [[ "$program" == *skip_kernel* ]]; then technique=skip_kernel
  elif [[ "$program" == *skip_k ]]; then technique=skip_k
  else technique=full; fi
}

input_path() {
  local program="$1" filename="$2" input_precision
  input_precision=$([[ "$program" == *float* ]] && echo float || echo double)
  echo "inputs/$input_precision/$filename"
}

measure() {
  local operation="$1" program="$2" n="$3" dist="$4" parameter="$5" input="$6" aux="$7" repetition="$8"
  local id p1 p2 o1 o2 e1 e2
  describe_program "$program"
  id="${operation}__${program}__n${n}__d${dist}__p${parameter}__r$(printf '%03d' "$repetition")"
  p1="$raw/$id.p1"; p2="$raw/$id.p2"; o1="$raw/$id.p1.out"; o2="$raw/$id.p2.out"
  e1="$raw/$id.p1.err"; e2="$raw/$id.p2.err"
  if [[ "$operation" == gemm ]]; then
    run_pass "$p1_events" "$p1" "$o1" "$e1" "bin/$program" "$input" "$parameter"
    if [[ ${#P2[@]} -gt 0 ]]; then run_pass "$p2_events" "$p2" "$o2" "$e2" "bin/$program" "$input" "$parameter"; else : >"$p2"; cp "$o1" "$o2"; fi
  else
    run_pass "$p1_events" "$p1" "$o1" "$e1" "bin/$program" "$input" "$aux"
    if [[ ${#P2[@]} -gt 0 ]]; then run_pass "$p2_events" "$p2" "$o2" "$e2" "bin/$program" "$input" "$aux"; else : >"$p2"; cp "$o1" "$o2"; fi
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$id" "$operation" "$program" "$precision" "$technique" "$n" "$dist" "$parameter" "$input" "$aux" "$repetition" \
    "$p1" "$p2" "$o1" "$o2" >> "$manifest"
  sleep "${TCC2_PAUSE_SECONDS:-0.05}"
}

warmup() {
  local operation="$1" program="$2" input="$3" parameter="$4" aux="$5"
  if [[ "$operation" == gemm ]]; then taskset -c "$cpu" "${arch[@]}" "bin/$program" "$input" "$parameter" >/dev/null
  else taskset -c "$cpu" "${arch[@]}" "bin/$program" "$input" "$aux" >/dev/null; fi
}

echo "Campanha: $campaign"
echo "CPU: $cpu | repeticoes: $repetitions | P1: $p1_events | P2: ${p2_events:-desativada}"

for n in "${sizes[@]}"; do
  for dist in "${distributions[@]}"; do
    matrix_name="matrix_n${n}_d${dist}.bin"
    for k in "${kernels[@]}"; do
      kernel_name="kernel_k${k}.bin"
      for program in "${conv_linear[@]}"; do
        matrix=$(input_path "$program" "$matrix_name"); kernel=$(input_path "$program" "$kernel_name")
        warmup conv_linear "$program" "$matrix" "$k" "$kernel"
      done
      for program in "${conv_malloc[@]}"; do
        matrix=$(input_path "$program" "$matrix_name"); kernel=$(input_path "$program" "$kernel_name")
        warmup conv_malloc "$program" "$matrix" "$k" "$kernel"
      done
      for ((rep=1; rep<=repetitions; rep++)); do
        for offset in 0 1 2 3; do
          idx=$(((offset + rep - 1) % 4)); program="${conv_linear[$idx]}"
          matrix=$(input_path "$program" "$matrix_name"); kernel=$(input_path "$program" "$kernel_name")
          measure conv_linear "$program" "$n" "$dist" "$k" "$matrix" "$kernel" "$rep"
        done
        for offset in 0 1 2 3; do
          idx=$(((offset + rep - 1) % 4)); program="${conv_malloc[$idx]}"
          matrix=$(input_path "$program" "$matrix_name"); kernel=$(input_path "$program" "$kernel_name")
          measure conv_malloc "$program" "$n" "$dist" "$k" "$matrix" "$kernel" "$rep"
        done
      done
    done
  done
  gemm_name="gemm_n${n}.bin"
  for block in "${blocks[@]}"; do
    for program in "${gemm[@]}"; do
      gemm_input=$(input_path "$program" "$gemm_name")
      warmup gemm "$program" "$gemm_input" "$block" ""
    done
    for ((rep=1; rep<=repetitions; rep++)); do
      for offset in 0 1 2 3; do
        idx=$(((offset + rep - 1) % 4)); program="${gemm[$idx]}"
        gemm_input=$(input_path "$program" "$gemm_name")
        measure gemm "$program" "$n" -1 "$block" "$gemm_input" "" "$rep"
      done
    done
  done
done

python3 scripts/parse_results.py "$campaign"
python3 scripts/collect_accuracy.py "$campaign" "$profile"
echo "Campanha concluida: $campaign"
