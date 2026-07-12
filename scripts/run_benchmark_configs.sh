#!/usr/bin/env bash
# Run cs336_systems/benchmark.py across a set of predefined model configs
# (small / medium / large / xl / 10B), matching the table from the assignment handout.
#
# Usage:
#   ./run_benchmark_configs.sh [--size SIZE]... [--pattern PATTERN]
#                               [--warmup-steps N] [--execution-steps N]
#                               [--vocab-size N] [--context-length N]
#                               [--repeat-times N] [--autocast true|false]
#                               [-- EXTRA_BENCHMARK_ARGS]
#
# --size can be repeated or given as a comma-separated list; defaults to "all".
#
# Examples:
#   ./run_benchmark_configs.sh                                       # run all sizes
#   ./run_benchmark_configs.sh --size small --pattern forward-and-backward
#   ./run_benchmark_configs.sh --size small --size medium
#   ./run_benchmark_configs.sh --size small,medium,large
#   ./run_benchmark_configs.sh --size all --repeat-times 20

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# name -> "d_model d_ff num_layers num_heads"
declare -A CONFIGS=(
  [small]="768 3072 12 12"
  [medium]="1024 4096 24 16"
  [large]="1280 5120 36 20"
  [xl]="2560 10240 32 32"
  [10B]="4608 12288 50 36"
)
ORDER=(small medium large xl 10B)

sizes=()
pattern="forward-only"
warmup_steps=5
execution_steps=10
vocab_size=10000
context_length=512
repeat_times=10
autocast="True"
extra_args=()

usage() {
  grep '^#' "$0" | sed 's/^#//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      IFS=',' read -r -a split <<<"$2"
      sizes+=("${split[@]}")
      shift 2
      ;;
    --pattern)
      pattern="$2"
      shift 2
      ;;
    --warmup-steps)
      warmup_steps="$2"
      shift 2
      ;;
    --execution-steps)
      execution_steps="$2"
      shift 2
      ;;
    --vocab-size)
      vocab_size="$2"
      shift 2
      ;;
    --context-length)
      context_length="$2"
      shift 2
      ;;
    --repeat-times)
      repeat_times="$2"
      shift 2
      ;;
    --autocast)
      autocast="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ ${#sizes[@]} -eq 0 || "${sizes[0]}" == "all" ]]; then
  sizes=("${ORDER[@]}")
fi

echo "Pattern=${pattern} warmup_steps=${warmup_steps} execution_steps=${execution_steps} repeat_times=${repeat_times} autocast=${autocast}"
echo "Sizes to run: ${sizes[*]}"
echo

printf "%-8s %-9s %-8s %-11s %-10s %-15s %-15s\n" "size" "d_model" "d_ff" "num_layers" "num_heads" "mean(sec)" "std(sec)"

for size in "${sizes[@]}"; do
  if [[ -z "${CONFIGS[$size]+x}" ]]; then
    echo "Unknown size '$size'. Valid options: ${ORDER[*]} (or 'all')" >&2
    exit 1
  fi
  read -r d_model d_ff num_layers num_heads <<<"${CONFIGS[$size]}"

  set +e
  output=$(uv run python ../cs336_systems/benchmark.py \
    --vocab_size "$vocab_size" \
    --context_length "$context_length" \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --num_layers "$num_layers" \
    --num_heads "$num_heads" \
    --pattern "$pattern" \
    --warmup-steps "$warmup_steps" \
    --execution-steps "$execution_steps" \
    --repeat-times "$repeat_times" \
    --autocast "$autocast" \
    "${extra_args[@]}" 2>&1)
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "[$size] FAILED (exit $status):" >&2
    echo "$output" >&2
    printf "%-8s %-9s %-8s %-11s %-10s %-15s %-15s\n" "$size" "$d_model" "$d_ff" "$num_layers" "$num_heads" "ERROR" "ERROR"
    continue
  fi

  # benchmark.py prints: "Latency: <mean> seconds ± <std> seconds"
  mean="N/A"
  std="N/A"
  if [[ "$output" =~ Latency:\ ([0-9.eE+-]+)\ seconds\ ±\ ([0-9.eE+-]+)\ seconds ]]; then
    mean="${BASH_REMATCH[1]}"
    std="${BASH_REMATCH[2]}"
  fi
  printf "%-8s %-9s %-8s %-11s %-10s %-15s %-15s\n" "$size" "$d_model" "$d_ff" "$num_layers" "$num_heads" "$mean" "$std"
done
