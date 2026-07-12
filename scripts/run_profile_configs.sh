#!/usr/bin/env bash
# Run `nsys profile` on cs336_systems/profile.py across a set of predefined
# model configs (small / medium / large / xl / 10B), matching the table from
# the assignment handout. Profiler results are written to the project's
# `results/` directory, named "{size}_ctx{context_length}" (nsys appends its
# own extension, e.g. small_ctx512.nsys-rep).
#
# Usage:
#   ./run_profile_configs.sh [--size SIZE]... [--context-length N]...
#                             [--vocab-size N] [--warmup-steps N] [--execution-steps N]
#                             [--autocast true|false] [--memory-profile true|false]
#                             [--pattern forward-only|full-training-step]
#                             [--output-dir DIR] [--trace ARG]
#                             [-- EXTRA_PROFILE_PY_ARGS]
#
# --size and --context-length can each be repeated or given as a
# comma-separated list. --size defaults to "all"; --context-length defaults
# to 512.
#
# --memory-profile true switches nsys_profile.py into memory_profile() mode
# (dumps a *_memory_snapshot.pickle instead of running the nvtx-annotated
# pass). Since that mode never emits the "benchmark_execution" NVTX range and
# nsys has nothing useful to capture, this script runs it directly via
# `uv run python ...` (no nsys wrapper, no .nsys-rep output). Otherwise it
# runs via `uv run nsys profile ... -- python ...` as before.
#
# Examples:
#   ./run_profile_configs.sh                                    # profile all sizes at ctx=512
#   ./run_profile_configs.sh --size small --context-length 512
#   ./run_profile_configs.sh --size small,medium --context-length 128,512
#   ./run_profile_configs.sh --size all --memory-profile true --pattern full-training-step
#   ./run_profile_configs.sh --size all -- --warmup-steps 3 --execution-steps 5

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
context_lengths=()
vocab_size=10000
warmup_steps=5
execution_steps=10
output_dir="../results"
trace="cuda,nvtx,osrt"
autocast="false"
memory_profile="false"
pattern="forward-only"
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
    --context-length)
      IFS=',' read -r -a split <<<"$2"
      context_lengths+=("${split[@]}")
      shift 2
      ;;
    --vocab-size)
      vocab_size="$2"
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
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --trace)
      trace="$2"
      shift 2
      ;;
    --autocast)
      autocast="$2"
      shift 2
      ;;
    --memory-profile)
      memory_profile="$2"
      shift 2
      ;;
    --pattern)
      pattern="$2"
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
if [[ ${#context_lengths[@]} -eq 0 ]]; then
  context_lengths=(512)
fi

mkdir -p "$output_dir"

echo "vocab_size=${vocab_size} warmup_steps=${warmup_steps} execution_steps=${execution_steps} trace=${trace}"
echo "autocast=${autocast} memory_profile=${memory_profile} pattern=${pattern}"
echo "Sizes to run: ${sizes[*]}"
echo "Context lengths: ${context_lengths[*]}"
echo "Output dir: ${output_dir}"
echo

failures=0
for size in "${sizes[@]}"; do
  if [[ -z "${CONFIGS[$size]+x}" ]]; then
    echo "Unknown size '$size'. Valid options: ${ORDER[*]} (or 'all')" >&2
    exit 1
  fi
  read -r d_model d_ff num_layers num_heads <<<"${CONFIGS[$size]}"

  for context_length in "${context_lengths[@]}"; do
    out_name="${size}_ctx${context_length}"
    out_path="${output_dir}/${out_name}"
    if [[ "$memory_profile" == "true" ]]; then
      echo "==> Profiling ${size} (ctx=${context_length}) [memory-profile mode, no nsys wrapper]"
    else
      echo "==> Profiling ${size} (ctx=${context_length}) -> ${out_path}"
    fi

    py_args=(
      --vocab_size "$vocab_size"
      --context_length "$context_length"
      --d_model "$d_model"
      --d_ff "$d_ff"
      --num_layers "$num_layers"
      --num_heads "$num_heads"
      --model_name "$size"
      --warmup-steps "$warmup_steps"
      --execution-steps "$execution_steps"
      --autocast "$autocast"
      --memory-profile "$memory_profile"
      --pattern "$pattern"
      "${extra_args[@]}"
    )

    set +e
    if [[ "$memory_profile" == "true" ]]; then
      # memory_profile() never emits the "benchmark_execution" NVTX range and
      # just dumps a *_memory_snapshot.pickle itself, so there is nothing for
      # nsys to usefully capture here: run the script directly instead.
      uv run python ../cs336_systems/nsys_profile.py "${py_args[@]}"
    else
      uv run nsys profile \
        -o "$out_path" \
        --force-overwrite=true \
        --capture-range=nvtx \
        --nvtx-capture=benchmark_execution \
        -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
        --capture-range-end=stop \
        --trace="$trace" \
        -- python ../cs336_systems/nsys_profile.py "${py_args[@]}"
    fi
    status=$?
    set -e

    if [[ $status -ne 0 ]]; then
      echo "[$size ctx=$context_length] FAILED (exit $status)" >&2
      failures=$((failures + 1))
    fi
    echo
  done
done

if [[ $failures -gt 0 ]]; then
  echo "$failures profiling run(s) failed." >&2
  exit 1
fi

echo "All profiling runs finished. Results written to ${output_dir}/"
