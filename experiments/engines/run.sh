#!/usr/bin/env bash
# Runs the backend engine benchmark end to end. Every (engine, model) pair runs in its
# own python process, sequentially, so that runs do not compete for cpu.
#
#   FRAMEWORKS_ENV  conda env with deepface, tensorflow and pytorch installed
#                   e.g. pip install "deepface[tensorflow,pytorch]"
#   ONNX_ENV        conda env with deepface's base requirements only - pip install deepface
#   WORK_DIR        where inputs and raw results are stored
#
# usage: WORK_DIR=/tmp/engines ./run.sh && python report.py --work-dir /tmp/engines
set -euo pipefail

FRAMEWORKS_ENV="${FRAMEWORKS_ENV:?set FRAMEWORKS_ENV to a conda env with tensorflow and pytorch}"
ONNX_ENV="${ONNX_ENV:?set ONNX_ENV to a conda env with deepface base requirements only}"
WORK_DIR="${WORK_DIR:-./work}"
HERE="$(cd "$(dirname "$0")" && pwd)"

DATA="$WORK_DIR/data"
OUT="$WORK_DIR/out"
mkdir -p "$DATA" "$OUT"

run() {
  conda run --no-capture-output -n "$1" python "$HERE/benchmark.py" "${@:2}"
}

# label env engine kind model - skips a pair already benchmarked, so a run can be resumed
bench() {
  if [ -f "$OUT/res_$1_$3_$4_$5.json" ]; then
    echo "=== skip $1 $3 $4 $5"
    return
  fi
  echo ">>> $1 $3 $4 $5"
  run "$2" worker --engine "$3" --kind "$4" --model "$5" --data "$DATA" --out "$OUT" --label "$1"
}

[ -f "$DATA/pairs.csv" ] || run "$FRAMEWORKS_ENV" prepare --data "$DATA"

MODELS=(Facenet Facenet512 VGG-Face ArcFace)

# label:env:engine - onnx runs in an environment without tensorflow and pytorch, as a
# plain `pip install deepface` user would have it
TARGETS=(
  "full:$FRAMEWORKS_ENV:tensorflow"
  "full:$FRAMEWORKS_ENV:pytorch"
  "onnxonly:$ONNX_ENV:onnx"
)

for target in "${TARGETS[@]}"; do
  IFS=":" read -r label env engine <<< "$target"
  for model in "${MODELS[@]}"; do
    # the model alone, on the same input arrays for every engine
    bench "$label" "$env" "$engine" recognition "$model"
    # end to end represent with the opencv detector, so only the recognition model differs
    bench "$label" "$env" "$engine" pipeline "$model"
  done
done

run "$FRAMEWORKS_ENV" sizes --out "$OUT" --label full
run "$ONNX_ENV" sizes --out "$OUT" --label onnxonly
