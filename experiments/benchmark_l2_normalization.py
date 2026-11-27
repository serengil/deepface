"""Compare L2 postprocessing with a saved baseline module.

Run from the repository root with NumPy available:
  git show <base>:deepface/modules/normalization.py > /tmp/normalization-before.py
  PYTHONPATH=. python experiments/benchmark_l2_normalization.py \
      --baseline /tmp/normalization-before.py

This measures normalization of already computed embeddings, not model inference.
"""

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from deepface.modules.normalization import normalize_embedding_l2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups must be nonnegative")

    spec = importlib.util.spec_from_file_location("normalization_before", args.baseline)
    if spec is None or spec.loader is None:
        parser.error("cannot load the baseline module")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    functions = {
        "baseline": baseline.normalize_embedding_l2,
        "patched": normalize_embedding_l2,
    }
    report = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "samples": args.samples,
        "warmups": args.warmups,
        "cases": [],
    }
    rng = np.random.default_rng(2741)
    for batch, dimension in ((1, 128), (64, 128), (64, 512), (512, 512),
                             (64, 4096), (512, 4096)):
        values = rng.normal(size=(batch, dimension)).tolist()
        source = values[0] if batch == 1 else values
        np.testing.assert_array_equal(
            functions["baseline"](source), functions["patched"](source)
        )
        timings = {name: [] for name in functions}
        for run in range(args.samples + args.warmups):
            order = list(functions) if run % 2 == 0 else list(reversed(functions))
            for name in order:
                start = time.perf_counter()
                functions[name](source)
                elapsed = (time.perf_counter() - start) * 1000
                if run >= args.warmups:
                    timings[name].append(elapsed)
        medians = {name: statistics.median(runs) for name, runs in timings.items()}
        result = {
            "batch": batch,
            "dimensions": dimension,
            "equal_outputs": True,
            "samples_ms": timings,
            "median_ms": medians,
            "baseline_over_patched": medians["baseline"] / medians["patched"],
        }
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
