"""
Turn the raw results of run.sh into markdown tables.

usage: python report.py --work-dir WORK_DIR > tables.md
"""

# built-in dependencies
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

# 3rd party dependencies
import numpy as np
import pandas as pd

# project dependencies
from deepface.config.threshold import thresholds

MODELS = ["Facenet", "Facenet512", "VGG-Face", "ArcFace"]
# (label, engine, title) - onnx runs in an environment without tensorflow and pytorch
COLUMNS = [
    ("full", "tensorflow", "TensorFlow"),
    ("full", "pytorch", "PyTorch"),
    ("onnxonly", "onnx", "ONNX"),
]
REFERENCE = ("full", "tensorflow")


def load(out_dir: str) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    results = {}
    for path in glob.glob(os.path.join(out_dir, "res_*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        results[(r["env"], r["engine"], r["model"], r["kind"])] = r
    return results


def table(header: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def fmt(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def metric_table(results, names, getter, digits=1, kind=None) -> str:
    rows = []
    for name in names:
        row = [name]
        k = kind or "recognition"
        for label, engine, _ in COLUMNS:
            r = results.get((label, engine, name, k))
            row.append(fmt(getter(r), digits) if r else "-")
        rows.append(row)
    return table(["model"] + [c[2] for c in COLUMNS], rows)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def recognition_parity(work_dir: str, prefix: str = "emb") -> str:
    out_dir = os.path.join(work_dir, "out")
    with open(os.path.join(work_dir, "data", "faces.json"), encoding="utf-8") as f:
        faces = json.load(f)
    index = {name: i for i, name in enumerate(faces)}
    pairs = pd.read_csv(os.path.join(work_dir, "data", "pairs.csv"))
    truth = (pairs["Decision"] == "Yes").values

    rows = []
    for model in MODELS:
        ref_path = os.path.join(out_dir, f"{prefix}_{REFERENCE[0]}_{REFERENCE[1]}_{model}.npy")
        ref = np.load(ref_path) if os.path.exists(ref_path) else None
        threshold = thresholds[model]["cosine"]
        for label, engine, title in COLUMNS:
            path = os.path.join(out_dir, f"{prefix}_{label}_{engine}_{model}.npy")
            if not os.path.exists(path):
                continue
            emb = np.load(path)
            distances = np.array(
                [
                    cosine(emb[index[x]], emb[index[y]])
                    for x, y in zip(pairs["file_x"], pairs["file_y"])
                ]
            )
            decisions = distances <= threshold
            accuracy = 100 * float((decisions == truth).mean())

            if ref is None or (label, engine) == REFERENCE:
                rows.append([model, title, "reference", "reference", "-", fmt(accuracy)])
                continue
            euclidean = np.linalg.norm(emb - ref, axis=1)
            ref_distances = np.array(
                [
                    cosine(ref[index[x]], ref[index[y]])
                    for x, y in zip(pairs["file_x"], pairs["file_y"])
                ]
            )
            agreement = 100 * float((decisions == (ref_distances <= threshold)).mean())
            rows.append(
                [
                    model,
                    title,
                    f"{euclidean.max():.2e}",
                    f"{np.abs(distances - ref_distances).max():.2e}",
                    fmt(agreement),
                    fmt(accuracy),
                ]
            )
    return table(
        [
            "model",
            "engine",
            "max euclidean vs TF",
            "max cosine-distance drift vs TF",
            "decision agreement vs TF (%)",
            "accuracy on 300 pairs (%)",
        ],
        rows,
    )


def sizes_section(out_dir: str) -> str:
    parts = []
    for path in sorted(glob.glob(os.path.join(out_dir, "sizes_*.json"))):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        rows = []
        for engine in ["tensorflow", "pytorch", "onnx"]:
            # an engine is reported from the environment it was benchmarked in
            if engine not in s or (s["env"], engine) not in [c[:2] for c in COLUMNS]:
                continue
            top = ", ".join(f"{k} {v:,.0f}" for k, v in list(s[engine]["packages"].items())[:5])
            rows.append(
                [engine, ", ".join(s[engine]["roots"]), fmt(s[engine]["total_mb"], 0), top]
            )
        parts.append(f"env `{s['env']}` (python {s['python']})\n\n")
        parts.append(
            table(["engine", "requirements", "added on disk (MB)", "largest packages (MB)"], rows)
        )
        parts.append("\n\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.join(args.work_dir, "out")
    results = load(out_dir)

    versions = {}
    for (label, engine, _, _), r in results.items():
        versions[(label, engine)] = (r["python"], r["framework_version"])
    print("## versions\n")
    print(
        table(
            ["engine", "python", "framework"],
            [[c[2], *versions.get(c[:2], ("-", "-"))] for c in COLUMNS],
        )
    )

    print("\n## sizes\n")
    print(sizes_section(out_dir))

    names = MODELS
    # the first run of an engine pays for a cold os file cache, the median is more telling
    imports: Dict[Tuple[str, str], List[float]] = {}
    for (label, engine, _, _), r in results.items():
        imports.setdefault((label, engine), []).append(r["import_s"])
    print("## framework import (s), median / min / max over all runs of an engine\n")
    print(
        table(
            ["engine", "median", "min", "max", "runs"],
            [
                [
                    c[2],
                    fmt(float(np.median(imports[c[:2]])), 2),
                    fmt(min(imports[c[:2]]), 2),
                    fmt(max(imports[c[:2]]), 2),
                    len(imports[c[:2]]),
                ]
                for c in COLUMNS
                if c[:2] in imports
            ],
        )
    )
    median_import = {k: float(np.median(v)) for k, v in imports.items()}
    print("\n## model build (s)\n")
    print(metric_table(results, names, lambda r: r["build_s"], 2))
    print("\n## first inference (s)\n")
    print(metric_table(results, names, lambda r: r["first_inference_s"], 3))
    print("\n## cold start = median import + build + first inference (s)\n")

    def cold(r):
        return median_import[(r["env"], r["engine"])] + r["build_s"] + r["first_inference_s"]

    print(metric_table(results, names, cold, 2))
    print("\n## recognition latency, batch 1, median (ms)\n")
    print(metric_table(results, MODELS, lambda r: r["single"]["median_ms"]))
    print("\n## recognition latency, batch 1, p90 (ms)\n")
    print(metric_table(results, MODELS, lambda r: r["single"]["p90_ms"]))
    print("\n## recognition throughput, batch 16 (faces/s)\n")
    print(metric_table(results, MODELS, lambda r: r["batch"]["faces_per_s"], 0))
    print("\n## memory after framework import (MB, RSS)\n")
    print(metric_table(results, MODELS[:1], lambda r: r["rss_after_import_mb"], 0))
    print("\n## memory after build (MB, RSS)\n")
    print(metric_table(results, names, lambda r: r["rss_after_build_mb"], 0))
    print("\n## peak memory (MB, max RSS)\n")
    print(metric_table(results, names, lambda r: r["peak_rss_mb"], 0))
    print("\n## implementations\n")
    for (label, engine, model, kind), r in sorted(results.items()):
        print(f"- {label} {engine} {kind} {model}: {r['implementation']}")

    print("\n## recognition parity\n")
    print(recognition_parity(args.work_dir))
    print("\n## end-to-end represent with opencv detector, median per image (ms)\n")
    print(metric_table(results, MODELS, lambda r: r["single"]["median_ms"], kind="pipeline"))
    print("\n## end-to-end cold start = median import + build + first represent (s)\n")
    print(
        metric_table(
            results,
            MODELS,
            cold,
            2,
            kind="pipeline",
        )
    )
    print("\n## end-to-end peak memory (MB)\n")
    print(metric_table(results, MODELS, lambda r: r["peak_rss_mb"], 0, kind="pipeline"))
    print("\n## end-to-end parity (opencv detector)\n")
    print(recognition_parity(args.work_dir, prefix="pipe"))


if __name__ == "__main__":
    main()
