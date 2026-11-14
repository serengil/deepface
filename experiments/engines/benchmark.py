"""
Benchmark deepface's tensorflow, pytorch and onnx backend engines against each other.

Every (engine, model) pair runs in a fresh python process, so import time, build time and
memory are measured from a clean start. Facial recognition models are measured twice:
- recognition: the model alone, fed with faces detected once by the `prepare` step, so every
  engine gets the very same input arrays
- pipeline: DeepFace.represent end to end with the opencv detector, which runs the same code
  on every engine, so only the facial recognition model differs

Steps:
    python benchmark.py prepare --data DIR
    python benchmark.py worker --engine onnx --kind recognition --model Facenet --data DIR --out DIR
    python benchmark.py sizes --out DIR --label full
    python report.py --work-dir DIR

See run.sh for the whole pipeline.
"""

# built-in dependencies
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Set

DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "unit", "dataset")

RECOGNITION_MODELS = {"Facenet": 160, "Facenet512": 160, "VGG-Face": 224, "ArcFace": 112}
PIPELINE_DETECTOR = "opencv"
WARMUP = 5
SINGLE_RUNS = 100
BATCH_SIZE = 16
BATCH_RUNS = 10
PIPELINE_REPEATS = 3


def rss_mb() -> float:
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        # keep the onnx-only environment free of extra packages
        import subprocess  # pylint: disable=import-outside-toplevel

        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out.strip()) / 1024


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macos, kilobytes on linux
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


# --------------------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------------------
def prepare(data_dir: str) -> None:
    """Detect faces once with opencv and store model-ready inputs for every engine"""
    import numpy as np
    import pandas as pd
    from deepface import DeepFace
    from deepface.modules import preprocessing

    os.makedirs(data_dir, exist_ok=True)
    pairs = pd.read_csv(os.path.join(DATASET, "master.csv"))
    images = sorted(set(pairs["file_x"]).union(pairs["file_y"]))

    faces = {}
    for name in images:
        face_objs = DeepFace.extract_faces(
            os.path.join(DATASET, name), detector_backend="opencv", enforce_detection=False
        )
        # rgb to bgr, as representation does
        faces[name] = face_objs[0]["face"][:, :, ::-1].astype(np.float32)

    for model_name, size in RECOGNITION_MODELS.items():
        batch = np.concatenate(
            [
                preprocessing.normalize_input(
                    preprocessing.resize_image(faces[name].copy(), (size, size)), "base"
                )
                for name in images
            ],
            axis=0,
        ).astype(np.float32)
        np.save(os.path.join(data_dir, f"faces_{model_name}.npy"), batch)

    with open(os.path.join(data_dir, "faces.json"), "w", encoding="utf-8") as f:
        json.dump(images, f)

    pairs.to_csv(os.path.join(data_dir, "pairs.csv"), index=False)
    print(f"prepared {len(images)} faces")


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------
def summarize(durations: List[float]) -> Dict[str, float]:
    ordered = sorted(durations)
    return {
        "median_ms": statistics.median(ordered) * 1000,
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))] * 1000,
        "mean_ms": statistics.fmean(ordered) * 1000,
    }


def worker(engine: str, kind: str, model_name: str, data_dir: str, out_dir: str, label: str):
    os.environ["DEEPFACE_BACKEND_ENGINE"] = engine
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    result: Dict[str, Any] = {
        "engine": engine,
        "kind": kind,
        "model": model_name,
        "env": label,
        "python": platform.python_version(),
        "rss_start_mb": rss_mb(),
    }

    import numpy as np  # pylint: disable=import-outside-toplevel

    # the framework import is a one-off cost every user pays at the first call
    tic = time.perf_counter()
    if engine == "tensorflow":
        import tensorflow as framework  # pylint: disable=import-outside-toplevel
    elif engine == "pytorch":
        import torch as framework  # pylint: disable=import-outside-toplevel
    else:
        import onnxruntime as framework  # pylint: disable=import-outside-toplevel
    result["framework_version"] = framework.__version__
    result["import_s"] = time.perf_counter() - tic
    result["rss_after_import_mb"] = rss_mb()

    from deepface.modules import modeling  # pylint: disable=import-outside-toplevel

    tic = time.perf_counter()
    model = modeling.build_model(task="facial_recognition", model_name=model_name)
    if kind == "pipeline":
        modeling.build_model(task="face_detector", model_name=PIPELINE_DETECTOR)
    result["build_s"] = time.perf_counter() - tic
    result["rss_after_build_mb"] = rss_mb()
    result["implementation"] = f"{type(model).__module__}.{type(model).__name__}"

    if kind == "recognition":
        inputs = np.load(os.path.join(data_dir, f"faces_{model_name}.npy"))
        assert tuple(model.input_shape) == inputs.shape[1:3], (model.input_shape, inputs.shape)

        tic = time.perf_counter()
        model.forward(inputs[:1])
        result["first_inference_s"] = time.perf_counter() - tic

        for i in range(WARMUP):
            model.forward(inputs[i : i + 1])

        durations = []
        for i in range(SINGLE_RUNS):
            j = i % len(inputs)
            tic = time.perf_counter()
            model.forward(inputs[j : j + 1])
            durations.append(time.perf_counter() - tic)
        result["single"] = summarize(durations)

        batch = inputs[:BATCH_SIZE]
        model.forward(batch)
        durations = []
        for _ in range(BATCH_RUNS):
            tic = time.perf_counter()
            model.forward(batch)
            durations.append(time.perf_counter() - tic)
        result["batch"] = summarize(durations)
        result["batch"]["faces_per_s"] = len(batch) / statistics.median(durations)

        # embeddings of every face, one by one, are kept for the parity check
        embeddings = np.array([model.forward(inputs[i : i + 1]) for i in range(len(inputs))])
        np.save(os.path.join(out_dir, f"emb_{label}_{engine}_{model_name}.npy"), embeddings)
    else:
        # end to end: raw image in, embedding out. detection is fixed to opencv, which runs
        # the very same code on every engine, so only the facial recognition model differs.
        import cv2  # pylint: disable=import-outside-toplevel
        from deepface import DeepFace  # pylint: disable=import-outside-toplevel

        with open(os.path.join(data_dir, "faces.json"), encoding="utf-8") as f:
            names = json.load(f)
        images = [cv2.imread(os.path.join(DATASET, name)) for name in names]

        def represent(img):
            return DeepFace.represent(
                img_path=img,
                model_name=model_name,
                detector_backend=PIPELINE_DETECTOR,
                enforce_detection=False,
            )

        tic = time.perf_counter()
        represent(images[0])
        result["first_inference_s"] = time.perf_counter() - tic

        for img in images[:WARMUP]:
            represent(img)

        durations = []
        embeddings = []
        for img in images:
            for repeat in range(PIPELINE_REPEATS):
                tic = time.perf_counter()
                objs = represent(img)
                durations.append(time.perf_counter() - tic)
                if repeat == 0:
                    embeddings.append(objs[0]["embedding"])
        result["single"] = summarize(durations)
        np.save(
            os.path.join(out_dir, f"pipe_{label}_{engine}_{model_name}.npy"), np.array(embeddings)
        )
    result["rss_end_mb"] = rss_mb()
    result["peak_rss_mb"] = peak_rss_mb()

    path = os.path.join(out_dir, f"res_{label}_{engine}_{kind}_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


# --------------------------------------------------------------------------------------
# sizes
# --------------------------------------------------------------------------------------
def closure(roots: List[str], stop: Optional[Set[str]] = None) -> Set[str]:
    """
    Installed distributions a list of distributions pulls in, recursively. Distributions
    in stop are neither collected nor traversed.
    """
    from importlib import metadata  # pylint: disable=import-outside-toplevel
    from packaging.requirements import Requirement  # pylint: disable=import-outside-toplevel
    from packaging.utils import canonicalize_name  # pylint: disable=import-outside-toplevel

    seen: Set[str] = set()
    stack = list(roots)
    while stack:
        name = canonicalize_name(stack.pop())
        if name in seen or name in (stop or set()):
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen.add(name)
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            stack.append(req.name)
    return seen


def dist_size_mb(name: str) -> float:
    from importlib import metadata  # pylint: disable=import-outside-toplevel

    dist = metadata.distribution(name)
    total = 0
    for file in dist.files or []:
        path = file.locate()
        if os.path.isfile(path):
            total += os.path.getsize(path)
    return total / 1024**2


def sizes(out_dir: str, label: str) -> None:
    """On disk footprint each backend engine adds on top of deepface's base requirements"""
    root = os.path.join(os.path.dirname(__file__), "..", "..")

    def read(file_name: str) -> List[str]:
        with open(os.path.join(root, file_name), encoding="utf-8") as f:
            lines = [line.split("#")[0].strip() for line in f]
        return [line.split(">")[0].split("=")[0].split("<")[0] for line in lines if line]

    base = read("requirements_base.txt")
    groups = {
        "tensorflow": read("requirements_tf.txt"),
        "pytorch": read("requirements_pytorch.txt"),
        "onnx": ["onnxruntime"],
    }
    # some base requirements, e.g. older retina-face releases, depend on an engine. engine
    # packages are kept out of the base set, so that they are attributed to their engine.
    engines = {"tensorflow", "tf-keras", "keras", "torch", "torchvision", "onnxruntime"}
    for roots in groups.values():
        engines.update(roots)
    base_closure = closure([p for p in base if p != "onnxruntime"], stop=engines)
    result: Dict[str, Any] = {"env": label, "python": platform.python_version()}
    for engine, roots in groups.items():
        dists = closure(roots)
        if not dists:
            continue
        extra = sorted(dists - base_closure)
        per_dist = {name: round(dist_size_mb(name), 1) for name in extra}
        result[engine] = {
            "roots": roots,
            "total_mb": round(sum(per_dist.values()), 1),
            "packages": dict(sorted(per_dist.items(), key=lambda kv: -kv[1])),
        }

    with open(os.path.join(out_dir, f"sizes_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "worker", "sizes"])
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--engine")
    parser.add_argument("--kind", choices=["recognition", "pipeline"])
    parser.add_argument("--model")
    parser.add_argument("--label", default="default")
    args = parser.parse_args()

    if args.step == "prepare":
        prepare(args.data)
    elif args.step == "worker":
        worker(args.engine, args.kind, args.model, args.data, args.out, args.label)
    else:
        sizes(args.out, args.label)
