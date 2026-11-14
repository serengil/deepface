# TensorFlow vs PyTorch vs ONNX Runtime

deepface's models were converted from TensorFlow to PyTorch and ONNX, and the backend engine is now picked with the `DEEPFACE_BACKEND_ENGINE` environment variable (`tensorflow`, `pytorch` or `onnx`). This experiment compares the three engines on install size, cold start, inference latency, memory and output parity.

Models: `Facenet`, `Facenet512`, `VGG-Face` and `ArcFace`, measured alone and end to end with the `opencv` detector.

## TL;DR

Average over Facenet, Facenet512, VGG-Face and ArcFace (lower is better unless noted):

| criterion | TensorFlow | PyTorch | ONNX | best |
|---|---|---|---|---|
| install size (MB) | 1,009 | 597 | **127** | ONNX |
| framework import (s) | 3.85 | 1.55 | **0.04** | ONNX |
| cold start, model only (s) | 6.1 | 2.7 | **1.5** | ONNX |
| cold start, opencv + model (s) | 6.3 | 4.2 | **2.6** | ONNX |
| latency, one face (ms) | 122.8 | 242.3 | **31.8** | ONNX |
| throughput, batch of 16 (faces/s, higher is better) | 53.5 | 16.8 | **106.2** | ONNX |
| latency, opencv + model per image (ms) | 284.3 | 480.6 | **224.3** | ONNX |
| memory after import (MB) | 334 | 179 | **33** | ONNX |
| memory after model build (MB) | 607 | 490 | **343** | ONNX |
| peak memory, model only (MB) | 874 | **842** | 845 | PyTorch ≈ ONNX |
| peak memory, opencv + model (MB) | 1,833 | 1,829 | **1,662** | ONNX |
| output parity with TensorFlow | reference | identical | identical | tie |

- **The outputs match.** PyTorch and ONNX produce the same embeddings as TensorFlow, within 2.2e-05 euclidean distance, and all 300 verification decisions are identical on every engine. Changing the engine does not change results.
- **ONNX comes out ahead on every speed, size and memory row.** It is already a base requirement, so `pip install deepface` is all it needs.
- **VGG-Face drives the averages.** It has a 512 MB weight file and is the slowest model on every engine. It is the only model where ONNX's peak memory is the highest (2,259 MB, reached while the model loads), which is why the model-only peak row is a near tie.

## Setup

| | |
|---|---|
| machine | Intel Core i7-9750H (6 cores, 12 threads), 16 GB RAM, macOS, CPU only |
| TensorFlow and PyTorch | python 3.8.12, tensorflow 2.7.0, torch 2.1.2, opencv 4.9 - one environment with both frameworks installed |
| ONNX | python 3.10.18, onnxruntime 1.23.2, opencv 5.0 - a clean `pip install deepface` environment, without tensorflow or pytorch |

ONNX runs in its own environment because that is what a user who never installs TensorFlow or PyTorch gets. That environment also has a newer Python and opencv, which matters only for the end-to-end parity (see [Parity](#parity)).

How it was measured:

- Every (engine, model) pair runs in a **fresh Python process**, one after another. The weights were already downloaded.
- **Model only:** 25 faces from `tests/unit/dataset` (every image used in `master.csv`) were detected **once** with opencv, then resized and saved. Every engine gets exactly the same input arrays. The timer covers only `model.forward`: 5 warm-up calls, then 100 calls with a single face, and 10 calls with a batch of 16 faces.
- **End to end:** `DeepFace.represent(img, model_name, detector_backend="opencv")` on the raw images, 3 runs per image. The opencv detector is the same code on every engine, so only the recognition model changes.
- **Memory:** process RSS after the framework import and after the model is built, plus peak RSS (`ru_maxrss`).
- **Size:** the size on disk of every installed distribution that an engine's requirements pull in (from `requirements_tf.txt`, `requirements_pytorch.txt` or `onnxruntime`), excluding packages that deepface's base requirements already bring.

## Install size

On-disk size each engine adds on top of deepface's base requirements:

| engine | requirements | added on disk (MB) | largest packages (MB) |
|---|---|---|---|
| TensorFlow | tensorflow, keras, mtcnn | **1,009** | tensorflow 868, grpcio 40, libclang 32, tensorboard 10, h5py 9 |
| PyTorch | torch, facenet-pytorch | **597** | torch 526, sympy 44, networkx 12, torchvision 7, mpmath 4 |
| ONNX | onnxruntime | **127** | onnxruntime 71, sympy 49, mpmath 4, protobuf 2 |

## Cold start

Framework import time: median over the 8 runs of each engine. The first run of an engine reads its files from a cold OS file cache, which explains the max column.

| engine | median (s) | min (s) | max (s) |
|---|---|---|---|
| TensorFlow | 3.85 | 3.62 | 5.70 |
| PyTorch | 1.55 | 1.14 | 8.59 |
| ONNX | **0.04** | 0.03 | 0.07 |

Time to be ready for traffic = median import + model build + first inference (s):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 5.94 | 2.37 | **0.80** |
| Facenet512 | 6.96 | 2.35 | **0.55** |
| VGG-Face | 6.17 | **3.82** | 3.86 |
| ArcFace | 5.22 | 2.42 | **0.71** |

Build time alone (s):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 1.94 | 0.73 | 0.75 |
| Facenet512 | 2.92 | 0.72 | 0.50 |
| VGG-Face | 2.11 | 1.54 | 3.73 |
| ArcFace | 1.21 | 0.70 | 0.65 |

Building VGG-Face is slowest on ONNX, most likely because onnxruntime optimizes the graph of the 512 MB model while it creates the session.

## Inference latency

### Model only

One face per call, median (ms):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 121.2 | 63.7 | **8.5** |
| Facenet512 | 111.3 | 63.7 | **8.6** |
| VGG-Face | 148.5 | 702.5 | **86.6** |
| ArcFace | 110.4 | 139.2 | **23.7** |

One face per call, p90 (ms):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 150.1 | 64.6 | **9.0** |
| Facenet512 | 123.1 | 65.2 | **9.2** |
| VGG-Face | 153.2 | 733.5 | **93.6** |
| ArcFace | 122.5 | 147.5 | **27.2** |

Throughput with a batch of 16 (faces/s, higher is better):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 88 | 27 | **181** |
| Facenet512 | 89 | 27 | **187** |
| VGG-Face | 12 | 3 | **14** |
| ArcFace | 25 | 10 | **43** |

TensorFlow costs about 11 ms per face when batched (Facenet) but 121 ms for a single face. Most of a single-face call is fixed per-call overhead, so TensorFlow benefits from batching far more than the other engines.

### End to end with the opencv detector

`DeepFace.represent(img, model_name, detector_backend="opencv")` on the raw dataset images, median per image (ms). Images are already decoded in memory. The time covers opencv detection on images up to 2048 px, alignment, preprocessing and the forward pass:

| pipeline | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| opencv + Facenet | 263.4 | 317.4 | **167.6** |
| opencv + Facenet512 | 270.5 | 239.3 | **219.6** |
| opencv + VGG-Face | 324.8 | 1,010.1 | **318.2** |
| opencv + ArcFace | 278.4 | 355.7 | **191.7** |

The opencv detection step on these high-resolution images accounts for about 150–170 ms of every call on every engine, so the gaps are smaller than in the model-only tables.

Cold start, end to end = median import + build of the model and detector + first `represent` (s):

| pipeline | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| opencv + Facenet | 6.73 | 3.67 | **1.40** |
| opencv + Facenet512 | 6.67 | 3.87 | **1.48** |
| opencv + VGG-Face | **5.92** | 6.11 | 6.01 |
| opencv + ArcFace | 5.91 | 3.28 | **1.64** |

## Memory

RSS after the framework import (MB): TensorFlow 334, PyTorch 179, ONNX **33**.

RSS after the model is built (MB):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 459 | 300 | **161** |
| Facenet512 | 466 | 306 | **169** |
| VGG-Face | 911 | 912 | **762** |
| ArcFace | 590 | 440 | **278** |

Peak RSS, model only (MB):

| model | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| Facenet | 570 | 403 | **275** |
| Facenet512 | 572 | 406 | **279** |
| VGG-Face | **1,612** | 1,956 | 2,259 |
| ArcFace | 740 | 604 | **565** |

Peak RSS, end to end with opencv (MB):

| pipeline | TensorFlow | PyTorch | ONNX |
|---|---|---|---|
| opencv + Facenet | 1,701 | 1,539 | **1,534** |
| opencv + Facenet512 | 1,707 | 1,555 | **1,521** |
| opencv + VGG-Face | 2,185 | 2,606 | **2,036** |
| opencv + ArcFace | 1,740 | 1,614 | **1,558** |

For VGG-Face model only, ONNX has the highest peak even though it uses the least memory once loaded. The peak most likely comes from onnxruntime holding extra copies of the 512 MB of weights while it creates the session. End to end, most of the 1.5–2.6 GB peak comes from opencv detection on the high-resolution images, on every engine.

## Parity

Embeddings of PyTorch and ONNX compared with TensorFlow's on the 25 faces, plus verification decisions (cosine distance, deepface's default thresholds) on the 300 pairs in `master.csv`.

Model only, same input arrays for every engine:

| model | engine | max euclidean vs TF | max cosine distance drift vs TF | same decision as TF | accuracy |
|---|---|---|---|---|---|
| Facenet | PyTorch | 1.48e-05 | 4.24e-07 | 100% | 99.7% |
| Facenet | ONNX | 1.51e-05 | 6.18e-07 | 100% | 99.7% |
| Facenet512 | PyTorch | 2.18e-05 | 5.18e-07 | 100% | 98.3% |
| Facenet512 | ONNX | 2.14e-05 | 4.48e-07 | 100% | 98.3% |
| VGG-Face | PyTorch | 2.34e-06 | 5.14e-07 | 100% | 99.7% |
| VGG-Face | ONNX | 9.79e-07 | 2.17e-07 | 100% | 99.7% |
| ArcFace | PyTorch | 1.15e-05 | 1.07e-06 | 100% | 100% |
| ArcFace | ONNX | 8.01e-06 | 5.16e-07 | 100% | 100% |

TensorFlow's own accuracy is the same on every model (Facenet 99.7%, Facenet512 98.3%, VGG-Face 99.7%, ArcFace 100%).

End to end with opencv, PyTorch stays within 2.4e-05 euclidean distance of TensorFlow. ONNX drifts by up to 1.2e-01 (Facenet512). That drift comes from opencv 5.0 in the ONNX environment (vs 4.9) returning slightly different face boxes, not from onnxruntime. When the inputs are identical, as in the model-only table above, ONNX matches TensorFlow to within 2.2e-05. All 300 decisions are still identical to TensorFlow's on every engine.

## Caveats

- **This is a single laptop CPU.** Absolute numbers will differ on other hardware, and GPU results are not covered.
- **The torch build on this machine is handicapped.** The macOS x86 build of torch 2.1.2 has no OpenMP (`torch.__config__.parallel_info()` reports *OpenMP not found*, MKL max threads 1). PyTorch on Linux or Apple Silicon is likely faster than shown here, especially for VGG-Face.
- **The framework versions are not the latest** (TensorFlow 2.7, torch 2.1.2). Newer releases may behave differently.
- **Each pair ran once.** Latencies are the median of 100 calls, but import and build times are single samples, which is why import time is reported as the median over 8 runs.

## Reproduce

```shell
# FRAMEWORKS_ENV: a conda env with `pip install "deepface[tensorflow,pytorch]"`
# ONNX_ENV: a conda env with `pip install deepface` only
WORK_DIR=/tmp/engines FRAMEWORKS_ENV=<env> ONNX_ENV=<env> ./run.sh
python report.py --work-dir /tmp/engines
```

`run.sh` skips pairs that already have a result, so an interrupted run can be resumed. `report.py` prints every table in this document.
