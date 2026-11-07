"""
Face detectors, grouped into sub-packages by the runtime executing their model.

    tf/        runs on tensorflow
    pytorch/   runs on pytorch
    onnx/      runs on onnxruntime
    opencv/    runs on opencv
    external/  runs inside a third party package

Detector and backend engine matrix. The backend engine is set by the DEEPFACE_BACKEND_ENGINE
environment variable, or defaults to the first installed one among tensorflow, pytorch and
onnxruntime.

    R - routed: the detector has an implementation per engine, and deepface picks the one
        of the current backend engine
    F - fixed: the detector has a single implementation in that column, and it is used
        whatever the backend engine is, as long as its requirement is installed

+--------------+----+---------+------+--------+----------+-----------------------------------+
| detector     | tf | pytorch | onnx | opencv | external | requirement                       |
+--------------+----+---------+------+--------+----------+-----------------------------------+
| mtcnn        | R  | R       | R    |        |          | tf: mtcnn (tensorflow extra)      |
|              |    |         |      |        |          | pytorch: facenet-pytorch          |
|              |    |         |      |        |          |   (pytorch extra)                 |
|              |    |         |      |        |          | onnx: base                        |
| retinaface   | R  | R       | R    |        |          | base - retina-face package routes |
|              |    |         |      |        |          |   to its tf, pytorch or onnx      |
|              |    |         |      |        |          |   model with the same environment |
|              |    |         |      |        |          |   variable                        |
| yolov8       |    | F       |      |        |          | ultralytics and torch             |
| yolov11      |    | F       |      |        |          | ultralytics and torch             |
| yolov12      |    | F       |      |        |          | ultralytics and torch             |
| yunet        |    |         | F    |        |          | base                              |
| centerface   |    |         | F    |        |          | base                              |
| opencv       |    |         |      | F      |          | base (haar cascade)               |
| ssd          |    |         |      | F      |          | base (caffe model, opencv < 5)    |
| dlib         |    |         |      |        | F        | dlib (hog)                        |
| mediapipe    |    |         |      |        | F        | mediapipe                         |
+--------------+----+---------+------+--------+----------+-----------------------------------+

The registry in deepface/modules/modeling.py (AVAILABLE_MODELS["face_detector"]) is the
source of truth, keep this matrix in sync with it.
"""
