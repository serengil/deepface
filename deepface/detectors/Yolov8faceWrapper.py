from deepface.detectors import FaceDetector

# Models names and paths
PATHS = {
    # The commented models are broken, their weights are available
    # but the models themselves are not working
    # See https://github.com/derronqi/yolov8-face/issues/3 for details
    # "yolov8-lite-t": "/.deepface/weights/yolov8-lite-t.pt",
    # "yolov8-lite-s": "/.deepface/weights/yolov8-lite-s.pt",
    "yolov8n": "/.deepface/weights/yolov8n-face.pt",
}

# Google Drive base URL
BASE_URL = "https://drive.google.com/uc?id="

# Models' Google Drive IDs
IDS = {
    # The commented models are broken, their weights are available
    # but the models themselves are not working
    # See https://github.com/derronqi/yolov8-face/issues/3 for details
    # "yolov8-lite-t": "1vFMGW8xtRVo9bfC9yJVWWGY7vVxbLh94",
    # "yolov8-lite-s": "1ckpBT8KfwURTvTm5pa-cMC89A0V5jbaq",
    "yolov8n": "1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
}


def build_model(model: str):
    """Function factory for YOLO models"""
    from deepface.commons.functions import get_deepface_home

    # Get model's weights path and Google Drive URL
    func_weights_path = f"{get_deepface_home()}{PATHS[model]}"
    func_url = f"{BASE_URL}{IDS[model]}"

    # Define function to build the model
    def _build_model(weights_path: str = func_weights_path, url: str = func_url):
        import gdown
        import os

        # Import the Ultralytics YOLO model
        from ultralytics import YOLO

        # Download the model's weights if they don't exist
        if not os.path.isfile(weights_path):
            gdown.download(url, weights_path, quiet=False)
            print(f"Downloaded YOLO model {os.path.basename(PATHS[model])}")

        # Return face_detector
        return YOLO(weights_path)

    return _build_model


def detect_face(face_detector, img, align=False):
    resp = []

    # Detect faces
    results = face_detector.predict(
        img, verbose=False, show=False, conf=0.25)[0]

    # For each face, extract the bounding box, the landmarks and confidence
    for result in results:
        # Extract the bounding box and the confidence
        x, y, w, h = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]

        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        detected_face = img[y: y + h, x: x + w].copy()

        if align:
            # Extract landmarks
            left_eye, right_eye, _, _, _ = result.keypoints.tolist()
            # Check the landmarks confidence before alignment
            if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                detected_face = FaceDetector.alignment_procedure(
                    detected_face, left_eye[:2], right_eye[:2]
                )

        resp.append((detected_face, [x, y, w, h], confidence))

    return resp
