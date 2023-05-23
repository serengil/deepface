from deepface.detectors import FaceDetector


def build_model():
    import gdown
    import os

    from ultralytics import YOLO

    from deepface.commons.functions import get_deepface_home

    weights_path = f"{get_deepface_home()}/.deepface/weights/yolov8n-face.pt"

    if not os.path.isfile(weights_path):
        url = "https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb"
        gdown.download(url, weights_path, quiet=False)
        print("Downloaded YOLO model yolo8vn-face.pt")

    # return face_detector
    return YOLO(weights_path)


def detect_face(face_detector, img, align=False):
    resp = []

    results = face_detector.predict(img, verbose=False, show=True, conf=0.25)[0]

    for result in results:
        x, y, w, h = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]

        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        detected_face = img[y : y + h, x : x + w].copy()

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
