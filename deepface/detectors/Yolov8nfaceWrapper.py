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
    confidence = -1
    detected_face = None

    # if align:
    #     raise NotImplementedError("`align` is not implemented for Yolov8Wrapper")

    results = face_detector.predict(img, verbose=False, show=True, conf=0.25)[0]

    for result in results:
        x, y, w, h = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]
        # print(f"Confidence: {confidence}, x: {x}, y: {y}, w: {w}, h: {h}")

        # print landmarks
        print(result.keypoints.tolist())
        # change to top left corner, width, height
        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        detected_face = img[y : y + h, x : x + w].copy()
        resp.append((detected_face, [x, y, w, h], confidence))

    return resp
