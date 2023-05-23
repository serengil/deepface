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
        # print(f"Confidence: {confidence}, x: {x}, y: {y}, w: {w}, h: {h}")

        # print landmarks

        # print(result.keypoints.tolist())
        # print(f"Left eye: {left_eye}, right eye: {right_eye}")

        # add eyes landmarks to img
        # import cv2

        # img = cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 2, (0, 0, 255), 2)
        # img = cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 2, (0, 255, 0), 2)
        # change to top left corner, width, height
        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        detected_face = img[y : y + h, x : x + w].copy()

        if align:
            left_eye, right_eye, _, _, _ = result.keypoints.tolist()
            # Check the landmarks confidence before alignment
            # print(f"Left eye: {left_eye[2]}, right eye: {right_eye[2]}")
            if left_eye[2] > 0.5 and right_eye[2] > 0.5:
                # print("Aligning face")
                # print(left_eye[:2], right_eye[:2])
                detected_face = FaceDetector.alignment_procedure(detected_face, left_eye[:2], right_eye[:2])

        resp.append((detected_face, [x, y, w, h], confidence))

    return resp
