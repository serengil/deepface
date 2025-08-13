# deepface/tools/ai_image_editor/detection.py

"""
Image processing utilities for AI Image Editor.
Some functions require TensorFlow and TensorFlow Hub — imports are lazy to avoid
forcing these dependencies on all DeepFace users.
"""

import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# COCO labels for object detection
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        import tensorflow_hub as hub
        model_url = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
        _model_cache = hub.load(model_url)
    return _model_cache


def detect(img, threshold=0.5):
    model = _load_model()
    image = np.array(img)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    import tensorflow as tf
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, 0)
    detection = model(image_tensor)

    boxes = detection['detection_boxes'][0].numpy()
    scores = detection['detection_scores'][0].numpy()
    classes = detection['detection_classes'][0].numpy().astype(np.int32)
    height, width = image.shape[:2]
    annotated_img = image.copy()

    for i in range(len(scores)):
        if scores[i] > threshold:
            class_id = classes[i]
            if class_id in COCO_LABELS:
                label = f"{COCO_LABELS[class_id]}: {scores[i]:.2f}"
                ymin = int(boxes[i][0] * height)
                xmin = int(boxes[i][1] * width)
                ymax = int(boxes[i][2] * height)
                xmax = int(boxes[i][3] * width)
                cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(annotated_img)


def resize(img, width, height):
    width = int(float(width))
    height = int(float(height))
    img = np.array(img)
    resized = cv2.resize(img, (width, height))
    return Image.fromarray(resized)


def rescale(img, scale_factor):
    if isinstance(scale_factor, (np.ndarray, list, tuple)):
        scale_factor = float(np.array(scale_factor).flatten()[0])
    else:
        scale_factor = float(scale_factor)
    img = np.array(img)
    height, width = img.shape[:2]
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    rescaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(rescaled)


def rotate(img, angle):
    img = np.array(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, M, (width, height))
    return Image.fromarray(rotated)


def masked_image(img, diameter=None):
    img = np.array(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    radius = int(diameter // 2) if diameter else min(height, width) // 4
    mask = cv2.circle(np.zeros((height, width), dtype=np.uint8), center, radius, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return Image.fromarray(masked)


def smart_blur(img, threshold=0.5, padding=20):
    model = _load_model()
    image = np.array(img)
    import tensorflow as tf
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, 0)
    detection = model(image_tensor)

    boxes = detection['detection_boxes'][0].numpy()
    scores = detection['detection_scores'][0].numpy()
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(scores)):
        if scores[i] > threshold:
            ymin = max(0, int(boxes[i][0] * height) - padding)
            xmin = max(0, int(boxes[i][1] * width) - padding)
            ymax = min(height, int(boxes[i][2] * height) + padding)
            xmax = min(width, int(boxes[i][3] * width) + padding)
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)

    inv_mask = cv2.bitwise_not(mask)
    blurred_img = cv2.GaussianBlur(image, (51, 51), 0)
    object_region = cv2.bitwise_and(image, image, mask=mask)
    background_region = cv2.bitwise_and(blurred_img, blurred_img, mask=inv_mask)
    result = cv2.add(object_region, background_region)
    return Image.fromarray(result)


def get_detected_object_types(img, threshold=0.5):
    model = _load_model()
    image = np.array(img)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    import tensorflow as tf
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, 0)
    detection = model(image_tensor)
    scores = detection['detection_scores'][0].numpy()
    classes = detection['detection_classes'][0].numpy().astype(np.int32)
    types = {COCO_LABELS[cls] for i, cls in enumerate(classes) if scores[i] > threshold and cls in COCO_LABELS}
    return sorted(types)


def object_remover(img, object_type, threshold=0.5, method='telea'):
    model = _load_model()
    image = np.array(img)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    import tensorflow as tf
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, 0)
    detection = model(image_tensor)

    boxes = detection['detection_boxes'][0].numpy()
    scores = detection['detection_scores'][0].numpy()
    classes = detection['detection_classes'][0].numpy().astype(np.int32)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(scores)):
        if scores[i] > threshold and COCO_LABELS.get(classes[i]) == object_type:
            ymin = int(boxes[i][0] * height)
            xmin = int(boxes[i][1] * width)
            ymax = int(boxes[i][2] * height)
            xmax = int(boxes[i][3] * width)
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)

    inpaint_method = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    inpainted = cv2.inpaint(image, mask, 3, inpaint_method)
    return Image.fromarray(inpainted)


def face_verify(img1, img2):
    try:
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        if img1_array.shape[-1] == 4:
            img1_array = img1_array[:, :, :3]
        if img2_array.shape[-1] == 4:
            img2_array = img2_array[:, :, :3]
        img2_resized = cv2.resize(img2_array, (img1_array.shape[1], img1_array.shape[0]))
        result = DeepFace.verify(
            img2_resized, img1_array,
            model_name="OpenFace",
            enforce_detection=False,
            distance_metric="cosine",
        )
        return result.get('verified', False)
    except Exception:
        return False
