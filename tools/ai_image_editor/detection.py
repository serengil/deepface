import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# ---------------------- FACE DETECTION ---------------------- #
def detect(img, threshold=0.9):
    try:
        detections = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
        result_img = np.array(img).copy()
        for det in detections:
            x, y, w, h = det['facial_area'].values()
            conf = det.get("confidence", 1.0)
            if conf >= threshold:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_img, f"{conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return Image.fromarray(result_img)
    except Exception as e:
        return f"Detection error: {str(e)}"

# ---------------------- FACE ANALYSIS ---------------------- #
def analyze_face(img):
    try:
        analysis = DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
        age = analysis[0]['age']
        gender = analysis[0]['gender']
        emotion = analysis[0]['dominant_emotion']
        race = analysis[0]['dominant_race']
        return f"Age: {age} | Gender: {gender} | Emotion: {emotion} | Race: {race}"
    except Exception as e:
        return f"Analysis error: {str(e)}"

# ---------------------- REMOVE OBJECTS ---------------------- #
def get_detected_object_types(img):
    # Simulating object detection class list (replace with actual model if needed)
    return ["person", "face", "car", "chair"]

def object_remover(img, object_type=None):
    try:
        objects_detected = get_detected_object_types(img)
        if object_type:
            if object_type.lower() == "face":
                detections = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
                mask = np.ones(np.array(img).shape, dtype=np.uint8) * 255
                for det in detections:
                    x, y, w, h = det['facial_area'].values()
                    mask[y:y+h, x:x+w] = 0
                processed_image = Image.fromarray(cv2.inpaint(np.array(img), cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 3, cv2.INPAINT_TELEA))
                return processed_image, objects_detected
            else:
                return img, objects_detected
        else:
            return img, objects_detected
    except Exception as e:
        return f"Remove object error: {str(e)}", []

# ---------------------- FACE VERIFICATION ---------------------- #
def face_verify(img1, img2, model_name="VGG-Face", threshold=0.4):
    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name=model_name, enforce_detection=False)
        verified = result.get("verified", False)
        distance = result.get("distance", None)

        # Save output images for display
        img1_display = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
        img2_display = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)

        return verified, distance, img1_display, img2_display
    except Exception as e:
        return False, None, None, None

# ---------------------- RESIZE ---------------------- #
def resize(img, width, height):
    try:
        return Image.fromarray(cv2.resize(np.array(img), (width, height)))
    except Exception as e:
        return f"Resize error: {str(e)}"

# ---------------------- RESCALE ---------------------- #
def rescale(img, scale_factor):
    try:
        img_arr = np.array(img)
        new_w = int(img_arr.shape[1] * scale_factor)
        new_h = int(img_arr.shape[0] * scale_factor)
        return Image.fromarray(cv2.resize(img_arr, (new_w, new_h)))
    except Exception as e:
        return f"Rescale error: {str(e)}"

# ---------------------- ROTATE ---------------------- #
def rotate(img, angle):
    try:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(img_arr, M, (w, h))
        return Image.fromarray(rotated)
    except Exception as e:
        return f"Rotate error: {str(e)}"

# ---------------------- MASKED IMAGE ---------------------- #
def masked_image(img, mask_diameter):
    try:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        cv2.circle(mask, center, mask_diameter // 2, (255, 255, 255), -1)
        masked = cv2.bitwise_and(img_arr, img_arr, mask=mask)
        return Image.fromarray(masked)
    except Exception as e:
        return f"Mask error: {str(e)}"

# ---------------------- SMART BLUR ---------------------- #
def smart_blur(img):
    try:
        img_arr = np.array(img)
        # Apply Gaussian blur for smart blurring
        blurred = cv2.GaussianBlur(img_arr, (15, 15), 0)
        return Image.fromarray(blurred)
    except Exception as e:
        return f"Smart blur error: {str(e)}"

# ---------------------- WEBCAM CAPTURE ---------------------- #
def webcam_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Webcam not available"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
