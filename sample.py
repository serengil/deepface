import cv2
from deepface import DeepFace
import torch

def is_cuda_available():
    return torch.cuda.is_available()

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("カメラが見つかりません。")
        return

    use_gpu = is_cuda_available()
    print(f"GPU使用: {use_gpu}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できません。")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            result = DeepFace.analyze(
                frame_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
            )

            if isinstance(result, list):
                emotion = result[0].get('dominant_emotion', 'N/A')
            elif isinstance(result, dict):
                emotion = result.get('dominant_emotion', 'N/A')
            else:
                emotion = 'N/A'

            cv2.putText(frame, f'Emotion: {emotion}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, 'Emotion: N/A', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
