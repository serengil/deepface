from pathlib import Path
import requests
import json

URL_VERIFY = "http://127.0.0.1:5005/verify"
HEADERS = {"Content-Type": "application/json"}

models = [
    # "VGG-Face",
    # "Facenet",
    # "Facenet512",
    # "OpenFace",
    # "DeepFace",
    # "DeepID",
    # "ArcFace",
    "Dlib",
    # "SFace",
    # "GhostFaceNet",
]

if __name__ == "__main__":
    result = {
        "results": [
            {
                "embedding": [
                    -1.2134509086608887,
                    -1.1512534618377686,
                ],
                "face_confidence": 0.92,
                "facial_area": {
                    "h": 998,
                    "left_eye": [1292, 1300],
                    "right_eye": [912, 1283],
                    "w": 998,
                    "x": 621,
                    "y": 906,
                },
            }
        ]
    }

    vec1 = result["results"][0]["embedding"]
    vec2 = vec1

    for model in models:
        payload = {
            "model_name": model,
            "img1_path": vec1,
            "img2_path": vec2,
        }

        response = requests.request(
            "POST", URL_VERIFY, data=json.dumps(payload), headers=HEADERS
        )

        print(response.text)
