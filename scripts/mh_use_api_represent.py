import base64
from pathlib import Path
import requests
import json

URL_REPRESENT = "http://127.0.0.1:5005/represent"
HEADERS = {"Content-Type": "application/json"}


def get_base64_from_file(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"


DATA_DIR = Path("/Users/markus/Development/theacare/deepface/data")

if __name__ == "__main__":
    image_path = (
        DATA_DIR
        / "SELFIE_ID_0_20-0001c8a62e--61a634f7b4827531ac65c81c/271_sets_03_12_21__0001c8a62e--61a634f7b4827531ac65c81c_age_40_name_Daria__Selfie_2.jpg"
    )
    b64encoded_string = get_base64_from_file(image_path)

    payload = {"model_name": "Facenet", "img": b64encoded_string}

    # img_path = "https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/tests/dataset/couple.jpg"
    # payload = {"model_name": "Facenet", "img": img_path}

    response = requests.request(
        "POST", URL_REPRESENT, data=json.dumps(payload), headers=HEADERS
    )

    print(response.text)
