# This is a test script to send a POST request to the API endpoint and get the response.
# Note: API has been brought up using the container of the docker image of the project.
# Note: this code has been tested on JPG and PNG files only.
# Note: Make sure to change the image_path and url variables to your own values.


# Required Libraries
import requests
import json
import base64

# image path
image_path = "path/to/image.jpg"

# API endpoint URL
url = "http://0.0.0.0:5000/represent"

# Read the image file and encode it as base64
with open(image_path, "rb") as image_file:
    encoded_string = "data:image/jpeg;base64," + \
        base64.b64encode(image_file.read()).decode("utf8")

# Prepare the payload data as JSON
payload = json.dumps({
    "model_name": "Facenet",
    "detector_backend": "mtcnn",
    "img_path": encoded_string
})

# Set the request headers
headers = {
    'Content-Type': 'application/json'
}

# Send a POST request to the API endpoint
response = requests.request("POST", url, headers=headers, data=payload)

# Print the response status code and decode it
print("-"*10)
if response.status_code == 200:
    print("\033[92mThe request was successful\033[0m")

    # Convert the response text to a dictionary
    response_dict = json.loads(response.text)

    # Print the response dictionary
    print(response_dict)

    print("And this is the embedding:",
          response_dict["results"][0]["embedding"])

else:
    print("\033[91mThe request was unsuccessful\033[0m")
    print("\033[91mError code:", response.status_code, "\033[0m")
    print("\033[91mError message:", response.text, "\033[0m")
