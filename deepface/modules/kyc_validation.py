import base64
import io
import os
import random
from typing import List
import asyncio


import cv2
from deepface import DeepFace
import numpy as np

detector_backend_extract = "retinaface"
detector_backend_verify = "retinaface"

def extract_frame_with_face(file: str) -> str:

  """
  Extracts frames with faces from a video as base64 bytes.
  This feature is for loading the selfie video, 
  and then extract the face of user, verify the face with user's face on ID card.
  only for use cases with one person in frame.

  Args:
  video: base64

  Returns:
  image base64 (str)
  """

  video_bytes = base64.b64decode(file)



  # Create a file-like object to write the video to.
  hash = str(random.getrandbits(128))
  temp_file = "./"+ hash
  with io.open(temp_file, "wb") as f:
    f.write(video_bytes)

  # Decode the video bytes.
  video_capture = cv2.VideoCapture(temp_file)
  os.remove(temp_file)


  
  # Extract frames with faces.
  frames_with_faces = []
  frame_counter = 0
  #detect every 30 frame
  face_capture_rate = 30
  #sample size to check wehther the video only have 1 person
  sample_size = 5

  while True:


    if sample_size <= len(frames_with_faces):
      break
    

    ret, frame = video_capture.read()
    if not ret:
       raise RuntimeError('No face detected')


    # Detect faces in the frame.
    if ((frame_counter % face_capture_rate == 0) or frame_counter == 0):
      faces = DeepFace.extract_faces(img_path=frame,
                                    detector_backend = detector_backend_extract,
                                    enforce_detection=False,)
  

      #If there is 1 face in the frame, add it to list and return
      if len(faces) == 1:
        # Encode the frame as a base64 string.
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        frames_with_faces.append(frame_base64)
      #If more than 1 face in frame, violate kyc rules, throw exception
      elif len(faces) > 1:
        raise RuntimeError('More than 1 person detected: len(faces) > 1')

       

    frame_counter = frame_counter + 1

  # Close the video capture.
  video_capture.release()
  """ Accuracy problem
  #if more then 1 frame have face, check whether frames of face are same person
  if len(frames_with_faces) > 1:
    result = asyncio.run(check_same_face(frames_with_faces))
    if result == False:
       raise RuntimeError('More than 1 person detected: check_same_face')
  """

  return frames_with_faces[0]


async def check_same_face(frames_with_faces: List[str]) -> bool:
    """
    check whether face in frame list are same person

    Args:
    list of base64 img: list[]

    Returns:
    All are same person return True, otherwise, Fales
    """
    pair_list = generate_pairs(frames_with_faces)

    tasks = [asyncio.create_task(call_verify_async(pair)) for pair in pair_list]

    results = await asyncio.gather(*tasks)
    print(str(results))
    return all(results)


async def call_verify_async(pairs:List[str]) -> bool:

    obj = DeepFace.verify(
    img1_path=pairs[0],
    img2_path=pairs[1],
    detector_backend = detector_backend_verify
    )
    return obj["verified"]


def generate_pairs(imgs:List[str]) -> List[List[str]]:
    """
    generate pairs of first img and all other imgs in the list,

    Args:
    list of base64 img: list[str]

    Returns:
    list of pair of img: list[list[str]
    """
    pairs = []
    for i in range(1, len(imgs)):
        pairs.append([imgs[0], imgs[i]])

    return pairs



   
