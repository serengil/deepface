#from retinaface import RetinaFace #this is not a must dependency
import cv2

def build_model():
    from retinaface import RetinaFace
    face_detector = RetinaFace.build_model()
    return face_detector

def detect_face(face_detector, img, align = True):

    from retinaface import RetinaFace
    from retinaface.commons import postprocess

    #---------------------------------

    resp = []

    # The BGR2RGB conversion will be done in the preprocessing step of retinaface.
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

    """
    face = None
    img_region = [0, 0, img.shape[0], img.shape[1]] #Really?

    faces = RetinaFace.extract_faces(img_rgb, model = face_detector, align = align)

    if len(faces) > 0:
        face = faces[0][:, :, ::-1]

    return face, img_region
    """

    #--------------------------

    obj = RetinaFace.detect_faces(img, model = face_detector, threshold = 0.9)

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]

            y = facial_area[1]
            h = facial_area[3] - y
            x = facial_area[0]
            w = facial_area[2] - x
            img_region = [x, y, w, h]

            #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
            detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            if align:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                #mouth_right = landmarks["mouth_right"]
                #mouth_left = landmarks["mouth_left"]

                detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)

            resp.append((detected_face, img_region))

    return resp
