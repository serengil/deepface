from deepface.basemodels import Facenet
from pathlib import Path
import os
import gdown

def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5'):

    model = Facenet.InceptionResNetV2(dimension = 512)

    #-------------------------

    home = str(Path.home())

    if os.path.isfile(home+'/.deepface/weights/facenet512_weights.h5') != True:
        print("facenet512_weights.h5 will be downloaded...")

        output = home+'/.deepface/weights/facenet_weights.h5'
        gdown.download(url, output, quiet=False)

    #-------------------------

    model.load_weights(home+'/.deepface/weights/facenet512_weights.h5')

    #-------------------------

    return model
