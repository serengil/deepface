import os
import bz2
import gdown
import numpy as np
from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger(module="basemodels.DlibResNet")

# pylint: disable=too-few-public-methods


class DlibResNet:
    def __init__(self):

        # this is not a must dependency
        import dlib  # 19.20.0

        self.layers = [DlibMetaData()]

        # ---------------------

        home = functions.get_deepface_home()
        weight_file = home + "/.deepface/weights/dlib_face_recognition_resnet_model_v1.dat"

        # ---------------------

        # download pre-trained model if it does not exist
        if os.path.isfile(weight_file) != True:
            logger.info("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")

            file_name = "dlib_face_recognition_resnet_model_v1.dat.bz2"
            url = f"http://dlib.net/files/{file_name}"
            output = f"{home}/.deepface/weights/{file_name}"
            gdown.download(url, output, quiet=False)

            zipfile = bz2.BZ2File(output)
            data = zipfile.read()
            newfilepath = output[:-4]  # discard .bz2 extension
            with open(newfilepath, "wb") as f:
                f.write(data)

        # ---------------------

        model = dlib.face_recognition_model_v1(weight_file)
        self.__model = model

        # ---------------------

        # return None  # classes must return None

    def predict(self, img_aligned):

        # functions.detectFace returns 4 dimensional images
        if len(img_aligned.shape) == 4:
            img_aligned = img_aligned[0]

        # functions.detectFace returns bgr images
        img_aligned = img_aligned[:, :, ::-1]  # bgr to rgb

        # deepface.detectFace returns an array in scale of [0, 1]
        # but dlib expects in scale of [0, 255]
        if img_aligned.max() <= 1:
            img_aligned = img_aligned * 255

        img_aligned = img_aligned.astype(np.uint8)

        model = self.__model

        img_representation = model.compute_face_descriptor(img_aligned)

        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)

        return img_representation


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
