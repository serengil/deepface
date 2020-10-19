# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
import re
from extension.constants import CONSTANTS
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

# for tensorflow 2.0 emotion model
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from keras.preprocessing import image as keras_Image

# nms code from deepsort
from yolo3.preprocessing import non_max_suppression


class YOLO(object):
    emotion_dict = {0: CONSTANTS['EMOTION']['UNSETTLED'], 1: CONSTANTS['EMOTION']['HAPPY'],
                    2: CONSTANTS['EMOTION']['NEUTRAL'], 3: CONSTANTS['EMOTION']['SAD'],
                    4: CONSTANTS['EMOTION']['ANXIOUS']}

    _defaults = {
        "model_path": 'yolov3_all/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo3_anchors.txt',
        "classes_path": 'model_data/pet_classes.txt',
        "emotion_model_dir": "emotion/",
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (608, 608),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()

        # init breed model
        self.boxes, self.scores, self.classes = self.generate()

        # init emotion models
        self.cat_emotion_model, self.dog_emotion_model = self.load_emotion_models()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_emotion_models(self):
        cat_model_path = os.path.join(self.emotion_model_dir, "Cat_classifier.h5")
        dog_model_path = os.path.join(self.emotion_model_dir, "Dog_classifier.h5")

        cat = tf.keras.models.load_model(cat_model_path)
        self.graph_cat = tf.get_default_graph()

        dog = tf.keras.models.load_model(dog_model_path)
        self.graph_dog = tf.get_default_graph()

        cat.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dog.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return cat, dog

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]]
                # K.learning_phase(): 0
            })

        # non-maxima suppression ACROSS classes
        indices = non_max_suppression(np.array(out_boxes),
                                      self.iou,
                                      out_scores)

        out_boxes = [out_boxes[i] for i in indices]
        out_scores = [out_scores[i] for i in indices]
        out_classes = [out_classes[i] for i in indices]

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        detection_dicts = []

        count = 0

        showid = len(out_classes) > 1

        for i, c in reversed(list(enumerate(out_classes))):

            # prediction of breed and bounding box
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # bounding box corners
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # crop image head and pass to emotion model
            head_image = image.crop([left, top, right, bottom])
            head_image = head_image.resize((200, 200))
            head_image = keras_Image.img_to_array(head_image)
            head_image = np.expand_dims(head_image, axis=0)
            head_image = preprocess_input(head_image)

            if predicted_class[0].islower():
                # lowercase: dog
                with self.graph_dog.as_default():
                    prediction = self.dog_emotion_model.predict(head_image)[0]
            else:
                # uppercase: cat
                with self.graph_cat.as_default():
                    prediction = self.cat_emotion_model.predict(head_image)[0]

            prediction = np.round(prediction, decimals=2)
            # prediction = [float(np.round(p * 100, 2)) for p in prediction]
            prediction_ = np.argmax(prediction)

            # label
            label = 'ID: {}'.format(count)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font=font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])

            if showid:
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c]
                )
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

            # put results into dict list
            if predicted_class[0].isupper():
                pet = "Cat"
            else:
                pet = "Dog"

            all_emotions = [
                {'name': CONSTANTS['EMOTION']['UNSETTLED'], 'value': prediction[0]},
                {'name': CONSTANTS['EMOTION']['HAPPY'], 'value': prediction[1]},
                {'name': CONSTANTS['EMOTION']['NEUTRAL'], 'value': prediction[2]},
                {'name': CONSTANTS['EMOTION']['SAD'], 'value': prediction[3]},
                {'name': CONSTANTS['EMOTION']['ANXIOUS'], 'value': prediction[4]},
            ]
            all_emotions.sort(key=lambda x: x['value'], reverse=True)
            all_emotions_result = []
            for each_emotion in all_emotions:
                all_emotions_result.append(
                    each_emotion['name'] + ' (' + str(np.round(each_emotion['value'] * 100)) + '%)')

            predicted_breed = re.sub(r"_", " ", predicted_class, flags=re.IGNORECASE)
            obj_dict = {
                'id': count,
                'pet': pet,
                'breed': "{} ({}%)".format(predicted_breed.title(), np.round(score * 100)),
                'breedTitle': predicted_breed.title(),
                'breedScore': np.round(score * 100),
                'emotion': {
                    'mostLikelyTitle': self.emotion_dict[prediction_],
                    'mostLikelyScore': np.round(np.max(prediction) * 100),
                    'mostLikely': "{} ({}%)".format(self.emotion_dict[prediction_], np.round(np.max(prediction) * 100)),
                    'allEmotions': all_emotions_result
                },
            }

            count += 1

            detection_dicts.append(obj_dict)

        # predicted_img_name = "predicted_image.jpg"
        # image.save(predicted_img_name, "JPEG")

        return image, detection_dicts

    def close_session(self):
        self.sess.close()
