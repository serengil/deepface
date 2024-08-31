# built-in dependencies
import os
from typing import List

# 3rd party dependencies
import numpy as np
import cv2

# project dependencies
from deepface.commons import weight_utils
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=c-extension-no-member

WEIGHTS_URL = "https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx"


class CenterFaceClient(Detector):
    def __init__(self):
        # BUG: model must be flushed for each call
        # self.model = self.build_model()
        pass

    def build_model(self):
        """
        Download pre-trained weights of CenterFace model if necessary and load built model
        """
        weights_path = weight_utils.download_weights_if_necessary(
            file_name="centerface.onnx", source_url=WEIGHTS_URL
        )

        return CenterFace(weight_path=weights_path)

    def detect_faces(self, img: np.ndarray) -> List["FacialAreaRegion"]:
        """
        Detect and align face with CenterFace

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        threshold = float(os.getenv("CENTERFACE_THRESHOLD", "0.80"))

        # BUG: model causes problematic results from 2nd call if it is not flushed
        # detections, landmarks = self.model.forward(
        #     img, img.shape[0], img.shape[1], threshold=threshold
        # )
        detections, landmarks = self.build_model().forward(
            img, img.shape[0], img.shape[1], threshold=threshold
        )

        for i, detection in enumerate(detections):
            boxes, confidence = detection[:4], detection[4]

            x = boxes[0]
            y = boxes[1]
            w = boxes[2] - x
            h = boxes[3] - y

            landmark = landmarks[i]

            right_eye = (int(landmark[0]), int(landmark[1]))
            left_eye = (int(landmark[2]), int(landmark[3]))
            # nose = (int(landmark[4]), int(landmark [5]))
            # mouth_right = (int(landmark[6]), int(landmark [7]))
            # mouth_left = (int(landmark[8]), int(landmark [9]))

            facial_area = FacialAreaRegion(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=min(max(0, float(confidence)), 1.0),
            )
            resp.append(facial_area)

        return resp


class CenterFace:
    """
    This class is heavily inspired from
        github.com/Star-Clouds/CenterFace/blob/master/prj-python/centerface.py
    """

    def __init__(self, weight_path: str):
        self.net = cv2.dnn.readNetFromONNX(weight_path)
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def forward(self, img, height, width, threshold=0.5):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        return self.inference_opencv(img, threshold)

    def inference_opencv(self, img, threshold):
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0,
            size=(self.img_w_new, self.img_h_new),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", "540"])
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        dets, lms = self.decode(
            heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold
        )
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = (
                dets[:, 0:4:2] / self.scale_w,
                dets[:, 1:4:2] / self.scale_h,
            )
            lms[:, 0:10:2], lms[:, 1:10:2] = (
                lms[:, 0:10:2] / self.scale_w,
                lms[:, 1:10:2] / self.scale_h,
            )
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)
        return dets, lms

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []
        if len(c0) > 0:
            # pylint:disable=consider-using-enumerate
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(
                    0, (c0[i] + o0 + 0.5) * 4 - s0 / 2
                )
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            lms = np.asarray(lms, dtype=np.float32)
            lms = lms[keep, :]
        return boxes, lms

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep
