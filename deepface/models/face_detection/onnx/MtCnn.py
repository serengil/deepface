# built-in dependencies
from typing import Any, List, Tuple

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons.onnx_utils import build_session
from deepface.models.Detector import Detector, FacialAreaRegion

# pylint: disable=line-too-long

# P-Net, R-Net and O-Net of github.com/timesler/facenet-pytorch exported to onnx. They take
# channels last inputs, and P-Net accepts any input size. Its weights are the same as the
# ones of github.com/ipazc/mtcnn that the tensorflow backend uses.
PNET_WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/mtcnn_pnet.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/mtcnn_pnet.onnx",
]
RNET_WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/mtcnn_rnet.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/mtcnn_rnet.onnx",
]
ONET_WEIGHTS_URL = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/mtcnn_onet.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/mtcnn_onet.onnx",
]

# defaults of facenet-pytorch, which the pytorch backend uses
MIN_FACE_SIZE = 20
THRESHOLDS = (0.6, 0.7, 0.7)
FACTOR = 0.709

# R-Net and O-Net inputs are fed in batches of this size at most
BATCH_SIZE = 512


# pylint: disable=too-few-public-methods
class MtCnnClient(Detector):
    """
    MtCnn Detector for the onnx backend engine. This is a numpy port of the pipeline of
    github.com/timesler/facenet-pytorch, so it finds the same faces as the pytorch backend.
    """

    def __init__(self) -> None:
        self.pnet = build_session(file_name="mtcnn_pnet.onnx", source_url=PNET_WEIGHTS_URL)
        self.rnet = build_session(file_name="mtcnn_rnet.onnx", source_url=RNET_WEIGHTS_URL)
        self.onet = build_session(file_name="mtcnn_onet.onnx", source_url=ONET_WEIGHTS_URL)

    def detect_faces(self, img: NDArray[Any]) -> List[FacialAreaRegion]:
        """
        Detect and align face with mtcnn

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        # mtcnn expects RGB but OpenCV read BGR
        img_rgb = img[:, :, ::-1].astype(np.float32)
        boxes, points = self.__detect(img_rgb)

        resp = []
        for box, point in zip(boxes, points):
            x1, y1, x2, y2, confidence = box
            # mtcnn detector assigns left eye with respect to the observer
            # but we are setting it with respect to the person itself
            right_eye = (int(point[0][0]), int(point[0][1]))
            left_eye = (int(point[1][0]), int(point[1][1]))

            facial_area = FacialAreaRegion(
                x=int(x1),
                y=int(y1),
                w=int(x2 - x1),
                h=int(y2 - y1),
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=float(confidence),
            )
            resp.append(facial_area)

        return resp

    def __detect(self, img: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
        """
        Run the three stages of mtcnn
        Args:
            img (np.ndarray): (h, w, 3) shaped float32 rgb image
        Returns:
            boxes (np.ndarray): (n, 5) shaped x1, y1, x2, y2 and confidence of faces,
                the largest face first
            points (np.ndarray): (n, 5, 2) shaped facial landmarks - eyes, nose and
                mouth corners - of the faces
        """
        h, w = img.shape[:2]
        no_faces = np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        # first stage - P-Net proposes candidates on an image pyramid
        boxes_list = []
        for scale in scale_pyramid(h, w):
            im_data = area_resize(img, int(h * scale + 1), int(w * scale + 1))
            reg, probs = self.pnet.run(None, {"input": normalize(im_data)[np.newaxis]})
            boxes_scale = generate_bounding_box(reg[0], probs[0, 1], scale, THRESHOLDS[0])
            pick = nms(boxes_scale[:, :4], boxes_scale[:, 4], 0.5)
            boxes_list.append(boxes_scale[pick])

        # the image is smaller than the minimum face size
        if len(boxes_list) == 0:
            return no_faces

        boxes = np.concatenate(boxes_list, axis=0)
        boxes = boxes[nms(boxes[:, :4], boxes[:, 4], 0.7)]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        boxes = np.stack(
            [
                boxes[:, 0] + boxes[:, 5] * regw,
                boxes[:, 1] + boxes[:, 6] * regh,
                boxes[:, 2] + boxes[:, 7] * regw,
                boxes[:, 3] + boxes[:, 8] * regh,
                boxes[:, 4],
            ],
            axis=1,
        )
        boxes = rerec(boxes)

        # second stage - R-Net refines the candidates
        boxes, im_data = crop_candidates(img, boxes, 24)
        if len(boxes) == 0:
            return no_faces

        reg, probs = run_in_batches(self.rnet, normalize(im_data))
        score = probs[:, 1]
        ipass = score > THRESHOLDS[1]
        boxes = np.concatenate([boxes[ipass, :4], score[ipass, np.newaxis]], axis=1)
        reg = reg[ipass]

        pick = nms(boxes[:, :4], boxes[:, 4], 0.7)
        boxes = rerec(bbreg(boxes[pick], reg[pick]))

        # third stage - O-Net finds the faces and their landmarks
        boxes, im_data = crop_candidates(img, boxes, 48)
        if len(boxes) == 0:
            return no_faces

        reg, landmarks, probs = run_in_batches(self.onet, normalize(im_data))
        score = probs[:, 1]
        ipass = score > THRESHOLDS[2]
        boxes = np.concatenate([boxes[ipass, :4], score[ipass, np.newaxis]], axis=1)
        reg = reg[ipass]
        landmarks = landmarks[ipass]

        w_i = boxes[:, 2:3] - boxes[:, 0:1] + 1
        h_i = boxes[:, 3:4] - boxes[:, 1:2] + 1
        points_x = w_i * landmarks[:, :5] + boxes[:, 0:1] - 1
        points_y = h_i * landmarks[:, 5:10] + boxes[:, 1:2] - 1
        points = np.stack([points_x, points_y], axis=2)
        boxes = bbreg(boxes, reg)

        pick = nms_min(boxes[:, :4], boxes[:, 4], 0.7)
        boxes, points = boxes[pick], points[pick]

        # largest face first
        order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
        return boxes[order], points[order]


def scale_pyramid(h: int, w: int) -> List[float]:
    """
    Find the scales the image is resized to, so that faces from MIN_FACE_SIZE up to the
    image size fit into the 12x12 window of P-Net
    Args:
        h (int): image height
        w (int): image width
    Returns:
        scales (list)
    """
    m = 12.0 / MIN_FACE_SIZE
    minl = min(h, w) * m

    scales = []
    scale = m
    while minl >= 12:
        scales.append(scale)
        scale = scale * FACTOR
        minl = minl * FACTOR
    return scales


def normalize(img: NDArray[Any]) -> NDArray[Any]:
    """
    Scale pixels into [-1, 1] as mtcnn expects
    Args:
        img (np.ndarray): image with pixels in [0, 255]
    Returns:
        img (np.ndarray)
    """
    return ((img - 127.5) * 0.0078125).astype(np.float32)


def area_resize(img: NDArray[Any], out_h: int, out_w: int) -> NDArray[Any]:
    """
    Resize an image as torch.nn.functional.interpolate does in area mode, which is
    adaptive average pooling: each output pixel is the mean of the input pixels between
    floor(i * in / out) and ceil((i + 1) * in / out). cv2's area interpolation weighs
    pixels at the bin edges partially instead, so it does not give the same result.
    Args:
        img (np.ndarray): (h, w, c) shaped image
        out_h (int): output height
        out_w (int): output width
    Returns:
        img (np.ndarray): (out_h, out_w, c) shaped float32 image
    """
    h, w, c = img.shape
    # rows are pooled first, then the image is transposed so that its columns are pooled
    # as rows too - gathering contiguous rows is much faster than gathering columns
    out = pool_rows(img.astype(np.float32, copy=False).reshape(h, w * c), out_h)
    out = out.reshape(out_h, w, c).transpose(1, 0, 2).reshape(w, out_h * c)
    out = pool_rows(out, out_w)
    return np.ascontiguousarray(out.reshape(out_w, out_h, c).transpose(1, 0, 2))


def pool_rows(img: NDArray[Any], out_size: int) -> NDArray[Any]:
    """
    Adaptive average pooling of the rows of a 2d array. A bin is a run of adjacent rows,
    so the k-th rows of all bins are gathered at once and added to the bins having one -
    this is a few vectorized passes over the input, much faster than np.add.reduceat.
    Args:
        img (np.ndarray): (in_size, n) shaped float32 array
        out_size (int): number of output rows
    Returns:
        pooled (np.ndarray): (out_size, n) shaped float32 array
    """
    in_size = img.shape[0]
    idx = np.arange(out_size)
    start = (idx * in_size) // out_size
    end = -((-(idx + 1) * in_size) // out_size)  # ceil division
    lengths = end - start

    sums = img[start]
    for k in range(1, int(lengths.max())):
        if lengths.min() > k:
            sums += img[start + k]
        else:
            bins = np.nonzero(lengths > k)[0]
            sums[bins] += img[start[bins] + k]

    return sums / lengths[:, np.newaxis].astype(np.float32)  # type: ignore[no-any-return]


def generate_bounding_box(
    reg: NDArray[Any], probs: NDArray[Any], scale: float, threshold: float
) -> NDArray[Any]:
    """
    Map the P-Net windows passing the threshold back to the original image
    Args:
        reg (np.ndarray): (4, h, w) shaped box regressions of P-Net
        probs (np.ndarray): (h, w) shaped face probabilities of P-Net
        scale (float): scale the image was resized with
        threshold (float): face probability threshold
    Returns:
        boxes (np.ndarray): (n, 9) shaped x1, y1, x2, y2, score and 4 regressions
    """
    stride = 2
    cellsize = 12

    ys, xs = np.nonzero(probs >= threshold)
    score = probs[ys, xs]
    reg = reg[:, ys, xs].T
    bb = np.stack([xs, ys], axis=1).astype(np.float32)
    q1 = np.floor((stride * bb + 1) / scale)
    q2 = np.floor((stride * bb + cellsize - 1 + 1) / scale)
    boxes = np.concatenate([q1, q2, score[:, np.newaxis], reg], axis=1)
    return boxes.astype(np.float32)  # type: ignore[no-any-return]


def nms(boxes: NDArray[Any], scores: NDArray[Any], threshold: float) -> NDArray[Any]:
    """
    Non maximum suppression with intersection over union, as torchvision.ops.nms does
    Args:
        boxes (np.ndarray): (n, 4) shaped x1, y1, x2, y2
        scores (np.ndarray): (n,) shaped scores
        threshold (float): boxes overlapping a kept one more than this are suppressed
    Returns:
        keep (np.ndarray): indices of the kept boxes, in decreasing order of score
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= threshold]
    return np.array(keep, dtype=np.int64)


def nms_min(boxes: NDArray[Any], scores: NDArray[Any], threshold: float) -> NDArray[Any]:
    """
    Non maximum suppression with intersection over the smaller box, which mtcnn applies
    to its final faces so that a box inside another one is suppressed
    Args:
        boxes (np.ndarray): (n, 4) shaped x1, y1, x2, y2
        scores (np.ndarray): (n,) shaped scores
        threshold (float): boxes overlapping a kept one more than this are suppressed
    Returns:
        keep (np.ndarray): indices of the kept boxes, in decreasing order of score
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)

    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)
        rest = order[:-1]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        overlap = (w * h) / np.minimum(area[i], area[rest])
        order = rest[overlap <= threshold]
    return np.array(keep, dtype=np.int64)


def bbreg(boxes: NDArray[Any], reg: NDArray[Any]) -> NDArray[Any]:
    """
    Calibrate boxes with the regressions of a network
    Args:
        boxes (np.ndarray): (n, 5) shaped x1, y1, x2, y2 and score
        reg (np.ndarray): (n, 4) shaped regressions
    Returns:
        boxes (np.ndarray): (n, 5) shaped calibrated boxes
    """
    boxes = boxes.copy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    boxes[:, 0] += reg[:, 0] * w
    boxes[:, 1] += reg[:, 1] * h
    boxes[:, 2] += reg[:, 2] * w
    boxes[:, 3] += reg[:, 3] * h
    return boxes


def rerec(boxes: NDArray[Any]) -> NDArray[Any]:
    """
    Turn boxes into squares around their centers
    Args:
        boxes (np.ndarray): (n, 5) shaped x1, y1, x2, y2 and score
    Returns:
        boxes (np.ndarray): (n, 5) shaped squares
    """
    boxes = boxes.copy()
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    side = np.maximum(w, h)
    boxes[:, 0] = boxes[:, 0] + w * 0.5 - side * 0.5
    boxes[:, 1] = boxes[:, 1] + h * 0.5 - side * 0.5
    boxes[:, 2] = boxes[:, 0] + side
    boxes[:, 3] = boxes[:, 1] + side
    return boxes


def crop_candidates(
    img: NDArray[Any], boxes: NDArray[Any], size: int
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Crop candidate boxes from the image and resize them to the input size of a network.
    Boxes are clipped into the image, and the ones left empty are dropped.
    Args:
        img (np.ndarray): (h, w, 3) shaped image
        boxes (np.ndarray): (n, 5) shaped x1, y1, x2, y2 and score
        size (int): input size of the network
    Returns:
        boxes (np.ndarray): (m, 5) shaped boxes that are cropped
        crops (np.ndarray): (m, size, size, 3) shaped crops
    """
    h, w = img.shape[:2]
    coords = np.trunc(boxes[:, :4]).astype(np.int64)
    x = np.maximum(coords[:, 0], 1)
    y = np.maximum(coords[:, 1], 1)
    ex = np.minimum(coords[:, 2], w)
    ey = np.minimum(coords[:, 3], h)

    valid = (ey > y - 1) & (ex > x - 1)
    crops = [
        area_resize(img[y[k] - 1 : ey[k], x[k] - 1 : ex[k]], size, size)
        for k in np.nonzero(valid)[0]
    ]
    if len(crops) == 0:
        return boxes[:0], np.zeros((0, size, size, 3), dtype=np.float32)
    return boxes[valid], np.stack(crops)


def run_in_batches(session: Any, im_data: NDArray[Any]) -> List[NDArray[Any]]:
    """
    Feed crops into R-Net or O-Net in batches
    Args:
        session (onnxruntime.InferenceSession): R-Net or O-Net
        im_data (np.ndarray): (n, size, size, 3) shaped normalized crops
    Returns:
        outputs (list): outputs of the network, each (n, ...) shaped
    """
    outs = [
        session.run(None, {"input": im_data[i : i + BATCH_SIZE]})
        for i in range(0, len(im_data), BATCH_SIZE)
    ]
    return [np.concatenate(out, axis=0) for out in zip(*outs)]
