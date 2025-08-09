# built-in dependencies
from typing import Any, List, Dict, Tuple
import os

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface.models.Detector import Detector, FacialAreaRegion
from deepface.commons.logger import Logger
from deepface.commons import weight_utils, folder_utils
from deepface.models.face_detection import OpenCv as OpenCvFD

logger = Logger()


WEIGHTS_URL = (
    "https://drive.google.com/uc?"
    "id=1VkMKWPJM0oaS8eyIVZ5flcJH70Pbi-_g"
)
SUB_DIR = "tinaface"
WEIGHT_FILENAME = "tinaface.onnx"


# pylint: disable=too-few-public-methods
class TinaFaceClient(Detector):
    """
    TinaFace detector backend (scaffold).

    Notes
    - This is a lightweight integration scaffold so DeepFace can select
      `detector_backend="tinaface"`.
    - Actual inference implementation depends on which format you provide
      the model in (e.g., ONNX via OpenCV DNN, or PyTorch via a 3rd-party repo).
    - To keep dependencies optional and consistent with other backends
      (e.g., `YuNet`, `CenterFace`), we recommend ONNX + OpenCV DNN.

    Implementation TODO
    - Implement `self._forward(img)` to run the ONNX and decode outputs into
      bounding boxes, landmarks and confidence scores. Output must be a list of
      FacialAreaRegion objects like other detectors.
    - Different TinaFace ONNX exports may have different output tensor names
      and formats; adjust post-processing accordingly.
    """

    def __init__(self):
        # Defer model build to first use so environments without weights still import
        self.model: Any = None
        self.runtime: str = "auto"  # "ort" (onnxruntime) or "cv2dnn"
        # eye finder via opencv cascades to populate landmarks when not provided
        self._eye_finder = OpenCvFD.OpenCvClient()

    def _build_model(self) -> Any:
        """
        Build TinaFace model (expected ONNX) using OpenCV DNN if a weights URL
        is provided via environment variable. Otherwise, raise an informative error.
        Returns:
            model (Any): backend-specific handle (e.g., cv2.dnn_Net)
        """
        # Try ONNX Runtime first for best compatibility with dynamic shapes
        ort_sess = None
        try:
            import onnxruntime as ort  # type: ignore
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if 'CUDAExecutionProvider' in ort.get_available_providers()
                         else ['CPUExecutionProvider'])
        except Exception:
            ort = None  # type: ignore
            providers = None

        # Ensure subdir exists and download the ONNX weights
        home = folder_utils.get_deepface_home()
        weights_dir = os.path.join(home, ".deepface", "weights")
        tinaface_dir = os.path.join(weights_dir, SUB_DIR)
        os.makedirs(tinaface_dir, exist_ok=True)

        weight_file = weight_utils.download_weights_if_necessary(
            file_name=os.path.join(SUB_DIR, WEIGHT_FILENAME),
            source_url=WEIGHTS_URL,
        )
        # Try to initialize ONNX Runtime session
        if ort is not None:
            try:
                ort_sess = ort.InferenceSession(weight_file, providers=providers)
                self.runtime = "ort"
                logger.info(f"TinaFace model loaded with ONNX Runtime from {weight_file}")
                return ort_sess
            except Exception:
                ort_sess = None

        # Fallback to OpenCV DNN if ORT is not available/failed
        try:
            import cv2  # noqa: F401
        except Exception as err:
            raise ImportError(
                "Neither onnxruntime nor OpenCV DNN could be initialized. Install one of them."
            ) from err

        import cv2
        try:
            net = cv2.dnn.readNetFromONNX(weight_file)
            self.runtime = "cv2dnn"
            logger.info(f"TinaFace model loaded with OpenCV DNN from {weight_file}")
            return net
        except Exception as err:
            raise ValueError(
                "Exception while loading TinaFace ONNX with OpenCV DNN. "
                "Ensure the provided file is a valid ONNX model."
            ) from err

    @staticmethod
    def _resize_keep_ratio_pad(img: np.ndarray,
                               dst_h: int,
                               dst_w: int,
                               size_divisor: int,
                               pad_value_bgr: List[float]) -> Dict[str, Any]:
        import cv2
        h, w = img.shape[:2]
        scale = min(dst_h / h, dst_w / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h)) if (new_w != w or new_h != h) else img.copy()

        # compute padded dims to be divisible by size_divisor
        pad_w = int((size_divisor - (new_w % size_divisor)) % size_divisor)
        pad_h = int((size_divisor - (new_h % size_divisor)) % size_divisor)

        padded = np.full((new_h + pad_h, new_w + pad_w, 3), pad_value_bgr, dtype=resized.dtype)
        padded[:new_h, :new_w, :] = resized

        meta = {
            "scale": scale,
            "new_w": new_w,
            "new_h": new_h,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "padded_w": new_w + pad_w,
            "padded_h": new_h + pad_h,
        }
        return {"img": padded, "meta": meta}

    @staticmethod
    def _preprocess(img: np.ndarray) -> Dict[str, Any]:
        # Doc-config
        img_norm_mean_rgb = [123.675, 116.28, 103.53]
        img_norm_std_rgb = [1.0, 1.0, 1.0]
        size_divisor = 32
        dst_shape = (1100, 1650)  # (H, W)

        # Resize keep ratio and pad with mean in BGR (mean[::-1])
        result = TinaFaceClient._resize_keep_ratio_pad(
            img=img,
            dst_h=dst_shape[0],
            dst_w=dst_shape[1],
            size_divisor=size_divisor,
            pad_value_bgr=img_norm_mean_rgb[::-1],
        )
        padded = result["img"].astype(np.float32)
        meta = result["meta"]

        # To RGB then Normalize
        padded = padded[:, :, ::-1]  # BGR -> RGB
        padded -= np.array(img_norm_mean_rgb, dtype=np.float32)
        padded /= np.array(img_norm_std_rgb, dtype=np.float32)

        # To tensor (N,C,H,W)
        chw = np.transpose(padded, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

        return {"input": input_tensor, "meta": meta}

    @staticmethod
    def _parse_out_rows(out: np.ndarray,
                        score_threshold: float,
                        orig_w: int,
                        orig_h: int,
                        scale: float) -> List[FacialAreaRegion]:
        resp: List[FacialAreaRegion] = []
        if out.ndim != 2 or out.shape[1] < 5:
            return resp
        for row in out:
            x1, y1, x2, y2 = row[0:4].tolist()
            score = float(row[4])
            if score < score_threshold:
                continue
            # Map back to original image space
            x1 = max(0, int(round(x1 / scale)))
            y1 = max(0, int(round(y1 / scale)))
            x2 = min(orig_w - 1, int(round(x2 / scale)))
            y2 = min(orig_h - 1, int(round(y2 / scale)))
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            left_eye = right_eye = nose = mouth_left = mouth_right = None
            if out.shape[1] >= 15:
                try:
                    le_x, le_y = int(round(row[5] / scale)), int(round(row[6] / scale))
                    re_x, re_y = int(round(row[7] / scale)), int(round(row[8] / scale))
                    n_x, n_y = int(round(row[9] / scale)), int(round(row[10] / scale))
                    ml_x, ml_y = int(round(row[11] / scale)), int(round(row[12] / scale))
                    mr_x, mr_y = int(round(row[13] / scale)), int(round(row[14] / scale))
                    left_eye = (le_x, le_y)
                    right_eye = (re_x, re_y)
                    nose = (n_x, n_y)
                    mouth_left = (ml_x, ml_y)
                    mouth_right = (mr_x, mr_y)
                except Exception:
                    pass

            resp.append(
                FacialAreaRegion(
                    x=x1,
                    y=y1,
                    w=w,
                    h=h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    confidence=score,
                    nose=nose,
                    mouth_left=mouth_left,
                    mouth_right=mouth_right,
                )
            )
        return resp

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float, top_k: int) -> List[int]:
        if boxes.size == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        if top_k > 0:
            order = order[:top_k]
        keep = []
        suppressed = np.zeros_like(order, dtype=bool)
        for i_idx in range(order.shape[0]):
            i = order[i_idx]
            if suppressed[i_idx]:
                continue
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[i_idx + 1:]])
            yy1 = np.maximum(y1[i], y1[order[i_idx + 1:]])
            xx2 = np.minimum(x2[i], x2[order[i_idx + 1:]])
            yy2 = np.minimum(y2[i], y2[order[i_idx + 1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[i_idx + 1:]] - inter)
            suppressed[i_idx + 1:][ovr >= iou_thresh] = True
        return keep

    @staticmethod
    def _generate_level_anchors(stride: int,
                                height: int,
                                width: int,
                                scales_per_octave: int,
                                octave_base_scale: float,
                                ratio: float
                                ) -> Tuple[
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray
                                ]:
        # Compute sizes: stride * (octave_base_scale * 2**(i/scales_per_octave))
        scale_factors = [
            octave_base_scale * (2 ** (i / scales_per_octave))
            for i in range(scales_per_octave)
        ]
        sqrt_r = np.sqrt(ratio)
        shift_x = (np.arange(0, width) + 0.5) * stride
        shift_y = (np.arange(0, height) + 0.5) * stride
        shift_xx, shift_yy = np.meshgrid(shift_x, shift_y)
        centers = np.stack([shift_xx, shift_yy], axis=-1)
        centers = centers.reshape(-1, 2)
        anchors = []
        for s in scale_factors:
            size = stride * s
            w = size * sqrt_r
            h = size / sqrt_r
            anchors.append([w, h])
        anchors = np.array(anchors, dtype=np.float32)  # (A, 2)
        A = anchors.shape[0]
        HW = centers.shape[0]
        # Repeat to (HW*A, ...)
        widths = np.repeat(anchors[:, 0], HW)
        heights = np.repeat(anchors[:, 1], HW)
        centers_tiled = np.tile(centers, (A, 1))
        cx = centers_tiled[:, 0]
        cy = centers_tiled[:, 1]
        return cx, cy, widths, heights

    def _decode_fpn_outputs(self,
                            outputs_map: Dict[str, np.ndarray],
                            meta: Dict[str, Any],
                            score_thr: float,
                            nms_iou_thr: float,
                            max_per_img: int,
                            orig_w: int,
                            orig_h: int) -> List[FacialAreaRegion]:
        # config per doc
        strides = [4, 8, 16, 32, 64, 128]
        use_sigmoid = True
        scales_per_octave = 3
        ratio = 1.3
        bbox_std = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

        all_boxes = []
        all_scores = []

        padded_h = meta["padded_h"]
        padded_w = meta["padded_w"]

        # Helper: find level tensors by matching spatial dims
        def find_level_tensors_for_stride(s: int):
            H_expect = int(np.ceil(padded_h / s))
            W_expect = int(np.ceil(padded_w / s))
            cls_list = []
            bbox_list = []
            for _, arr in outputs_map.items():
                a = np.squeeze(arr)
                if not (isinstance(a, np.ndarray) and a.ndim == 3):
                    continue
                C, H, W = a.shape
                if abs(H - H_expect) <= 1 and abs(W - W_expect) <= 1:
                    # Heuristic: bbox if C % 4 == 0 and >= 12; cls if remainder anchors
                    if C % 4 == 0 and C // 4 >= scales_per_octave:
                        bbox_list.append(a)
                    else:
                        cls_list.append(a)
            return cls_list, bbox_list

        for s in strides:
            cls_list, bbox_list = find_level_tensors_for_stride(s)
            if len(cls_list) == 0 or len(bbox_list) == 0:
                continue
            # pick the largest channel tensor as bbox, smallest as cls if ambiguous
            bbox = max(bbox_list, key=lambda a: a.shape[0])  # (C, H, W)
            cls = min(cls_list, key=lambda a: a.shape[0])    # (C, H, W)

            Cc, H, W = cls.shape
            num_anchors = scales_per_octave  # ratios=1 only
            # reshape
            cls_scores = cls.reshape(num_anchors, Cc // num_anchors, H, W)
            if use_sigmoid:
                cls_scores = self._sigmoid(cls_scores)
            # pick foreground score (C=1)
            cls_scores = cls_scores[:, 0, :, :]  # (A,H,W)
            bbox_preds = bbox.reshape(num_anchors, 4, H, W)  # (A,4,H,W)

            # anchors per level
            cx, cy, aw, ah = self._generate_level_anchors(
                stride=s,
                height=H,
                width=W,
                scales_per_octave=scales_per_octave,
                octave_base_scale=2 ** (4 / 3),
                ratio=ratio,
            )
            A = num_anchors
            HW = H * W
            # tile preds to (HW*A)
            scores = cls_scores.reshape(A, HW).transpose(1, 0).reshape(-1)
            dx = bbox_preds[:, 0, :, :].reshape(A, HW).transpose(1, 0).reshape(-1)
            dy = bbox_preds[:, 1, :, :].reshape(A, HW).transpose(1, 0).reshape(-1)
            dw = bbox_preds[:, 2, :, :].reshape(A, HW).transpose(1, 0).reshape(-1)
            dh = bbox_preds[:, 3, :, :].reshape(A, HW).transpose(1, 0).reshape(-1)

            # decode
            x_c = dx * bbox_std[0] * aw + cx
            y_c = dy * bbox_std[1] * ah + cy
            w = np.exp(dw * bbox_std[2]) * aw
            h = np.exp(dh * bbox_std[3]) * ah
            x1 = x_c - 0.5 * w
            y1 = y_c - 0.5 * h
            x2 = x_c + 0.5 * w
            y2 = y_c + 0.5 * h

            # scale back to original image space
            sc = meta["scale"]
            x1 = x1 / sc
            y1 = y1 / sc
            x2 = x2 / sc
            y2 = y2 / sc

            boxes = np.stack([x1, y1, x2, y2], axis=1)
            all_boxes.append(boxes)
            all_scores.append(scores)

        if len(all_boxes) == 0:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # clip and threshold
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)

        keep_mask = scores >= score_thr
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]

        # NMS
        keep = self._nms(boxes, scores, iou_thresh=nms_iou_thr, top_k=10000)
        boxes = boxes[keep]
        scores = scores[keep]

        # top-k per image
        if boxes.shape[0] > max_per_img:
            topk_idx = np.argsort(scores)[-max_per_img:][::-1]
            boxes = boxes[topk_idx]
            scores = scores[topk_idx]

        results: List[FacialAreaRegion] = []
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            results.append(
                FacialAreaRegion(
                    x=x1,
                    y=y1,
                    w=w,
                    h=h,
                    confidence=float(sc),
                )
            )
        return results

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align faces with TinaFace

        Args:
            img (np.ndarray): pre-loaded image as numpy array (BGR)

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        if self.model is None:
            self.model = self._build_model()

        score_threshold = 0.35  # per doc
        height, width = img.shape[0], img.shape[1]

        # Preprocess per documentation
        pp = self._preprocess(img)
        input_tensor = pp["input"]
        meta = pp["meta"]

        if self.runtime == "ort":
            sess = self.model
            input_name = sess.get_inputs()[0].name
            outputs_list = sess.get_outputs()
            output_names = [o.name for o in outputs_list]
            outputs = sess.run(output_names, {input_name: input_tensor})
            outputs_map: Dict[str, np.ndarray] = dict(zip(output_names, outputs))

            # Preferred 'out' parsed if available
            if "out" in outputs_map:
                out = np.squeeze(outputs_map["out"])  # Nx(>=5)
                resp = self._parse_out_rows(out, score_threshold, width, height, meta["scale"])
                if len(resp) > 0:
                    return self._populate_missing_eyes(img, resp)
            # Fall back: search for any 2D array with >=5 cols
            for _, arr in outputs_map.items():
                out = np.squeeze(arr)
                if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 5:
                    resp = self._parse_out_rows(out, score_threshold, width, height, meta["scale"])
                    if len(resp) > 0:
                        return self._populate_missing_eyes(img, resp)

            # Attempt FPN-style decode from multi-branch outputs
            resp = self._decode_fpn_outputs(
                outputs_map=outputs_map,
                meta=meta,
                score_thr=score_threshold,
                nms_iou_thr=0.45,
                max_per_img=300,
                orig_w=width,
                orig_h=height,
            )
            if len(resp) > 0:
                return self._populate_missing_eyes(img, resp)

        else:
            # OpenCV DNN fallback
            self.model.setInput(input_tensor)
            outs = self.model.forward(self.model.getUnconnectedOutLayersNames())
            if isinstance(outs, (list, tuple)):
                for out in outs:
                    out = np.squeeze(out)
                    if (
                        isinstance(out, np.ndarray)
                        and out.ndim == 2
                        and out.shape[1] >= 5
                    ):
                        resp = self._parse_out_rows(
                            out, score_threshold, width, height, meta["scale"]
                        )
                        if len(resp) > 0:
                            return self._populate_missing_eyes(img, resp)
            # Build a map for heuristic FPN decode
            outputs_map = {}
            if isinstance(outs, (list, tuple)):
                for idx, arr in enumerate(outs):
                    outputs_map[f"out_{idx}"] = arr
            else:
                outputs_map["out"] = outs

            resp = self._decode_fpn_outputs(
                outputs_map={k: np.squeeze(v) for k, v in outputs_map.items()},
                meta=meta,
                score_thr=score_threshold,
                nms_iou_thr=0.45,
                max_per_img=300,
                orig_w=width,
                orig_h=height,
            )
            if len(resp) > 0:
                return self._populate_missing_eyes(img, resp)

        # If here, output format is not recognized; no detections
        return []

    def _populate_missing_eyes(
        self, img: np.ndarray, faces: List[FacialAreaRegion]
    ) -> List[FacialAreaRegion]:
        updated: List[FacialAreaRegion] = []
        for fa in faces:
            if fa.left_eye is None or fa.right_eye is None:
                x, y, w, h = fa.x, fa.y, fa.w, fa.h
                x = max(0, x)
                y = max(0, y)
                w = max(0, min(img.shape[1] - x, w))
                h = max(0, min(img.shape[0] - y, h))
                if w > 0 and h > 0:
                    roi = img[int(y): int(y + h), int(x): int(x + w)]
                    le, re = self._eye_finder.find_eyes(roi)
                    if le is not None:
                        fa.left_eye = (int(x + le[0]), int(y + le[1]))
                    if re is not None:
                        fa.right_eye = (int(x + re[0]), int(y + re[1]))
                    # Heuristic fallback: place eyes at typical positions
                    if fa.left_eye is None or fa.right_eye is None:
                        le_x = int(x + 0.70 * w)
                        re_x = int(x + 0.30 * w)
                        eye_y = int(y + 0.40 * h)
                        fa.left_eye = fa.left_eye or (le_x, eye_y)
                        fa.right_eye = fa.right_eye or (re_x, eye_y)
            updated.append(fa)
        return updated
        