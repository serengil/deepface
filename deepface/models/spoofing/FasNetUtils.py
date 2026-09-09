# built-in dependencies
from typing import Any, List, Tuple, Union

# 3rd party dependencies
import cv2
import numpy as np
from numpy.typing import NDArray

# These functions are copied from Minivision's Silent-Face-Anti-Spoofing Repo
# licensed under Apache License 2.0
# Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing
# They are framework agnostic, both the tensorflow and the pytorch backends use them.


def _get_new_box(
    src_w: int,
    src_h: int,
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...]],
    scale: float,
) -> Tuple[int, int, int, int]:
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2]
    box_h = bbox[3]
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop(
    org_img: NDArray[Any],
    bbox: Union[List[Union[int, float]], Tuple[Union[int, float], ...]],
    scale: float,
    out_w: int,
    out_h: int,
) -> Any:
    src_h, src_w, _ = np.shape(org_img)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
    img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
    dst_img = cv2.resize(img, (out_w, out_h))
    return dst_img
