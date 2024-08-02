# Minivision's Silent-Face-Anti-Spoofing Repo licensed under Apache License 2.0
# Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/model_lib/MiniFASNet.py

# built-in dependencies
from typing import Union

# 3rd party dependencies
import cv2
import numpy as np

# project dependencies
from deepface.commons import folder_utils, file_utils
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods
class Fasnet:
    """
    Mini Face Anti Spoofing Net Library from repo: github.com/minivision-ai/Silent-Face-Anti-Spoofing
    """

    def __init__(self):
        # pytorch is an opitonal dependency, enforce it to be installed if class imported
        try:
            import torch
        except Exception as err:
            raise ValueError(
                "You must install torch with `pip install pytorch` command to use face anti spoofing module"
            ) from err

        home = folder_utils.get_deepface_home()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # download pre-trained models if not installed yet
        file_utils.download_external_file(
            file_name="2.7_80x80_MiniFASNetV2.pth",
            exact_file_path=f"{home}/.deepface/weights/2.7_80x80_MiniFASNetV2.pth",
            url="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        )

        file_utils.download_external_file(
            file_name="4_0_0_80x80_MiniFASNetV1SE.pth",
            exact_file_path=f"{home}/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth",
            url="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
        )

        # guarantees Fasnet imported and torch installed
        from deepface.spoofmodels import FasNetBackbone

        # Fasnet will use 2 distinct models to predict, then it will find the sum of predictions
        # to make a final prediction

        first_model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5)).to(device)
        second_model = FasNetBackbone.MiniFASNetV1SE(conv6_kernel=(5, 5)).to(device)

        # load model weight for first model
        state_dict = torch.load(
            f"{home}/.deepface/weights/2.7_80x80_MiniFASNetV2.pth", map_location=device
        )
        keys = iter(state_dict)
        first_layer_name = keys.__next__()

        if first_layer_name.find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            first_model.load_state_dict(new_state_dict)
        else:
            first_model.load_state_dict(state_dict)

        # load model weight for second model
        state_dict = torch.load(
            f"{home}/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth", map_location=device
        )
        keys = iter(state_dict)
        first_layer_name = keys.__next__()

        if first_layer_name.find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            second_model.load_state_dict(new_state_dict)
        else:
            second_model.load_state_dict(state_dict)

        # evaluate models
        _ = first_model.eval()
        _ = second_model.eval()

        self.first_model = first_model
        self.second_model = second_model

    def analyze(self, img: np.ndarray, facial_area: Union[list, tuple]):
        """
        Analyze a given image spoofed or not
        Args:
            img (np.ndarray): pre loaded image
            facial_area (list or tuple): facial rectangle area coordinates with x, y, w, h respectively
        Returns:
            result (tuple): a result tuple consisting of is_real and score
        """
        import torch
        import torch.nn.functional as F

        x, y, w, h = facial_area
        first_img = crop(img, (x, y, w, h), 2.7, 80, 80)
        second_img = crop(img, (x, y, w, h), 4, 80, 80)

        test_transform = Compose(
            [
                ToTensor(),
            ]
        )

        first_img = test_transform(first_img)
        first_img = first_img.unsqueeze(0).to(self.device)

        second_img = test_transform(second_img)
        second_img = second_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            first_result = self.first_model.forward(first_img)
            first_result = F.softmax(first_result).cpu().numpy()

            second_result = self.second_model.forward(second_img)
            second_result = F.softmax(second_result).cpu().numpy()

        prediction = np.zeros((1, 3))
        prediction += first_result
        prediction += second_result

        label = np.argmax(prediction)
        is_real = True if label == 1 else False  # pylint: disable=simplifiable-if-expression
        score = prediction[0][label] / 2

        return is_real, score


# subsdiary classes and functions


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    import torch

    # handle numpy array
    # IR image channel=1: modify by lzc --> 20190730
    if pic.ndim == 2:
        pic = pic.reshape((pic.shape[0], pic.shape[1], 1))

    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    # return img.float().div(255)  modify by zkx
    return img.float()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, pic):
        return to_tensor(pic)


def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2]
    box_h = bbox[3]
    # pylint: disable=nested-min-max
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


def crop(org_img, bbox, scale, out_w, out_h):
    src_h, src_w, _ = np.shape(org_img)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
    img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
    dst_img = cv2.resize(img, (out_w, out_h))
    return dst_img
