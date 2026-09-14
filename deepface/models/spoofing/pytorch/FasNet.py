# built-in dependencies
from typing import Union, Any, Tuple, List

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray

# project dependencies
from deepface.commons import weight_utils
from deepface.models.spoofing.FasNetUtils import crop
from deepface.commons.logger import Logger

logger = Logger()

# pylint: disable=line-too-long, too-few-public-methods, nested-min-max
FIRST_WEIGHTS_URL = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
SECOND_WEIGHTS_URL = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"


class Fasnet:
    """
    Mini Face Anti Spoofing Net Library from repo: github.com/minivision-ai/Silent-Face-Anti-Spoofing

    Minivision's Silent-Face-Anti-Spoofing Repo licensed under Apache License 2.0
    Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/model_lib/MiniFASNet.py
    """

    def __init__(self) -> None:
        # pytorch is an opitonal dependency, enforce it to be installed if class imported
        try:
            import torch
        except Exception as err:
            raise ValueError(
                "You must install torch with `pip install torch` command to use face anti spoofing module"
            ) from err

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # download pre-trained models if not installed yet
        first_model_weight_file = weight_utils.download_weights_if_necessary(
            file_name="2.7_80x80_MiniFASNetV2.pth",
            source_url=FIRST_WEIGHTS_URL,
        )

        second_model_weight_file = weight_utils.download_weights_if_necessary(
            file_name="4_0_0_80x80_MiniFASNetV1SE.pth",
            source_url=SECOND_WEIGHTS_URL,
        )

        # guarantees Fasnet imported and torch installed
        from deepface.models.spoofing.pytorch import FasNetBackbone

        # Fasnet will use 2 distinct models to predict, then it will find the sum of predictions
        # to make a final prediction

        first_model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5)).to(device)
        second_model = FasNetBackbone.MiniFASNetV1SE(conv6_kernel=(5, 5)).to(device)

        # load model weight for first model
        state_dict = torch.load(first_model_weight_file, map_location=device)
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
        state_dict = torch.load(second_model_weight_file, map_location=device)
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

    def analyze(
        self,
        img: NDArray[Any],
        facial_area: Union[List[Union[int, float]], Tuple[Union[int, float], ...]],
    ) -> Tuple[bool, float]:
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


def to_tensor(pic: NDArray[Any]) -> Any:
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
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, img: NDArray[Any]) -> NDArray[Any]:
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    def __call__(self, pic: Any) -> Any:
        return to_tensor(pic)
