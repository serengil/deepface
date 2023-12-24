from typing import Any
from deepface.basemodels.DlibResNet import DlibResNet


def loadModel() -> Any:
    return DlibResNet()
