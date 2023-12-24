import cv2
import pandas as pd
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger("tests/test_find.py")


def test_find_with_exact_path():
    dfs = DeepFace.find(img_path="dataset/img1.jpg", db_path="dataset", silent=True)
    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        logger.debug(df.head())
        assert df.shape[0] > 0
    logger.info("✅ test find for exact path done")


def test_find_with_array_input():
    img1 = cv2.imread("dataset/img1.jpg")
    dfs = DeepFace.find(img1, db_path="dataset", silent=True)

    for df in dfs:
        logger.debug(df.head())
        assert df.shape[0] > 0

    logger.info("✅ test find for array input done")
