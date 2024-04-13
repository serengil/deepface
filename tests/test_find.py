# built-in dependencies
import os

# 3rd party dependencies
import cv2
import pandas as pd

# project dependencies
from deepface import DeepFace
from deepface.modules import verification
from deepface.modules import recognition
from deepface.commons import file_utils
from deepface.commons import logger as log

logger = log.get_singletonish_logger()


threshold = verification.find_threshold(model_name="VGG-Face", distance_metric="cosine")


def test_find_with_exact_path():
    img_path = os.path.join("dataset", "img1.jpg")
    dfs = DeepFace.find(img_path=img_path, db_path="dataset", silent=True)
    assert len(dfs) > 0
    for df in dfs:
        assert isinstance(df, pd.DataFrame)

        # one is img1.jpg itself
        identity_df = df[df["identity"] == img_path]
        assert identity_df.shape[0] > 0

        # validate reproducability
        assert identity_df["distance"].values[0] < threshold

        df = df[df["identity"] != img_path]
        logger.debug(df.head())
        assert df.shape[0] > 0
    logger.info("✅ test find for exact path done")


def test_find_with_array_input():
    img_path = os.path.join("dataset", "img1.jpg")
    img1 = cv2.imread(img_path)
    dfs = DeepFace.find(img1, db_path="dataset", silent=True)
    assert len(dfs) > 0
    for df in dfs:
        assert isinstance(df, pd.DataFrame)

        # one is img1.jpg itself
        identity_df = df[df["identity"] == img_path]
        assert identity_df.shape[0] > 0

        # validate reproducability
        assert identity_df["distance"].values[0] < threshold

        df = df[df["identity"] != img_path]
        logger.debug(df.head())
        assert df.shape[0] > 0

    logger.info("✅ test find for array input done")


def test_find_with_extracted_faces():
    img_path = os.path.join("dataset", "img1.jpg")
    face_objs = DeepFace.extract_faces(img_path)
    img = face_objs[0]["face"]
    dfs = DeepFace.find(img, db_path="dataset", detector_backend="skip", silent=True)
    assert len(dfs) > 0
    for df in dfs:
        assert isinstance(df, pd.DataFrame)

        # one is img1.jpg itself
        identity_df = df[df["identity"] == img_path]
        assert identity_df.shape[0] > 0

        # validate reproducability
        assert identity_df["distance"].values[0] < threshold

        df = df[df["identity"] != img_path]
        logger.debug(df.head())
        assert df.shape[0] > 0
    logger.info("✅ test find for extracted face input done")


def test_filetype_for_find():
    """
    only images as jpg and png can be loaded into database
    """
    img_path = os.path.join("dataset", "img1.jpg")
    dfs = DeepFace.find(img_path=img_path, db_path="dataset", silent=True)

    df = dfs[0]

    # img47 is webp even though its extension is jpg
    assert df[df["identity"] == "dataset/img47.jpg"].shape[0] == 0


def test_filetype_for_find_bulk_embeddings():
    imgs = file_utils.list_images("dataset")

    assert len(imgs) > 0

    # img47 is webp even though its extension is jpg
    assert "dataset/img47.jpg" not in imgs
