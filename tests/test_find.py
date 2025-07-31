# built-in dependencies
import os

# 3rd party dependencies
import cv2
import pandas as pd

# project dependencies
from deepface import DeepFace
from deepface.modules import verification
from deepface.commons import image_utils
from deepface.commons.logger import Logger

logger = Logger()


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
        assert identity_df["distance"].values[0] <= threshold

        df = df[df["identity"] != img_path]
        logger.debug(df.head())
        assert df.shape[0] > 0

        assert "confidence" in df.columns
        # confidence is between 0 and 100
        assert df["confidence"].max() <= 100
        assert df["confidence"].min() >= 0
        # also we just show verified ones in results
        assert df["confidence"].max() <= 100
        assert df["confidence"].min() >= 51

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
    # List
    list_imgs = image_utils.list_images("dataset")

    assert len(list_imgs) > 0

    # img47 is webp even though its extension is jpg
    assert "dataset/img47.jpg" not in list_imgs

    # Generator
    gen_imgs = list(image_utils.yield_images("dataset"))

    assert len(gen_imgs) > 0

    # img47 is webp even though its extension is jpg
    assert "dataset/img47.jpg" not in gen_imgs

    assert gen_imgs == list_imgs


def test_find_without_refresh_database():
    import shutil, hashlib

    img_path = os.path.join("dataset", "img1.jpg")

    # 1. Calculate hash of the .pkl file;
    # 2. Move random image to the temporary created directory;
    # 3. As a result, there will be a difference between the .pkl file and the disk files;
    # 4. If refresh_database=False, then .pkl file should not be updated.
    #    Recalculate hash and compare it with the hash from pt. 1;
    # 5. After successful check, the image will be moved back to the original destination;

    pkl_path = "dataset/ds_model_vggface_detector_opencv_aligned_normalization_base_expand_0.pkl"
    with open(pkl_path, "rb") as f:
        hash_before = hashlib.sha256(f.read())

    image_name = "img28.jpg"
    tmp_dir = "dataset/temp_image"
    os.mkdir(tmp_dir)
    shutil.move(os.path.join("dataset", image_name), os.path.join(tmp_dir, image_name))

    dfs = DeepFace.find(img_path=img_path, db_path="dataset", silent=True, refresh_database=False)

    with open(pkl_path, "rb") as f:
        hash_after = hashlib.sha256(f.read())

    shutil.move(os.path.join(tmp_dir, image_name), os.path.join("dataset", image_name))
    os.rmdir(tmp_dir)

    assert hash_before.hexdigest() == hash_after.hexdigest()

    logger.info("✅ .pkl hashes before and after the recognition process are the same")

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
    logger.info("✅ test find without refresh database done")
