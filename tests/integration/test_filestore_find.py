# built-in dependencies
import os
import io
import ftplib

# 3rd party dependencies
import pytest
import boto3
import pandas as pd
from lightdsa import LightDSA

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

# services in docker/docker-compose-filestore.yml
os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:8333")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "deepface_user")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "deepface_pass")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

BUCKET = "deepface"
FTP_HOST, FTP_PORT, FTP_USER, FTP_PASS = "localhost", 2121, "deepface_user", "deepface_pass"

DATASET = os.path.join(os.path.dirname(__file__), "..", "unit", "dataset")
IMAGES = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img6.jpg"]
TARGET = os.path.join(DATASET, "img1.jpg")

PKL = "ds_model_vggface_detector_opencv_aligned_normalization_base_expand_0.pkl"


def __ftp() -> ftplib.FTP:
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT)
    ftp.login(FTP_USER, FTP_PASS)
    return ftp


@pytest.fixture
def s3_prefix():
    prefix = "faces"
    s3 = boto3.client("s3")
    if BUCKET not in [bucket["Name"] for bucket in s3.list_buckets()["Buckets"]]:
        s3.create_bucket(Bucket=BUCKET)
    for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
        s3.delete_object(Bucket=BUCKET, Key=obj["Key"])
    for img in IMAGES:
        s3.upload_file(os.path.join(DATASET, img), BUCKET, f"{prefix}/sub/{img}")
    yield prefix


@pytest.fixture
def ftp_dir():
    directory = "/ftp/deepface_user/faces"
    ftp = __ftp()
    try:
        ftp.mkd(directory)
    except ftplib.error_perm:
        pass
    try:
        ftp.mkd(f"{directory}/sub")
    except ftplib.error_perm:
        pass
    for name in ftp.nlst(directory) + ftp.nlst(f"{directory}/sub"):
        try:
            ftp.delete(name if name.startswith("/") else f"{directory}/{name}")
        except ftplib.error_perm:
            pass  # a directory
    for img in IMAGES:
        with open(os.path.join(DATASET, img), "rb") as f:
            ftp.storbinary(f"STOR {directory}/sub/{img}", f)
    ftp.quit()
    yield directory


def __validate(dfs, db_prefix: str):
    assert len(dfs) > 0
    df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert all(identity.startswith(db_prefix) for identity in df["identity"])
    assert f"{db_prefix}/sub/img1.jpg" in df["identity"].values


def test_find_on_s3(s3_prefix):
    db_path = f"s3://{BUCKET}/{s3_prefix}"
    dfs = DeepFace.find(img_path=TARGET, db_path=db_path, silent=True)
    __validate(dfs, db_path)

    s3 = boto3.client("s3")
    s3.head_object(Bucket=BUCKET, Key=f"{s3_prefix}/{PKL}")  # raises if pickle not stored

    # add a new image and run again with the stored pickle
    s3.upload_file(os.path.join(DATASET, "img5.jpg"), BUCKET, f"{s3_prefix}/img5.jpg")
    dfs = DeepFace.find(img_path=TARGET, db_path=db_path, silent=True)
    __validate(dfs, db_path)
    assert f"{db_path}/img5.jpg" in dfs[0]["identity"].values
    logger.info("✅ find on s3 done")


def test_find_on_s3_signed(s3_prefix):
    db_path = f"s3://{BUCKET}/{s3_prefix}"
    cs = LightDSA(algorithm_name="eddsa")
    dfs = DeepFace.find(img_path=TARGET, db_path=db_path, silent=True, credentials=cs)
    __validate(dfs, db_path)
    boto3.client("s3").head_object(Bucket=BUCKET, Key=f"{s3_prefix}/{PKL}.ldsa")

    with pytest.raises(ValueError, match="signature file"):
        DeepFace.find(img_path=TARGET, db_path=db_path, silent=True)
    logger.info("✅ signed find on s3 done")


def test_find_on_ftp(ftp_dir):
    db_path = f"ftp://{FTP_USER}:{FTP_PASS}@{FTP_HOST}:{FTP_PORT}{ftp_dir}"
    identity_prefix = f"ftp://{FTP_HOST}:{FTP_PORT}{ftp_dir}"

    dfs = DeepFace.find(img_path=TARGET, db_path=db_path, silent=True)
    __validate(dfs, identity_prefix)

    ftp = __ftp()
    buffer = io.BytesIO()
    ftp.retrbinary(f"RETR {ftp_dir}/{PKL}", buffer.write)
    assert len(buffer.getvalue()) > 0

    # remove an image, it must be dropped from the stored pickle
    ftp.delete(f"{ftp_dir}/sub/img2.jpg")
    ftp.quit()
    dfs = DeepFace.find(img_path=TARGET, db_path=db_path, silent=True)
    __validate(dfs, identity_prefix)
    assert f"{identity_prefix}/sub/img2.jpg" not in dfs[0]["identity"].values
    logger.info("✅ find on ftp done")
