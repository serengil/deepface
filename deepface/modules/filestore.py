# built-in dependencies
import os
import io
import ftplib
import hashlib
import posixpath
from abc import ABC, abstractmethod
from typing import Any, List, Optional, cast
from urllib.parse import urlparse, unquote

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import cv2

# project dependencies
from deepface.commons import image_utils
from deepface.modules.exceptions import PathNotFound
from deepface.commons.logger import Logger

logger = Logger()


class FileStore(ABC):
    """
    Abstraction of the place where facial database images and the representations
    pickle live. Paths handled by a file store are the identities stored in the pickle.
    """

    @abstractmethod
    def validate(self) -> None:
        """Raise PathNotFound if the store is not reachable"""

    @abstractmethod
    def join(self, file_name: str) -> str:
        """Return the path of a file directly under the root of the store"""

    @abstractmethod
    def list_images(self) -> List[str]:
        """List image paths available in the store"""

    @abstractmethod
    def find_hash(self, path: str) -> str:
        """Find a cheap hash of a file based on its properties, not its content"""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check a file exists in the store"""

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read a file's content from the store"""

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write content into a file in the store"""

    def load_image(self, path: str) -> Any:
        """
        Load an image in a form that detection.extract_faces accepts
        Args:
            path (str): image path in the store
        Returns:
            img (str or np.ndarray): exact image path or image in BGR format
        """
        img = cv2.imdecode(np.frombuffer(self.read_bytes(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image {path}")
        return cast(NDArray[Any], img)


class LocalFileStore(FileStore):
    def __init__(self, root: str) -> None:
        self.root = root

    def validate(self) -> None:
        if not os.path.isdir(self.root):
            raise PathNotFound(f"Passed path {self.root} does not exist!")

    def join(self, file_name: str) -> str:
        return os.path.join(self.root, file_name)

    def list_images(self) -> List[str]:
        return list(image_utils.yield_images(path=self.root))

    def find_hash(self, path: str) -> str:
        return image_utils.find_image_hash(path)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)

    def load_image(self, path: str) -> Any:
        # extract_faces loads the image from its path itself
        return path


class S3FileStore(FileStore):
    """
    File store for s3://bucket/prefix paths. Credentials, region and a custom endpoint
    (e.g. MinIO) are resolved by boto3 itself - e.g. with AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION and AWS_ENDPOINT_URL environment variables.
    """

    def __init__(self, uri: str) -> None:
        # Import here to avoid mandatory dependency
        try:
            import boto3
        except (ModuleNotFoundError, ImportError) as e:
            raise ValueError(
                "boto3 is an optional dependency, ensure the library is installed."
                "Please install using 'pip install boto3' "
            ) from e

        parsed = urlparse(uri)
        if not parsed.netloc:
            raise ValueError(f"Bucket name is missing in {uri}")

        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = boto3.client("s3")

    def __key(self, path: str) -> str:
        parsed = urlparse(path)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(f"{path} is not in bucket {self.bucket}")
        return parsed.path.lstrip("/")

    def validate(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception as err:  # pylint: disable=broad-except
            raise PathNotFound(f"Passed bucket {self.bucket} is not accessible!") from err

    def join(self, file_name: str) -> str:
        return f"s3://{self.bucket}/{posixpath.join(self.prefix, file_name)}"

    def list_images(self) -> List[str]:
        images = []
        prefix = f"{self.prefix}/" if self.prefix else ""
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if posixpath.splitext(key)[1].lower() in image_utils.IMAGE_EXTS:
                    images.append(f"s3://{self.bucket}/{key}")
        return images

    def find_hash(self, path: str) -> str:
        head = self.client.head_object(Bucket=self.bucket, Key=self.__key(path))
        properties = f"{head['ContentLength']}-{head['ETag']}-{head['LastModified']}"
        return hashlib.sha1(properties.encode("utf-8")).hexdigest()

    def exists(self, path: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self.__key(path))
            return True
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def read_bytes(self, path: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self.__key(path))
        return cast(bytes, obj["Body"].read())

    def write_bytes(self, path: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.__key(path), Body=data)


class FtpFileStore(FileStore):
    """
    File store for ftp://user:password@host:port/path paths. If user or password is
    not in the uri, DEEPFACE_FTP_USER and DEEPFACE_FTP_PASSWORD environment variables
    are used, and anonymous login otherwise.
    """

    def __init__(self, uri: str) -> None:
        parsed = urlparse(uri)
        if not parsed.hostname:
            raise ValueError(f"Host is missing in {uri}")

        self.host = parsed.hostname
        self.port = parsed.port or 21
        self.user = unquote(parsed.username or os.environ.get("DEEPFACE_FTP_USER") or "")
        self.password = unquote(parsed.password or os.environ.get("DEEPFACE_FTP_PASSWORD") or "")
        self.root = "/" + unquote(parsed.path).strip("/")
        self.ftp: Optional[ftplib.FTP] = None

    def __connection(self) -> ftplib.FTP:
        if self.ftp is not None:
            try:
                self.ftp.voidcmd("NOOP")
                return self.ftp
            except ftplib.all_errors:
                self.ftp = None

        ftp = ftplib.FTP()
        ftp.connect(host=self.host, port=self.port, timeout=60)
        ftp.login(user=self.user, passwd=self.password)
        self.ftp = ftp
        return ftp

    def __path(self, path: str) -> str:
        parsed = urlparse(path)
        if parsed.scheme != "ftp" or parsed.hostname != self.host:
            raise ValueError(f"{path} is not in ftp server {self.host}")
        return unquote(parsed.path)

    def __uri(self, path: str) -> str:
        # identities do not carry credentials
        port = "" if self.port == 21 else f":{self.port}"
        return f"ftp://{self.host}{port}{path}"

    def __walk(self, ftp: ftplib.FTP, directory: str) -> List[str]:
        try:
            entries = [
                (name, facts.get("type") == "dir")
                for name, facts in ftp.mlsd(directory, facts=["type"])
                if facts.get("type") in {"dir", "file"}
            ]
        except ftplib.error_perm:
            # some servers (e.g. vsftpd) do not support MLSD
            entries = []
            for item in ftp.nlst(directory):
                name = posixpath.basename(item.rstrip("/"))
                try:
                    ftp.cwd(posixpath.join(directory, name))
                    is_dir = True
                except ftplib.error_perm:
                    is_dir = False
                entries.append((name, is_dir))

        files = []
        for name, is_dir in entries:
            if name in {".", ".."}:
                continue
            full_path = posixpath.join(directory, name)
            if is_dir:
                files += self.__walk(ftp, full_path)
            else:
                files.append(full_path)
        return files

    def validate(self) -> None:
        try:
            self.__connection().cwd(self.root)
        except ftplib.all_errors as err:
            raise PathNotFound(f"Passed path {self.__uri(self.root)} is not accessible!") from err

    def join(self, file_name: str) -> str:
        return self.__uri(posixpath.join(self.root, file_name))

    def list_images(self) -> List[str]:
        return [
            self.__uri(path)
            for path in self.__walk(self.__connection(), self.root)
            if posixpath.splitext(path)[1].lower() in image_utils.IMAGE_EXTS
        ]

    def find_hash(self, path: str) -> str:
        ftp = self.__connection()
        remote_path = self.__path(path)
        ftp.voidcmd("TYPE I")
        size = ftp.size(remote_path)
        modification_time = ftp.voidcmd(f"MDTM {remote_path}")
        properties = f"{size}-{modification_time}"
        return hashlib.sha1(properties.encode("utf-8")).hexdigest()

    def exists(self, path: str) -> bool:
        ftp = self.__connection()
        ftp.voidcmd("TYPE I")
        try:
            ftp.size(self.__path(path))
            return True
        except ftplib.error_perm:
            return False

    def read_bytes(self, path: str) -> bytes:
        buffer = io.BytesIO()
        self.__connection().retrbinary(f"RETR {self.__path(path)}", buffer.write)
        return buffer.getvalue()

    def write_bytes(self, path: str, data: bytes) -> None:
        self.__connection().storbinary(f"STOR {self.__path(path)}", io.BytesIO(data))


def build_file_store(db_path: str) -> FileStore:
    """
    Build the file store for the given db_path
    Args:
        db_path (str): local folder, s3://bucket/prefix or ftp://user:pass@host:port/path
    Returns:
        store (FileStore): file store with validated connection
    """
    scheme = urlparse(db_path).scheme.lower() if "://" in db_path else ""
    store: FileStore
    if scheme == "s3":
        store = S3FileStore(db_path)
    elif scheme == "ftp":
        store = FtpFileStore(db_path)
    elif len(scheme) <= 1:
        # no scheme, or a windows drive letter such as C://my_db
        store = LocalFileStore(db_path)
    else:
        raise ValueError(f"Unsupported db_path scheme {scheme}. Options: s3, ftp or local folder.")

    store.validate()
    return store
