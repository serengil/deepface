# built-in dependencies
import bz2
import os
import zipfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock
import pytest

# project dependencies
from deepface.commons import folder_utils, weight_utils, package_utils
from deepface.commons.logger import Logger

# pylint: disable=unused-argument

logger = Logger()


def test_loading_broken_weights():
    # keras model is required to load h5 weights, skip on pytorch and onnx environments
    pytest.importorskip("tensorflow")

    # conditional imports
    if package_utils.get_tf_major_version() == 1:
        from keras.models import Sequential
        from keras.layers import Dropout, Dense
    else:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dropout, Dense

    home = folder_utils.get_deepface_home()
    weight_file = os.path.join(home, ".deepface/weights/vgg_face_weights.h5")

    # construct a dummy model
    model = Sequential()

    # Add layers to the model
    model.add(
        Dense(units=64, activation="relu", input_shape=(100,))
    )  # Input layer with 100 features
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(units=32, activation="relu"))  # Hidden layer
    model.add(Dense(units=10, activation="softmax"))  # Output layer with 10 classes

    # vgg's weights cannot be loaded to this model
    with pytest.raises(
        ValueError, match="An exception occurred while loading the pre-trained weights from"
    ):
        model = weight_utils.load_model_weights(model=model, weight_file=weight_file)

    logger.info("✅ test loading broken weight file is done")


@pytest.fixture(autouse=True)
def mock_requests_head():
    # sources are checked with a head request before being downloaded with gdown
    with mock.patch("requests.head") as mocked:
        mocked.return_value.status_code = 200
        yield mocked


@mock.patch("deepface.commons.folder_utils.get_deepface_home")  # Update with your actual module
@mock.patch("gdown.download")  # Mocking gdown's download function
@mock.patch("os.path.isfile")  # Mocking os.path.isfile
@mock.patch("os.makedirs")  # Mocking os.makedirs to avoid FileNotFoundError
@mock.patch("zipfile.ZipFile")  # Mocking the ZipFile class
@mock.patch("bz2.BZ2File")  # Mocking the BZ2File class
@mock.patch("builtins.open", new_callable=mock.mock_open())  # Mocking open
class TestDownloadWeightFeature:
    def test_download_weights_for_available_file(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedir: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        mock_isfile.return_value = True
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")

        file_name = "model_weights.h5"
        source_url = "http://example.com/model_weights.zip"

        result = weight_utils.download_weights_if_necessary(file_name, source_url)

        assert os.path.normpath(result) == os.path.normpath(
            os.path.join("/mock/home", ".deepface/weights", file_name)
        )

        mock_gdown.assert_not_called()
        mock_zipfile.assert_not_called()
        mock_bz2file.assert_not_called()
        logger.info("✅ test download weights for available file is done")

    def test_download_weights_if_necessary_gdown_failure(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedirs: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        # Setting up the mock return values
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")
        mock_isfile.return_value = False  # Simulate file not being present

        file_name = "model_weights.h5"
        source_url = "http://example.com/model_weights.h5"

        # Simulate gdown.download raising an exception
        mock_gdown.side_effect = Exception("Download failed!")

        # Call the function and check for ValueError
        with pytest.raises(
            ValueError,
            match=f"⛓️‍💥 An exception occurred while downloading {file_name} from {source_url}.",
        ):
            weight_utils.download_weights_if_necessary(file_name, source_url)

        logger.info("✅ test for downloading weights while gdown fails done")

    def test_download_weights_if_necessary_no_compression(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedir: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        # Setting up the mock return values
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")
        mock_isfile.return_value = False  # Simulate file not being present

        file_name = "model_weights.h5"
        source_url = "http://example.com/model_weights.h5"

        # Call the function
        result = weight_utils.download_weights_if_necessary(file_name, source_url)

        # Normalize the expected path
        expected_path = os.path.normpath("/mock/home/.deepface/weights/model_weights.h5")

        # Assert that gdown.download was called with the correct parameters
        mock_gdown.assert_called_once_with(source_url, expected_path, quiet=False)

        # Assert that the return value is correct
        assert result == expected_path

        # Assert that zipfile.ZipFile and bz2.BZ2File were not called
        mock_zipfile.assert_not_called()
        mock_bz2file.assert_not_called()

        logger.info("✅ test download weights with no compression is done")

    def test_download_weights_if_necessary_backup_source(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedir: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")
        mock_isfile.return_value = False

        file_name = "model_weights.h5"
        source_urls = [
            "http://example.com/model_weights.h5",
            "http://backup.example.com/model_weights.h5",
        ]

        # primary source fails, backup one succeeds
        mock_gdown.side_effect = [Exception("Download failed!"), None]

        result = weight_utils.download_weights_if_necessary(file_name, source_urls)

        expected_path = os.path.normpath("/mock/home/.deepface/weights/model_weights.h5")
        assert result == expected_path
        assert mock_gdown.call_args_list == [
            mock.call(source_urls[0], expected_path, quiet=False),
            mock.call(source_urls[1], expected_path, quiet=False),
        ]

        logger.info("✅ test download weights from backup source is done")

    def test_download_weights_if_necessary_backup_source_when_primary_not_found(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedir: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
        mock_requests_head: MagicMock,
    ):
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")
        mock_isfile.return_value = False

        file_name = "model_weights.h5"
        source_urls = [
            "http://example.com/model_weights.h5",
            "http://backup.example.com/model_weights.h5",
        ]

        # gdown does not raise for 404, so the primary source must be skipped before it
        not_found, found = MagicMock(status_code=404), MagicMock(status_code=200)
        mock_requests_head.side_effect = [not_found, found]

        weight_utils.download_weights_if_necessary(file_name, source_urls)

        expected_path = os.path.normpath("/mock/home/.deepface/weights/model_weights.h5")
        mock_gdown.assert_called_once_with(source_urls[1], expected_path, quiet=False)

        logger.info("✅ test download weights from backup source when primary is not found done")

    def test_download_weights_if_necessary_primary_source_first(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedir: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")
        mock_isfile.return_value = False

        file_name = "model_weights.h5"
        source_urls = [
            "http://example.com/model_weights.h5",
            "http://backup.example.com/model_weights.h5",
        ]

        weight_utils.download_weights_if_necessary(file_name, source_urls)

        # backup source must not be touched when the primary one succeeds
        expected_path = os.path.normpath("/mock/home/.deepface/weights/model_weights.h5")
        mock_gdown.assert_called_once_with(source_urls[0], expected_path, quiet=False)

        logger.info("✅ test download weights from primary source is done")

    def test_download_weights_if_necessary_all_sources_fail(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedir: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        mock_get_deepface_home.return_value = os.path.normpath("/mock/home")
        mock_isfile.return_value = False

        file_name = "model_weights.h5"
        source_urls = [
            "http://example.com/model_weights.h5",
            "http://backup.example.com/model_weights.h5",
        ]

        mock_gdown.side_effect = Exception("Download failed!")

        with pytest.raises(
            ValueError,
            match=f"⛓️‍💥 An exception occurred while downloading {file_name} from "
            f"{source_urls[0]}, {source_urls[1]}.",
        ):
            weight_utils.download_weights_if_necessary(file_name, source_urls)

        assert mock_gdown.call_count == 2

        logger.info("✅ test download weights while all sources fail is done")

    def test_download_weights_for_non_supported_compress_type(
        self,
        mock_open: MagicMock,
        mock_zipfile: MagicMock,
        mock_bz2file: MagicMock,
        mock_makedirs: MagicMock,
        mock_isfile: MagicMock,
        mock_gdown: MagicMock,
        mock_get_deepface_home: MagicMock,
    ):
        mock_isfile.return_value = False

        file_name = "model_weights.h5"
        source_url = "http://example.com/model_weights.bz2"
        compress_type = "7z"
        with pytest.raises(ValueError, match="unimplemented compress type - 7z"):
            _ = weight_utils.download_weights_if_necessary(file_name, source_url, compress_type)
        logger.info("✅ test download weights for unsupported compress type is done")


def test_download_weights_if_necessary_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    folder_utils.initialize_folder()
    weight = tmp_path / ".deepface" / "weights" / "model_weights.h5"
    companion = weight.with_name("companion.bin")
    companion.write_bytes(b"old companion")
    nested = weight.parent / "metadata" / "config.json"

    def download(url, output, quiet):
        assert not weight.exists()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("metadata/", b"")
            archive.writestr("metadata/config.json", b"{}")
            archive.writestr(companion.name, b"new companion")
            archive.writestr(weight.name, b"model weights")

    mocked_download = mock.Mock(side_effect=download)
    monkeypatch.setattr(weight_utils.gdown, "download", mocked_download)
    replacements = []
    replace = os.replace

    def publish(source, target):
        replacements.append(Path(target))
        replace(source, target)

    monkeypatch.setattr(weight_utils.os, "replace", publish)
    source_url = "http://example.com/model_weights.zip"
    result = weight_utils.download_weights_if_necessary(weight.name, source_url, "zip")

    assert Path(result) == weight
    assert weight.read_bytes() == b"model weights"
    assert companion.read_bytes() == b"new companion"
    assert nested.read_bytes() == b"{}"
    assert replacements[-1] == weight
    assert set(replacements) == {weight, companion, nested}
    mocked_download.assert_called_once_with(source_url, mock.ANY, quiet=False)
    assert set(weight.parent.iterdir()) == {weight, companion, nested.parent}


def test_download_weights_if_necessary_bz2(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    folder_utils.initialize_folder()
    weight = tmp_path / ".deepface" / "weights" / "model_weights.h5"

    def download(url, output, quiet):
        assert not weight.exists()
        Path(output).write_bytes(bz2.compress(b"model weights"))

    mocked_download = mock.Mock(side_effect=download)
    monkeypatch.setattr(weight_utils.gdown, "download", mocked_download)
    source_url = "http://example.com/model_weights.bz2"
    result = weight_utils.download_weights_if_necessary(weight.name, source_url, "bz2")

    assert Path(result) == weight
    assert weight.read_bytes() == b"model weights"
    mocked_download.assert_called_once_with(source_url, mock.ANY, quiet=False)
    assert list(weight.parent.iterdir()) == [weight]


@pytest.mark.parametrize("compression", ["zip", "bz2"])
def test_compressed_weight_fallback(tmp_path, monkeypatch, compression):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    folder_utils.initialize_folder()
    weight = tmp_path / ".deepface" / "weights" / "model.bin"
    cached_peer = weight.with_name("other.bin")
    cached_peer.write_bytes(b"other model")
    sources = []

    def download(url, output, quiet):
        sources.append(url)
        assert not weight.exists()
        if url == "primary":
            Path(output).write_bytes(b"truncated archive")
        elif compression == "bz2":
            Path(output).write_bytes(bz2.compress(b"real model"))
        else:
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr(weight.name, b"real model")

    monkeypatch.setattr(weight_utils.gdown, "download", download)
    result = weight_utils.download_weights_if_necessary(
        weight.name, ["primary", "backup"], compression
    )
    assert Path(result).read_bytes() == b"real model"
    assert sources == ["primary", "backup"]
    assert cached_peer.read_bytes() == b"other model"
    assert set(weight.parent.iterdir()) == {weight, cached_peer}


def test_zip_corrupt_companion_retries_before_publishing(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    folder_utils.initialize_folder()
    weight = tmp_path / ".deepface" / "weights" / "model.bin"
    companion = weight.with_name("companion.bin")
    companion.write_bytes(b"cached companion")
    sources = []

    def download(url, output, quiet):
        sources.append(url)
        # A failure in a later member must not publish the already extracted files.
        assert not weight.exists()
        assert companion.read_bytes() == b"cached companion"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(companion.name, b"updated companion")
            archive.writestr(weight.name, b"real model")
            archive.writestr("last.bin", b"last member data")
        if url == "primary":
            data = Path(output).read_bytes()
            Path(output).write_bytes(data.replace(b"last member data", b"bad! member data"))

    monkeypatch.setattr(weight_utils.gdown, "download", download)
    result = weight_utils.download_weights_if_necessary(weight.name, ["primary", "backup"], "zip")

    assert Path(result).read_bytes() == b"real model"
    assert sources == ["primary", "backup"]
    assert companion.read_bytes() == b"updated companion"
    last_member = weight.with_name("last.bin")
    assert last_member.read_bytes() == b"last member data"
    assert set(weight.parent.iterdir()) == {weight, companion, last_member}


def test_zip_missing_requested_weight(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
    folder_utils.initialize_folder()
    weight = tmp_path / ".deepface" / "weights" / "model.bin"

    def download(url, output, quiet):
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("different.bin", b"other model")

    monkeypatch.setattr(weight_utils.gdown, "download", download)
    with pytest.raises(ValueError, match="while downloading model.bin"):
        weight_utils.download_weights_if_necessary(weight.name, "source", "zip")
    assert list(weight.parent.iterdir()) == []
