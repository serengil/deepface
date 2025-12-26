# built-in dependencies
import os
from unittest import mock
from unittest.mock import MagicMock
import pytest

# project dependencies
from deepface.commons import folder_utils, weight_utils, package_utils
from deepface.commons.logger import Logger

# pylint: disable=unused-argument

logger = Logger()

tf_version = package_utils.get_tf_major_version()

# conditional imports
if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import (
        Dropout,
        Dense,
    )
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dropout,
        Dense,
    )


def test_loading_broken_weights():
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

    def test_download_weights_if_necessary_zip(
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
        source_url = "http://example.com/model_weights.zip"
        compress_type = "zip"

        # Call the function
        result = weight_utils.download_weights_if_necessary(file_name, source_url, compress_type)

        # Assert that gdown.download was called with the correct parameters
        mock_gdown.assert_called_once_with(
            source_url,
            os.path.normpath("/mock/home/.deepface/weights/model_weights.h5.zip"),
            quiet=False,
        )

        # Simulate the unzipping behavior
        mock_zipfile.return_value.__enter__.return_value.extractall = mock.Mock()

        # Call the function again to simulate unzipping
        with mock_zipfile.return_value as zip_ref:
            zip_ref.extractall(os.path.normpath("/mock/home/.deepface/weights"))

        # Assert that the zip file was unzipped correctly
        zip_ref.extractall.assert_called_once_with(os.path.normpath("/mock/home/.deepface/weights"))

        # Assert that the return value is correct
        assert result == os.path.normpath("/mock/home/.deepface/weights/model_weights.h5")

        logger.info("✅ test download weights for zip is done")

    def test_download_weights_if_necessary_bz2(
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
        source_url = "http://example.com/model_weights.bz2"
        compress_type = "bz2"

        # Simulate the download success
        mock_gdown.return_value = None

        # Simulate the BZ2 file reading behavior
        mock_bz2file.return_value.__enter__.return_value.read.return_value = b"fake data"

        # Call the function under test
        result = weight_utils.download_weights_if_necessary(file_name, source_url, compress_type)

        # Assert that gdown.download was called with the correct parameters
        mock_gdown.assert_called_once_with(
            source_url,
            os.path.normpath("/mock/home/.deepface/weights/model_weights.h5.bz2"),
            quiet=False,
        )

        # Ensure open() is called once for writing the decompressed data
        mock_open.assert_called_once_with(
            os.path.normpath("/mock/home/.deepface/weights/model_weights.h5"), "wb"
        )

        # TODO: find a way to check write is called

        # Assert that the return value is correct
        assert result == os.path.normpath("/mock/home/.deepface/weights/model_weights.h5")

        logger.info("✅ test download weights for bz2 is done")

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
