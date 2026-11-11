"""Grayscale extraction must accept the normalized faces returned by extraction."""

import numpy as np
import pytest

from deepface.modules import detection


COLORS = np.array(
    [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
    dtype=np.uint8,
)
NORMALIZED_LUMA = np.array([[0.114, 0.587], [0.299, 1.0]])
GRAY_OPTIONS = ({"color_face": "gray"}, {"grayscale": True})


@pytest.mark.parametrize("options", GRAY_OPTIONS)
def test_normalized_extraction_output_can_be_converted_to_gray(options):
    face = detection.extract_faces(
        COLORS, detector_backend="skip", color_face="bgr"
    )[0]["face"]
    assert face.dtype == np.float64
    before = face.copy()
    face.setflags(write=False)

    result = detection.extract_faces(face, detector_backend="skip", **options)[0]

    np.testing.assert_allclose(result["face"], NORMALIZED_LUMA, rtol=0, atol=1e-7)
    assert result["face"].shape == (2, 2)
    assert result["facial_area"]["w"] == 2
    assert result["facial_area"]["h"] == 2
    np.testing.assert_array_equal(face, before)


@pytest.mark.parametrize("options", GRAY_OPTIONS)
@pytest.mark.parametrize("normalize_face", [False, True])
def test_raw_float64_grayscale_preserves_the_requested_scale(options, normalize_face):
    # A non-contiguous view also checks that conversion never modifies the source.
    source = COLORS.astype(np.float64)[:, ::-1]
    before = source.copy()
    result = detection.extract_faces(
        source,
        detector_backend="skip",
        normalize_face=normalize_face,
        **options,
    )[0]["face"]

    scale = 1 if normalize_face else 255
    np.testing.assert_allclose(result, NORMALIZED_LUMA[:, ::-1] * scale, rtol=1e-7)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_existing_grayscale_input_dtypes_keep_their_pixel_values(dtype):
    source = COLORS.astype(dtype)
    expected = np.array([[29, 150], [76, 255]])
    if dtype == np.float32:
        source /= 255
        expected = NORMALIZED_LUMA

    result = detection.extract_faces(
        source, detector_backend="skip", color_face="gray", normalize_face=False
    )[0]["face"]

    assert result.dtype == dtype
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("array_batch", [False, True])
def test_float64_grayscale_batch_keeps_image_order(array_batch):
    source = COLORS.astype(np.float64) / 255
    images = [source, source[:, ::-1]]
    if array_batch:
        images = np.stack(images)

    results = detection.extract_faces(images, detector_backend="skip", color_face="gray")

    assert len(results) == 2
    for faces, expected in zip(results, [NORMALIZED_LUMA, NORMALIZED_LUMA[:, ::-1]]):
        assert len(faces) == 1
        np.testing.assert_allclose(faces[0]["face"], expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("color_face", ["rgb", "bgr"])
def test_float64_color_output_keeps_its_original_precision(color_face):
    source = COLORS.astype(np.float64) / 255
    source[0, 0, 0] = 0.123456789123456
    expected = source[:, :, ::-1] if color_face == "rgb" else source

    result = detection.extract_faces(
        source, detector_backend="skip", color_face=color_face, normalize_face=False
    )[0]["face"]

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)
