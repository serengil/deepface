"""Reject corrupt protobuf fields before rewriting ONNX graph shapes."""

import pytest

from deepface.commons import onnx_utils


@pytest.mark.parametrize(
    "data", [b"", b"\x80", b"\x80" * 10 + b"\x00", b"\xff" * 9 + b"\x02"]
)
def test_invalid_varints_raise_value_error(data):
    with pytest.raises(ValueError, match="varint"):
        onnx_utils.read_varint(data, 0)


@pytest.mark.parametrize("pos", [-1, 1, 100])
def test_varint_offset_must_be_in_message(pos):
    with pytest.raises(ValueError, match="varint"):
        onnx_utils.read_varint(b"\x00", pos)


@pytest.mark.parametrize(
    "data", [b"\x09abc", b"\x0dabc", b"\x12\x05abc", b"\x00\x00", b"\x3a\x05abc"]
)
def test_incomplete_or_invalid_fields_are_not_silently_copied(data):
    with pytest.raises(ValueError, match="protobuf"):
        onnx_utils.strip_io_shapes(data)


def test_complete_unknown_fields_are_preserved():
    # varint, fixed64, fixed32, length-delimited fields unrelated to graph shapes
    data = b"\x08\xac\x02\x11abcdefgh\x1dABCD\x22\x03xyz"
    assert onnx_utils.strip_io_shapes(data) == data


def test_largest_uint64_varint_round_trip():
    encoded = onnx_utils.encode_varint(2**64 - 1)
    assert onnx_utils.read_varint(encoded, 0) == (2**64 - 1, 10)
