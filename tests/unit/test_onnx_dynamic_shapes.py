"""build_session(dynamic_shapes=True) must accept inputs in shapes other than declared ones."""

import numpy as np
import pytest

from deepface.commons import onnx_utils

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def make_model():
    # (1, 3, 4, 4) declared conv graph, which works on any spatial size
    weight = onnx.numpy_helper.from_array(np.ones((2, 3, 1, 1), dtype=np.float32), "w")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Conv", ["x", "w"], ["y"])],
        "fixed",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 4, 4])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 4, 4])],
        [weight],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def test_strip_io_shapes_drops_only_shapes():
    model = make_model()
    stripped = onnx.load_from_string(onnx_utils.strip_io_shapes(model.SerializeToString()))

    for value_info in list(model.graph.input) + list(model.graph.output):
        value_info.type.tensor_type.ClearField("shape")
    assert stripped.SerializeToString() == model.SerializeToString()


def test_stripped_graph_accepts_other_shapes():
    raw = make_model().SerializeToString()
    img = np.ones((2, 3, 8, 6), dtype=np.float32)

    with pytest.raises(Exception):
        ort.InferenceSession(raw).run(None, {"x": img})

    session = ort.InferenceSession(onnx_utils.strip_io_shapes(raw))
    np.testing.assert_allclose(session.run(None, {"x": img})[0], np.full((2, 2, 8, 6), 3))


def test_varint_round_trip():
    for value in [0, 1, 127, 128, 300, 2**32, 2**63 - 1]:
        assert onnx_utils.read_varint(onnx_utils.encode_varint(value), 0) == (
            value,
            len(onnx_utils.encode_varint(value)),
        )
