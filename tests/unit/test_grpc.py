# built-in dependencies
import unittest

# 3rd party dependencies
import cv2
import numpy as np
import pytest

# project dependencies
from deepface.commons.logger import Logger

# grpc is an optional dependency listed in requirements_additional.txt
grpc = pytest.importorskip("grpc", reason="grpcio is not installed")
json_format = pytest.importorskip("google.protobuf.json_format", reason="protobuf is not installed")
MessageToDict = json_format.MessageToDict

try:
    from deepface.api.proto import deepface_pb2, deepface_pb2_grpc
    from deepface.api.src.grpc_server import create_server
except ImportError:
    pytest.skip("gRPC stubs not generated, run `make grpc` first", allow_module_level=True)

logger = Logger()


def read_bytes(img_path: str) -> bytes:
    with open(img_path, "rb") as f:
        return f.read()


class TestGrpcApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, port = create_server(port=0)
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"localhost:{port}")
        cls.stub = deepface_pb2_grpc.DeepFaceServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(None)

    def test_represent_with_path(self):
        response = self.stub.Represent(
            deepface_pb2.RepresentRequest(
                img=deepface_pb2.Image(path="dataset/img1.jpg"), model_name="Facenet"
            )
        )
        result = MessageToDict(response.result)
        assert len(result["results"]) > 0
        for face in result["results"]:
            assert len(face["embedding"]) == 128
            assert face.get("facial_area") is not None
            assert face.get("face_confidence") is not None
        logger.info("✅ grpc represent with path test done")

    def test_verify_with_bytes(self):
        with open("dataset/img1.jpg", "rb") as f:
            img1 = f.read()
        with open("dataset/img2.jpg", "rb") as f:
            img2 = f.read()

        response = self.stub.Verify(
            deepface_pb2.VerifyRequest(
                img1=deepface_pb2.Image(content=img1),
                img2=deepface_pb2.Image(content=img2),
            )
        )
        result = MessageToDict(response.result)
        assert result["verified"] is True
        assert result["model"] == "VGG-Face"
        assert result.get("distance") is not None
        logger.info("✅ grpc verify with bytes test done")

    def test_analyze(self):
        response = self.stub.Analyze(
            deepface_pb2.AnalyzeRequest(
                img=deepface_pb2.Image(path="dataset/img1.jpg"), actions=["age", "gender"]
            )
        )
        result = MessageToDict(response.result)
        assert len(result["results"]) > 0
        for demography in result["results"]:
            assert demography.get("age") is not None
            assert demography.get("dominant_gender") is not None
            assert demography.get("dominant_emotion") is None
        logger.info("✅ grpc analyze test done")

    def test_missing_image(self):
        with pytest.raises(grpc.RpcError) as err:
            self.stub.Represent(deepface_pb2.RepresentRequest())
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        logger.info("✅ grpc missing image test done")

    def test_invalid_image(self):
        with pytest.raises(grpc.RpcError) as err:
            self.stub.Represent(
                deepface_pb2.RepresentRequest(img=deepface_pb2.Image(content=b"not an image"))
            )
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        logger.info("✅ grpc invalid image test done")

    def test_missing_file(self):
        with pytest.raises(grpc.RpcError) as err:
            self.stub.Represent(
                deepface_pb2.RepresentRequest(
                    img=deepface_pb2.Image(path="dataset/img_does_not_exist.jpg")
                )
            )
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        logger.info("✅ grpc missing file test done")

    def test_represent_bytes_and_path_give_same_embedding(self):
        embeddings = []
        for img in [
            deepface_pb2.Image(path="dataset/img1.jpg"),
            deepface_pb2.Image(content=read_bytes("dataset/img1.jpg")),
        ]:
            response = self.stub.Represent(
                deepface_pb2.RepresentRequest(img=img, model_name="Facenet")
            )
            result = MessageToDict(response.result)
            assert len(result["results"]) == 1
            embeddings.append(np.array(result["results"][0]["embedding"]))

        distance = np.linalg.norm(embeddings[0] - embeddings[1])
        assert distance < 1e-5, distance
        logger.info("✅ grpc represent bytes and path parity test done")

    def test_represent_with_png_bytes(self):
        img = cv2.imread("dataset/img1.jpg")
        is_encoded, png = cv2.imencode(".png", img)
        assert is_encoded

        response = self.stub.Represent(
            deepface_pb2.RepresentRequest(
                img=deepface_pb2.Image(content=png.tobytes()), model_name="Facenet"
            )
        )
        result = MessageToDict(response.result)
        assert len(result["results"]) == 1
        assert len(result["results"][0]["embedding"]) == 128
        logger.info("✅ grpc represent with png bytes test done")

    def test_represent_bytes_with_many_faces(self):
        img = deepface_pb2.Image(content=read_bytes("dataset/couple.jpg"))

        response = self.stub.Represent(
            deepface_pb2.RepresentRequest(img=img, model_name="Facenet")
        )
        assert len(MessageToDict(response.result)["results"]) > 1

        response = self.stub.Represent(
            deepface_pb2.RepresentRequest(img=img, model_name="Facenet", max_faces=1)
        )
        assert len(MessageToDict(response.result)["results"]) == 1
        logger.info("✅ grpc represent bytes with many faces test done")

    def test_verify_with_bytes_of_different_persons(self):
        response = self.stub.Verify(
            deepface_pb2.VerifyRequest(
                img1=deepface_pb2.Image(content=read_bytes("dataset/img1.jpg")),
                img2=deepface_pb2.Image(content=read_bytes("dataset/img3.jpg")),
            )
        )
        result = MessageToDict(response.result)
        assert result["verified"] is False
        logger.info("✅ grpc verify with bytes of different persons test done")

    def test_verify_with_bytes_and_path(self):
        response = self.stub.Verify(
            deepface_pb2.VerifyRequest(
                img1=deepface_pb2.Image(content=read_bytes("dataset/img1.jpg")),
                img2=deepface_pb2.Image(path="dataset/img2.jpg"),
            )
        )
        result = MessageToDict(response.result)
        assert result["verified"] is True
        logger.info("✅ grpc verify with bytes and path test done")

    def test_analyze_with_bytes(self):
        response = self.stub.Analyze(
            deepface_pb2.AnalyzeRequest(
                img=deepface_pb2.Image(content=read_bytes("dataset/img1.jpg")),
                actions=["age", "gender"],
            )
        )
        result = MessageToDict(response.result)
        assert len(result["results"]) > 0
        for demography in result["results"]:
            assert demography.get("age") is not None
            assert demography.get("dominant_gender") is not None
        logger.info("✅ grpc analyze with bytes test done")

    def test_bytes_without_face(self):
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        is_encoded, jpg = cv2.imencode(".jpg", noise)
        assert is_encoded
        img = deepface_pb2.Image(content=jpg.tobytes())

        with pytest.raises(grpc.RpcError) as err:
            self.stub.Represent(deepface_pb2.RepresentRequest(img=img))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        response = self.stub.Represent(
            deepface_pb2.RepresentRequest(img=img, enforce_detection=False)
        )
        assert len(MessageToDict(response.result)["results"]) == 1
        logger.info("✅ grpc bytes without face test done")

    def test_empty_bytes(self):
        with pytest.raises(grpc.RpcError) as err:
            self.stub.Represent(deepface_pb2.RepresentRequest(img=deepface_pb2.Image(content=b"")))
        assert err.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        logger.info("✅ grpc empty bytes test done")
