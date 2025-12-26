# built-in dependencies
import uuid


# 3rd party dependencies
import pytest
from lightphe import LightPHE
from lightphe.models.Tensor import EncryptedTensor


# project dependencies
from deepface import DeepFace
from deepface.config.threshold import thresholds

experiment_id = uuid.uuid4()
cs = LightPHE(algorithm_name="Paillier", precision=19)
cs.export_keys(f"/tmp/{experiment_id}_secret.txt")
cs.export_keys(f"/tmp/{experiment_id}_public.txt", public=True)


def test_no_encrypt():
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="VGG-Face",
    )

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert "encrypted_embedding" not in result


def test_encrypt_single():
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="VGG-Face",
        cryptosystem=cs,
    )

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert "encrypted_embedding" in result
    encrypted_embedding = result["encrypted_embedding"]
    assert encrypted_embedding is not None
    assert isinstance(encrypted_embedding, EncryptedTensor)
    assert isinstance(encrypted_embedding.fractions, list)
    assert len(encrypted_embedding.fractions) == 4096  # vgg-face embedding


def test_encrypt_batch():
    results = DeepFace.represent(
        img_path=["dataset/img1.jpg", "dataset/couple.jpg", "dataset/img2.jpg"],
        model_name="VGG-Face",
        cryptosystem=cs,
    )

    assert isinstance(results, list)
    assert len(results) == 3

    expected_faces = [1, 2, 1]
    for idx, current_results in enumerate(results):
        assert isinstance(current_results, list)
        assert len(current_results) == expected_faces[idx]
        for result in current_results:
            assert "encrypted_embedding" in result
            encrypted_embedding = result["encrypted_embedding"]
            assert encrypted_embedding is not None
            assert isinstance(encrypted_embedding, EncryptedTensor)
            assert isinstance(encrypted_embedding.fractions, list)
            assert len(encrypted_embedding.fractions) == 4096  # vgg-face embedding


def test_encrypt_single_skip_case():
    """
    Test that encryption is skipped when embeddings contain negative values.
    """
    # Dim values of Facenet are roughly in [-3, 3], so no normalization leads to negative values.
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        cryptosystem=cs,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert "encrypted_embedding" not in result

    # secondly, facenet embeddings aren't l2 normalized by default similar to vgg-face
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        cryptosystem=cs,
        minmax_normalize=True,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert "encrypted_embedding" not in result

    # do l2 normalization and min max normalization to encrypt
    results = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="Facenet",
        cryptosystem=cs,
        minmax_normalize=True,
        l2_normalize=True,
    )
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert "encrypted_embedding" in result
    encrypted_embedding = result["encrypted_embedding"]
    assert isinstance(encrypted_embedding, EncryptedTensor)
    assert isinstance(encrypted_embedding.fractions, list)
    assert len(encrypted_embedding.fractions) == 128  # facenet embedding


def test_encrypt_batch_skip_case():
    # Dim values of Facenet are roughly in [-3, 3], so no normalization leads to negative values.
    results = DeepFace.represent(
        img_path=["dataset/img1.jpg", "dataset/couple.jpg", "dataset/img2.jpg"],
        model_name="Facenet",
        cryptosystem=cs,
    )
    assert isinstance(results, list)
    assert len(results) == 3
    expected_faces = [1, 2, 1]
    for idx, result in enumerate(results):
        assert isinstance(result, list)
        assert len(result) == expected_faces[idx]
        for res in result:
            assert "encrypted_embedding" not in res

    # secondly, facenet embeddings aren't l2 normalized by default similar to vgg-face
    results = DeepFace.represent(
        img_path=["dataset/img1.jpg", "dataset/couple.jpg", "dataset/img2.jpg"],
        model_name="Facenet",
        cryptosystem=cs,
        minmax_normalize=True,
    )
    assert isinstance(results, list)
    assert len(results) == 3
    expected_faces = [1, 2, 1]
    for idx, result in enumerate(results):
        assert isinstance(result, list)
        assert len(result) == expected_faces[idx]
        for res in result:
            assert "encrypted_embedding" not in res

    # do l2 normalization and min max normalization to encrypt
    results = DeepFace.represent(
        img_path=["dataset/img1.jpg", "dataset/couple.jpg", "dataset/img2.jpg"],
        model_name="Facenet",
        cryptosystem=cs,
        minmax_normalize=True,
        l2_normalize=True,
    )
    assert isinstance(results, list)
    assert len(results) == 3
    expected_faces = [1, 2, 1]
    for idx, result in enumerate(results):
        assert isinstance(result, list)
        assert len(result) == expected_faces[idx]
        for res in result:
            assert "encrypted_embedding" in res
            encrypted_embedding = res["encrypted_embedding"]
            assert encrypted_embedding is not None
            assert isinstance(encrypted_embedding, EncryptedTensor)
            assert isinstance(encrypted_embedding.fractions, list)
        assert len(encrypted_embedding.fractions) == 128  # facenet embedding


def test_homomorphic_encryption():
    # this only has public key
    cloud_cs = LightPHE(
        algorithm_name="Paillier", precision=19, key_file=f"/tmp/{experiment_id}_public.txt"
    )

    encrypted_embedding = DeepFace.represent(
        img_path="dataset/img2.jpg",
        model_name="VGG-Face",
        cryptosystem=cloud_cs,
    )[0]["encrypted_embedding"]

    # cloud system should not be able to decrypt
    with pytest.raises(ValueError, match="must have private key"):
        cloud_cs.decrypt(encrypted_embedding)

    plain_embedding = DeepFace.represent(
        img_path="dataset/img1.jpg",
        model_name="VGG-Face",
    )[
        0
    ]["embedding"]

    # compute cosine similarity in encrypted domain
    encrypted_cosine_similarity = encrypted_embedding @ plain_embedding

    # on-prem system has private key
    onprem_cs = LightPHE(
        algorithm_name="Paillier", precision=19, key_file=f"/tmp/{experiment_id}_secret.txt"
    )

    # on-prem system can decrypt the result
    decrypted_cosine_similarity = onprem_cs.decrypt(encrypted_cosine_similarity)[0]
    decrypted_cosine_distance = 1 - decrypted_cosine_similarity

    threshold = thresholds["VGG-Face"]["cosine"]

    assert decrypted_cosine_distance < threshold
