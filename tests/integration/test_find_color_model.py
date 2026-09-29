"""Model-backed self-retrieval check using the existing project test image."""

from pathlib import Path
import shutil

import pytest

from deepface.modules import recognition, verification


@pytest.mark.parametrize("batched", [False, True])
def test_sface_self_retrieval_matches_verify(tmp_path, batched):
    source = Path(__file__).resolve().parents[1] / "unit" / "dataset" / "img1.jpg"
    image = str(tmp_path / "img1.jpg")
    shutil.copyfile(source, image)
    options = dict(
        model_name="SFace", detector_backend="opencv", distance_metric="euclidean"
    )
    verified = verification.verify(img1_path=image, img2_path=image, **options)
    result = recognition.find(
        img_path=image,
        db_path=str(tmp_path),
        batched=batched,
        similarity_search=True,
        silent=True,
        **options
    )[0]
    rows = result if batched else result.to_dict("records")
    assert len(rows) == 1
    assert rows[0]["identity"] == image
    assert verified["distance"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["distance"] == pytest.approx(verified["distance"], abs=1e-6)
