from typing import List, TypedDict


class EmbeddingRecord(TypedDict):
    img_name: str
    face: List[List[List[float]]]  # equivalent to float[][][]
    model_name: str
    detector_backend: str
    embedding: List[float]
    aligned: bool
    l2_normalized: bool
