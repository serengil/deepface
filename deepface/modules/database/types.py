from typing import List, TypedDict, Optional


class EmbeddingRecord(TypedDict):
    id: Optional[int]
    img_name: str
    face: Optional[List[List[List[float]]]]  # equivalent to float[][][]
    model_name: str
    detector_backend: str
    embedding: List[float]
    aligned: bool
    l2_normalized: bool
