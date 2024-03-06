import time
from typing import Any, Dict, Union
import numpy as np

def verify_embeddings(
    emb1: np.ndarray,
    emb2: np.ndarray,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
) -> Dict[str, Any]:
    tic = time.time()
    
    if distance_metric == "cosine":
        distance = find_cosine_distance(emb1, emb2)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(emb1, emb2)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(l2_normalize(emb1), l2_normalize(emb2))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    
    threshold = find_threshold(model_name, distance_metric)
    verified = distance <= threshold
    
    toc = time.time()
    
    return {
        "verified": verified,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "time": round(toc - tic, 2),
    }

def find_cosine_distance(source_representation, test_representation) -> np.float64:
    a = np.dot(source_representation, test_representation)
    b = np.linalg.norm(source_representation) * np.linalg.norm(test_representation)
    return 1 - (a / b)

def find_euclidean_distance(source_representation, test_representation) -> np.float64:
    return np.linalg.norm(source_representation - test_representation)

def l2_normalize(x) -> np.ndarray:
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def find_threshold(model_name: str, distance_metric: str) -> float:
    thresholds = {
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        # Add more models as needed
    }
    
    return thresholds.get(model_name, {}).get(distance_metric, 0.4)
