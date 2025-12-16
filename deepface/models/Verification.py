
from dataclasses import dataclass
from typing import Optional

from deepface.models.Face import Face

@dataclass
class Verification:
    face1: Optional[Face]
    face2: Optional[Face]
    verified: bool
    distance: float
    threshold: float
    detector_backend: str
    model: str
    metric: str
    time: float

    def to_dict(self):
        return {
            "face1": self.face1.to_dict() if self.face1 else None,
            "face2": self.face2.to_dict() if self.face2 else None,
            "verified": self.verified,
            "distance": self.distance,
            "threshold": self.threshold,
            "detector_backend": self.detector_backend,
            "model": self.model,
            "metric": self.metric,
            "time": self.time,
        }
