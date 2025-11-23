from typing import Tuple

# these values are determined empirically for each model from unit test items
minmax_values = {
    "VGG-Face": (0.0, 0.27054874177488775),
    "Facenet": (-3.541942596435547, 3.247769594192505),
    "Facenet512": (-4.388302803039551, 3.643190622329712),
    "OpenFace": (-0.34191709756851196, 0.26318004727363586),
    "DeepFace": (0.0, 17.294939041137695),
    "DeepID": (0.0, 127.86836242675781),
    "Dlib": (-0.41398656368255615, 0.5201137661933899),
    "ArcFace": (-2.945136308670044, 2.087090015411377),
}


def get_minmax_values(model_name: str) -> Tuple[float, float]:
    if model_name not in minmax_values:
        return (0, 0)
    return minmax_values[model_name]
