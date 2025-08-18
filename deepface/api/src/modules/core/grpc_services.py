import grpc

from deepface.commons.logger import Logger
from deepface.commons import image_utils
from deepface.api.proto.deepface_pb2 import AnalyzeRequest, AnalyzeResponse, RepresentResponse, VerifyResponse, Models, Detectors, DistanceMetrics
from deepface.api.proto.deepface_pb2_grpc import DeepFaceServiceServicer

from deepface import DeepFace

from deepface.api.src.modules.core.common import (
    default_detector_backend,
    default_enforce_detection,
    default_align,
    default_anti_spoofing,
    default_max_faces,
    default_model_name,
    default_distance_metric,
)

logger = Logger()

class DeepFaceService(DeepFaceServiceServicer):

    def Analyze(self, request, context) -> AnalyzeResponse:
        response = AnalyzeResponse()

        logger.info(f"Received Analyze request: {request}")

        try:
            demographies = DeepFace.analyze(
                img_path=image_utils.load_image_from_web(request.image_url),
                actions=actions_enum_to_string(request.actions),
                enforce_detection=request.enforce_detection
                if request.HasField("enforce_detection") else
                default_enforce_detection,
                detector_backend=Detectors.Name(request.detector_backend).lower() if 
                request.HasField("detector_backend") else default_detector_backend,
                align=request.align
                if request.HasField("align") else default_align,
                anti_spoofing=request.anti_spoofing if
                request.HasField("anti_spoofing") else default_anti_spoofing,
            )
        except Exception as err:
            context.set_details(f"Exception while analyzing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response
        
        logger.debug(f"Demographies received: {demographies}")

        for demography in demographies:
            if isinstance(demography, list):
                demography = demography[0]
            result = response.results.add()
            if "age" in demography:
                result.age = int(demography.get("age", 0))
            if "gender" in demography:
                result.gender.man = demography.get("gender", {}).get("Man", 0.0)
                result.gender.woman = demography.get("gender", {}).get("Woman", 0.0)
            if "face_confidence" in demography:
                result.face_confidence = float(demography.get("face_confidence", 0.0))
            if "dominant_emotion" in demography:
                result.dominant_emotion = demography.get("dominant_emotion", "")
            if "dominant_gender" in demography:
                result.dominant_gender = demography.get("dominant_gender", "")
            if "dominant_race" in demography:
                result.dominant_race = demography.get("dominant_race", "")
            if "race" in demography:
                race = demography.get("race", {})
                result.race.asian = race.get("asian", 0.0)
                result.race.indian = race.get("indian", 0.0)
                result.race.black = race.get("black", 0.0)
                result.race.white = race.get("white", 0.0)
                result.race.middle_eastern = race.get("middle eastern", 0.0)
                result.race.latino_hispanic = race.get("latino hispanic", 0.0)
            if "region" in demography:
                fill_facial_area(result.facial_area, demography["region"])
            if "emotion" in demography:
                emotion = demography.get("emotion", {})
                result.emotion.angry = emotion.get("angry", 0.0)
                result.emotion.disgust = emotion.get("disgust", 0.0)
                result.emotion.fear = emotion.get("fear", 0.0)
                result.emotion.happy = emotion.get("happy", 0.0)
                result.emotion.sad = emotion.get("sad", 0.0)
                result.emotion.surprise = emotion.get("surprise", 0.0)
                result.emotion.neutral = emotion.get("neutral", 0.0)

        return response

    def Represent(self, request, context) -> RepresentResponse:
        response = RepresentResponse()

        try:
            results = DeepFace.represent(
                img_path=image_utils.load_image_from_web(request.image_url),
                model_name=model_name_enum_to_string(request.model_name) if
                request.HasField("model_name") else default_model_name,
                detector_backend=Detectors.Name(request.detector_backend).lower() if
                request.HasField("detector_backend") else default_detector_backend,
                enforce_detection=request.enforce_detection if 
                request.HasField("enforce_detection") else default_enforce_detection,
                align=request.align
                if request.HasField("align") else default_align,
                anti_spoofing=request.anti_spoofing if
                request.HasField("anti_spoofing") else default_anti_spoofing,
                max_faces=request.max_faces
                if request.HasField("max_faces") else default_max_faces,
            )
        except Exception as err:
            context.set_details(f"Exception while representing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response

        for result in results:
            if isinstance(result, list):
                result = result[0]
            rep = response.results.add()
            if "embedding" in result:
                rep.embedding.extend(result["embedding"])
            if "face_confidence" in result:
                rep.face_confidence = float(result["face_confidence"])
            if "facial_area" in result:
                fill_facial_area(rep.facial_area, result["facial_area"])

        logger.debug(results)

        return response

    def Verify(self, request, context) -> VerifyResponse:
        response = VerifyResponse()

        try:

            results = DeepFace.verify(
                img1_path=image_utils.load_image_from_web(request.image1_url),
                img2_path=image_utils.load_image_from_web(request.image2_url),
                model_name=model_name_enum_to_string(request.model_name) if
                request.HasField("model_name") else default_model_name,
                detector_backend=Detectors.Name(request.detector_backend).lower() if
                request.HasField("detector_backend") else default_detector_backend,
                distance_metric=DistanceMetrics.Name(request.distance_metric).lower() if
                request.HasField("distance_metric") else default_distance_metric,
                align=request.align
                if request.HasField("align") else default_align,
                enforce_detection=request.enforce_detection
                if request.HasField("enforce_detection") else
                default_enforce_detection,
                anti_spoofing=request.anti_spoofing
                if request.HasField("anti_spoofing") else default_anti_spoofing,
            )
            if "verified" in results:
                response.verified = bool(results["verified"])
            if "distance" in results:
                response.distance = float(results["distance"])
            if "facial_areas" in results:
                facial_areas = results["facial_areas"]
                fill_facial_area(response.facial_areas.img1, facial_areas["img1"])
                fill_facial_area(response.facial_areas.img2, facial_areas["img2"])
            if "threshold" in results:
                response.threshold = float(results["threshold"])
            if "time" in results:
                response.time = float(results["time"])
            if "similarity_metric" in results:
                metric_str = results["similarity_metric"].upper()
                if hasattr(DistanceMetrics, metric_str):
                    response.similarity_metric = getattr(DistanceMetrics, metric_str)
                else:
                    response.similarity_metric = DistanceMetrics.COSINE
            if "detector_backend" in results:
                backend_str = results["detector_backend"].upper()
                if hasattr(Detectors, backend_str):
                    response.detector_backend = getattr(Detectors, backend_str)
                else:
                    response.detector_backend = Detectors.OPENCV
            if "model" in results:
                model_str = results["model"].upper()
                if hasattr(Models, model_str):
                    response.model = getattr(Models, model_str)
                else:
                    response.model = Models.VGG_FACE

        except Exception as err:
            context.set_details(f"Exception while representing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response

        logger.debug(results)

        return response


def actions_enum_to_string(actions) -> list[str]:
    """
    Convert the actions enum to a list of action names.
    """
    action_names = []
    for action in actions:
        match action:
            case AnalyzeRequest.Action.AGE:
                action_names.append("age")
            case AnalyzeRequest.Action.GENDER:
                action_names.append("gender")
            case AnalyzeRequest.Action.RACE:
                action_names.append("race")
            case AnalyzeRequest.Action.EMOTION:
                action_names.append("emotion")
    return action_names

def fill_facial_area(facial_area_msg, data: dict):
    """
    Fill a FacialArea protobuf message from a dict.
    """
    for key in ["left_eye", "right_eye", "mouth_left", "mouth_right", "nose"]:
        value = data.get(key)
        if value is None:
            value = [0]
        elif not isinstance(value, (list, tuple)):
            value = [value]
        value = [v if v is not None else 0 for v in value]
        getattr(facial_area_msg, key).extend(value)
    facial_area_msg.h = int(data.get("h") or 0)
    facial_area_msg.w = int(data.get("w") or 0)
    facial_area_msg.x = int(data.get("x") or 0)
    facial_area_msg.y = int(data.get("y") or 0)

def model_name_enum_to_string(model_name) -> str:
    """
    Convert a model name enum to a string.
    """
    match model_name:
        case Models.VGG_FACE:
            return "VGG-Face"
        case Models.OPENFACE:
            return "OpenFace"
        case Models.FACENET:
            return "Facenet"
        case Models.FACENET512:
            return "Facenet512"
        case Models.DEEPFACE:
            return "DeepFace"
        case Models.DEEPID:
            return "DeepID"
        case Models.DLIB_MODEL:
            return "Dlib"
        case Models.ARCFACE:
            return "ArcFace"
        case Models.SFACE:
            return "SFace"
        case Models.GHOSTFACENET:
            return "GhostFaceNet"
        case Models.BUFFALO_L:
            return "Buffalo_L"
        case _:
            return "unknown"
