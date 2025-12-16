import math
import grpc
from typing import Optional

from deepface.commons.logger import Logger
from deepface.commons import image_utils
from deepface.api.proto.deepface_pb2 import AnalyzeRequest, AnalyzeResponse, RepresentResponse, VerifyResponse, Models, Detectors, DistanceMetrics
from deepface.api.proto.deepface_pb2_grpc import DeepFaceServiceServicer

from deepface import DeepFace
from deepface.models.Face import FaceLandmarks, FacePose, FaceSpoofing, FaceQuality

from deepface.api.src.modules.core.common import (
    default_detector_backend,
    default_anti_spoofing,
    default_max_faces,
    default_model_name,
    default_distance_metric,
)

logger = Logger()

class DeepFaceService(DeepFaceServiceServicer):

    def Analyze(self, request, context) -> AnalyzeResponse:
        response = AnalyzeResponse()

        logger.info(f"Received Analyze request: {request.image_url}")

        try:
            faces = DeepFace.analyze(
                img_path=image_utils.load_image_from_web(request.image_url),
                actions=actions_enum_to_string(request.actions),
                detector_backend=Detectors.Name(request.detector_backend).lower() if request.HasField("detector_backend") else default_detector_backend,
                anti_spoofing=request.anti_spoofing if request.HasField("anti_spoofing") else default_anti_spoofing
            )
        except Exception as err:
            context.set_details(f"Exception while analyzing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response
        
        logger.debug(f"{len(faces)} faces received: {faces}")

        for face in faces:
            result = response.results.add()
            result.face_confidence = face.confidence or 0.0
            landmarks_to_proto(face.landmarks, result.facial_area)
            face_quality_to_proto(face.quality, result.face_quality)
            face_pose_to_proto(face.pose, result.face_pose)
            spoofing_scores_to_proto(face.spoofing, result.spoofing_scores)

            if face.demographics is not None:
                result.age = face.demographics.age or 0

                result.dominant_gender = face.demographics.dominant_gender() or ""
                result.gender.man = face.demographics.genders.get("Man", 0.0)
                result.gender.woman = face.demographics.genders.get("Woman", 0.0)

                result.dominant_emotion = face.demographics.dominant_emotion() or ""
                result.emotion.angry = face.demographics.emotions.get("angry", 0.0)
                result.emotion.disgust = face.demographics.emotions.get("disgust", 0.0)
                result.emotion.fear = face.demographics.emotions.get("fear", 0.0)
                result.emotion.happy = face.demographics.emotions.get("happy", 0.0)
                result.emotion.sad = face.demographics.emotions.get("sad", 0.0)
                result.emotion.surprise = face.demographics.emotions.get("surprise", 0.0)
                result.emotion.neutral = face.demographics.emotions.get("neutral", 0.0)
    
                result.dominant_race = face.demographics.dominant_race() or ""
                result.race.asian = face.demographics.races.get("asian", 0.0)
                result.race.indian = face.demographics.races.get("indian", 0.0)
                result.race.black = face.demographics.races.get("black", 0.0)
                result.race.white = face.demographics.races.get("white", 0.0)
                result.race.middle_eastern = face.demographics.races.get("middle eastern", 0.0)
                result.race.latino_hispanic = face.demographics.races.get("latino hispanic", 0.0)

        return response

    def Represent(self, request, context) -> RepresentResponse:
        response = RepresentResponse()

        logger.info(f"Received Represent request: {request.image_url}")

        try:
            faces = DeepFace.represent(
                img_path=image_utils.load_image_from_web(request.image_url),
                model_name=model_name_enum_to_string(request.model_name) if request.HasField("model_name") else default_model_name,
                detector_backend=Detectors.Name(request.detector_backend).lower() if request.HasField("detector_backend") else default_detector_backend,
                anti_spoofing=request.anti_spoofing if request.HasField("anti_spoofing") else default_anti_spoofing,
                max_faces=request.max_faces if request.HasField("max_faces") else default_max_faces
            )
        except Exception as err:
            context.set_details(f"Exception while representing: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response

        logger.debug(f"{len(faces)} faces received: {faces}")

        for face in faces:
            result = response.results.add()
            result.face_confidence = face.confidence or 0.0
            landmarks_to_proto(face.landmarks, result.facial_area)
            face_quality_to_proto(face.quality, result.face_quality)
            face_pose_to_proto(face.pose, result.face_pose)
            spoofing_scores_to_proto(face.spoofing, result.spoofing_scores)
            if face.embedding is not None:
                result.embedding.extend(face.embedding)

        return response

    def Verify(self, request, context) -> VerifyResponse:
        response = VerifyResponse()

        logger.info(f"Received Verify request: {request.image1_url} vs {request.image2_url}")

        try:

            verif = DeepFace.verify(
                img1_path=image_utils.load_image_from_web(request.image1_url),
                img2_path=image_utils.load_image_from_web(request.image2_url),
                model_name=model_name_enum_to_string(request.model_name) if request.HasField("model_name") else default_model_name,
                detector_backend=Detectors.Name(request.detector_backend).lower() if request.HasField("detector_backend") else default_detector_backend,
                distance_metric=DistanceMetrics.Name(request.distance_metric).lower() if request.HasField("distance_metric") else default_distance_metric,
                anti_spoofing=request.anti_spoofing if request.HasField("anti_spoofing") else default_anti_spoofing
            )

            logger.debug(f"verif response received: {verif}")

            response.verified = verif.verified or False
            response.distance = verif.distance or math.inf
            response.threshold = verif.threshold or math.inf
            response.time = verif.time or 0.0
            if verif.metric is not None:
                metric_str = verif.metric.upper()
                if hasattr(DistanceMetrics, metric_str):
                    response.similarity_metric = getattr(DistanceMetrics, metric_str)
                else:
                    response.similarity_metric = DistanceMetrics.COSINE
            if verif.detector_backend is not None:
                backend_str = verif.detector_backend.upper()
                if hasattr(Detectors, backend_str):
                    response.detector_backend = getattr(Detectors, backend_str)
                else:
                    response.detector_backend = Detectors.OPENCV
            if verif.model is not None:
                model_str = verif.model.upper()
                if hasattr(Models, model_str):
                    response.model = getattr(Models, model_str)
                else:
                    response.model = Models.VGG_FACE
                    response.detector_backend = Detectors.OPENCV
            if verif.face1 is not None:
                landmarks_to_proto(verif.face1.landmarks, response.facial_areas.img1)
                spoofing_scores_to_proto(verif.face1.spoofing, response.img1_spoofing_scores)
                face_quality_to_proto(verif.face1.quality, response.img1_face_quality)
                face_pose_to_proto(verif.face1.pose, response.img1_face_pose)
                if verif.face1.embedding is not None:
                    response.img1_embedding.extend(verif.face1.embedding)
            if verif.face2 is not None:
                landmarks_to_proto(verif.face2.landmarks, response.facial_areas.img2)
                spoofing_scores_to_proto(verif.face2.spoofing, response.img2_spoofing_scores)
                face_quality_to_proto(verif.face2.quality, response.img2_face_quality)
                face_pose_to_proto(verif.face2.pose, response.img2_face_pose)
                if verif.face2.embedding is not None:
                    response.img2_embedding.extend(verif.face2.embedding)

        except Exception as err:
            context.set_details(f"Exception while verifying: {str(err)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return response

        return response

def landmarks_to_proto(landmarks: Optional[FaceLandmarks], proto_landmarks):
    """
    Convert landmarks to proto format.
    """
    if landmarks is None:
        return
    proto_landmarks.left_eye.extend(landmarks.left_eye or [])
    proto_landmarks.right_eye.extend(landmarks.right_eye or [])
    proto_landmarks.mouth_left.extend(landmarks.mouth_left or [])
    proto_landmarks.mouth_right.extend(landmarks.mouth_right or [])
    proto_landmarks.nose.extend(landmarks.nose or [])

def face_quality_to_proto(quality: Optional[FaceQuality], proto_quality):
    """
    Convert face quality to proto format.
    """
    if quality is None:
        return
    proto_quality.sharpness = quality.sharpness or 0.0
    proto_quality.brightness = quality.brightness or 0.0
    proto_quality.contrast = quality.contrast or 0.0

def face_pose_to_proto(pose: Optional[FacePose], proto_pose):
    """
    Convert face pose to proto format.
    """
    if pose is None:
        return
    proto_pose.yaw = pose.yaw or 0.0
    proto_pose.pitch = pose.pitch or 0.0
    proto_pose.roll = pose.roll or 0.0

def spoofing_scores_to_proto(spoofing: Optional[FaceSpoofing], proto_spoofing_scores):
    """
    Convert spoofing scores to proto format.
    """
    if spoofing is None:
        return
    proto_spoofing_scores.spoof_confidence = spoofing.spoof_confidence or 0.0
    proto_spoofing_scores.real_confidence = spoofing.real_confidence or 0.0
    proto_spoofing_scores.uncertainty = spoofing.uncertainty_confidence or 0.0

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
