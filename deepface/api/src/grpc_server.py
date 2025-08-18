import argparse
from concurrent import futures

import grpc

from deepface import DeepFace
from deepface.commons.logger import Logger

import tensorflow as tf

# Import your generated gRPC module and service implementation for the unified service
import deepface.api.proto.deepface_pb2_grpc as deepface_grpc
from deepface.api.src.modules.core.grpc_services import DeepFaceService

logger = Logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=50051, help="Port of serving api")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Maximum worker threads")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    # Register the unified DeepFaceService
    deepface_grpc.add_DeepFaceServiceServicer_to_server(DeepFaceService(), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    # Start the gRPC server
    logger.info(f"gRPC server running on port {args.port}")
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")

    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        logger.info(f"TensorFlow is using v{len(tf.config.list_physical_devices('GPU'))} GPU(s).")
    else:
        logger.info("TensorFlow is not using any GPU.")
    server.wait_for_termination()
