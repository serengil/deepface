import argparse
from concurrent import futures

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

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

    threadPool = futures.ThreadPoolExecutor(max_workers=args.workers)
    server = grpc.server(threadPool)
    # Register the unified DeepFaceService
    deepface_grpc.add_DeepFaceServiceServicer_to_server(DeepFaceService(), server)
    # Register health checking service
    health_servicer = health.HealthServicer(
        experimental_non_blocking=True,
        experimental_thread_pool=threadPool,
    )
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    # Start the gRPC server
    logger.info(f"gRPC server running on port {args.port} with {args.workers} workers")
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")

    # Check if TensorFlow is using the GPU
    if tf.config.list_physical_devices('GPU'):
        logger.info(f"TensorFlow is using v{len(tf.config.list_physical_devices('GPU'))} GPU(s).")
    else:
        logger.info("TensorFlow is not using any GPU.")
    server.wait_for_termination()
