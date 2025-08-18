import argparse

import app

from deepface.commons.logger import Logger

logger = Logger()

if __name__ == "__main__":
    deepface_app = app.create_app()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port of serving api")
    args = parser.parse_args()

    # Start Flask HTTP server
    logger.info(f"HTTP server running on port {args.port}")
    deepface_app.run(host="0.0.0.0", port=args.port)
