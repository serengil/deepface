test:
	cd tests/unit && python -m pytest . -s --disable-warnings

test-tf:
	DEEPFACE_BACKEND_ENGINE=tensorflow $(MAKE) test

test-pytorch:
	DEEPFACE_BACKEND_ENGINE=pytorch $(MAKE) test

integration-test:
	cd tests/integration && python -m pytest . -s --disable-warnings

grpc:
	python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. deepface/api/proto/deepface.proto

grpc-server:
	cd deepface/api/src && python grpc_server.py

lint:
	python -m pylint deepface/ --fail-under=10 && mypy deepface/

coverage:
	pip install pytest-cov && cd tests/unit && python -m pytest --cov=deepface