test:
	cd tests && python -m pytest . -s --disable-warnings

lint:
	python -m pylint deepface/ api/ --fail-under=10

coverage:
	pip install pytest-cov && cd tests && python -m pytest --cov=deepface