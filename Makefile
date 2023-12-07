test:
	cd tests && python -m pytest unit_tests.py -s --disable-warnings

lint:
	python -m pylint deepface/ --fail-under=10