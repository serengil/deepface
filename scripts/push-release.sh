cd ..

echo "deleting existing release related files"
rm -rf dist/*
rm -rf build/*

echo "creating a package for current release - pypi compatible"
python setup.py sdist bdist_wheel

echo "pushing the release to pypi"
python -m twine upload dist/*