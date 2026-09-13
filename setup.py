import json
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def parse_requirements(file_name):
    """Read a requirements file, skipping its blank lines and comments"""
    with open(file_name, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith("#")]


requirements = parse_requirements("requirements.txt")

# deepface runs either on tensorflow or on pytorch, and neither of them is a base
# requirement. `pip install deepface[tensorflow]` and `pip install deepface[pytorch]`
# install the backend engine you want, without dragging the other one in.
tensorflow_requirements = parse_requirements("requirements_tf.txt")
pytorch_requirements = parse_requirements("requirements_pytorch.txt")

with open("package_info.json", "r", encoding="utf-8") as f:
    package_info = json.load(f)

setuptools.setup(
    name="deepface",
    version=package_info["version"],
    author="Sefik Ilkin Serengil",
    author_email="serengil@gmail.com",
    description=(
        "A Lightweight Face Recognition and Facial Attribute Analysis Framework"
        " (Age, Gender, Emotion, Race) for Python"
    ),
    data_files=[
        (
            "",
            [
                "README.md",
                "requirements.txt",
                # TODO: use requirements_base.txt instead of requirements.txt in the next release, and remove requirements.txt from the package. This is a breaking change, so it should be done in a major release.
                # "requirements_base.txt",
                "requirements_tf.txt",
                "requirements_pytorch.txt",
                "package_info.json",
            ],
        )
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/serengil/deepface",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["deepface = deepface.DeepFace:cli"],
    },
    python_requires=">=3.7",
    license="MIT",
    install_requires=requirements,
    extras_require={
        "tensorflow": tensorflow_requirements,
        "pytorch": pytorch_requirements,
    },
)
