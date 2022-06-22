from deepface import DeepFace

dataset = [
    'dataset/img1.jpg',
	'dataset/img5.jpg',
	'dataset/img6.jpg',
    'dataset/img7.jpg',
    'dataset/img9.jpg',
    'dataset/img11.jpg',
    'dataset/img11.jpg',
]


def test_gender_prediction():
    detectors = ['opencv', 'ssd', 'retinaface', 'mtcnn']  # dlib not tested
    for detector in detectors:
        test_gender_prediction_with_detector(detector)


def test_gender_prediction_with_detector(detector):
    results = DeepFace.analyze(dataset, actions=('gender',), detector_backend=detector, prog_bar=False,
                               enforce_detection=False)
    for result in results:
        assert 'gender' in result.keys()
        assert 'dominant_gender' in result.keys() and result["dominant_gender"] in ["Man", "Woman"]
        if result["dominant_gender"] == "Man":
            assert result["gender"]["Man"] > result["gender"]["Woman"]
        else:
            assert result["gender"]["Man"] < result["gender"]["Woman"]
    print(f'detector {detector} passed')
    return True


if __name__ == "__main__":
    test_gender_prediction()
