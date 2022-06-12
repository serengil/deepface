from deepface import DeepFace

dataset = [
	'dataset/img1.jpg',
	'dataset/img5.jpg',
	'dataset/img6.jpg',
	'dataset/img8.jpg',
    'dataset/img7.jpg',
    'dataset/img9.jpg',
    'dataset/img11.jpg',
    'dataset/img11.jpg',
]

detectors = ['opencv', 'ssd', 'retinaface', 'mtcnn']  # dlib not tested


def test_gender_prediction():
    for detector in detectors:
        results = DeepFace.analyze(dataset, actions=('gender',), detector_backend=detector, prog_bar=False)
        for key in results.keys():
            result = results[key]
            assert 'gender' in result.keys()
            assert 'dominant_gender' in result.keys() and result["dominant_gender"] in ["Man", "Woman"]
            if result["dominant_gender"] == "Man":
                assert result["gender"]["Man"] > result["gender"]["Woman"]
            else:
                assert result["gender"]["Man"] < result["gender"]["Woman"]
        print(f'detector {detector} passed')


if __name__ == "__main__":
    test_gender_prediction()
