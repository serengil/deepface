from deepface import DeepFace

dataset = [
    'https://datasets-626827236627.s3.amazonaws.com/avatars/orly-hamzani-046368aa_06122022.jpg?AWSAccessKeyId=ASIAZD4OQSUJ3IBDFZUN&Signature=pnRMEwKHNA5PfwDgHNoGavUxZvM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEID%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCfwrDPDOtQ7qval0EdjUQEhah2PvnNeJmO3KqjRmSzzQIhALmpYTK%2BIGxCNwfBBqtqCvme8cAhS6S2LCc6ti4rHC6VKtsECIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjI2ODI3MjM2NjI3IgxmYcKEgfQcYn6Ktf0qrwSTcdMXW6jBfrKQn1FtoasmBah2JYCy2X%2Bb07cLFKKNrwH7YLFlCoXiKWP6ntzrm0R2wSsr%2BTmn6RH8WoiXnnaC%2BagFiyUdqPlTdBGy2L%2BO2EKBNA0FnRx%2FnHR0rLG%2Fmcv7cSVcG%2Bsthv5nVSafUgDlQ8dLlcW%2FiGk10eSSAM2tMcUOPyoOLlsSJ%2B0RiSppJEhy3%2F5S63p7RT2fVdlSE1XH%2BnQYvIoUEw0uB3rVywTMknFFi7h8teki%2BGE%2BuONqcHwxSjfnDE53DzZZmmZ%2F5yW2TiK3KbbV974PVgGxA4epRwFCGKqmf7%2FeQCxjmOhPpvXL%2FWiltAIIxtUxlx6IL6IfTsf8i0bVU9cO4Jj14r5%2BdGt96h%2FHGftm%2BdBp1v7JD0vBIxiwFFgv073YjBBmg4gUoryI%aqaaaaaaaa',
    'https://datasets-626827236627.s3.amazonaws.com/avatars/%25D7%25A9%25D7%2599-%25D7%259E%25D7%2595%25D7%25A2%25D7%259C%25D7%259D-011174120_06122022.jpg?AWSAccessKeyId=ASIAZD4OQSUJ3IBDFZUN&Signature=QPaXXLUwBvflmxMEcJ98TiSvMTk%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEID%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCfwrDPDOtQ7qval0EdjUQEhah2PvnNeJmO3KqjRmSzzQIhALmpYTK%2BIGxCNwfBBqtqCvme8cAhS6S2LCc6ti4rHC6VKtsECIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjI2ODI3MjM2NjI3IgxmYcKEgfQcYn6Ktf0qrwSTcdMXW6jBfrKQn1FtoasmBah2JYCy2X%2Bb07cLFKKNrwH7YLFlCoXiKWP6ntzrm0R2wSsr%2BTmn6RH8WoiXnnaC%2BagFiyUdqPlTdBGy2L%2BO2EKBNA0FnRx%2FnHR0rLG%2Fmcv7cSVcG%2Bsthv5nVSafUgDlQ8dLlcW%2FiGk10eSSAM2tMcUOPyoOLlsSJ%2B0RiSppJEhy3%2F5S63p7RT2fVdlSE1XH%2BnQYvIoUEw0uB3rVywTMknFFi7h8teki%2BGE%2BuONqcHwxSjfnDE53DzZZmmZ%2F5yW2TiK3KbbV974PVgGxA4epRwFCGKqmf7%2FeQCxjmOhPpvXL%2FWiltAIIxtUxlx6IL6IfTsf8i0bVU9cO4Jj14r5%2BdGt96h%2FHGftm%2BdBp1v7JD0vBIxiwFFgv073YjBBmg4gUoryI%2BKaWwf8ISc%2FNtJEg5e2ouslYm4GYYkDFosaYv1WIPztmWybvGeERGlpJg4apsJMEp2McrL7bT1dRPYRSiK9IZYSLXiW3gUnN3KpV62xDD6x5Y1ZZOw92dri5YdHu%2FyPUtn2JaZGDsgKNsSu2QuxFnDK5kiJJjeykTJPGEmqoP7EynzBnD3uCDGj3pH8GBseU8MR2fkHn%2F8ARqpwx%2F8U34XMTUgFX%2BQSCw6qEuFbDCMngyMAngFR2aF6aAqq7Oms1A61bKaGjHCzsEEyEhamLbpvqbGwjN7vrQhuiU6VsKLSnozfUI9eiGFGdko4xmXxbJkuPJHHiAEEtJoKSMPf8oJUGOqgB%2BS48LRDY83OvX5W2mX%2FTqb9UvUdwE9VB2HBQ11mkE36LD9BtOyLhxaFuMYFOUqk1JBKatWSV49UlQUZuRnEFnNmlB4v2zLX3qQ6WjbkZsGIGUynNugAGbtoxqHW9wWklimJKha1nDv3JmF4vDlfWMLfCe42Kth0gxEpXmMkdjY%2BSdJm%2FeELOzTqFrp3yMm%2FziDttBzx3mZnHgz0qaZIEp0BVmn4kSRH5&Expires=1655198312',
    'https://datasets-626827236627.s3.amazonaws.com/avatars/orly-hamzani-046368aa_06122022.jpg?AWSAccessKeyId=ASIAZD4OQSUJ3IBDFZUN&Signature=pnRMEwKHNA5PfwDgHNoGavUxZvM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEID%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCfwrDPDOtQ7qval0EdjUQEhah2PvnNeJmO3KqjRmSzzQIhALmpYTK%2BIGxCNwfBBqtqCvme8cAhS6S2LCc6ti4rHC6VKtsECIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjI2ODI3MjM2NjI3IgxmYcKEgfQcYn6Ktf0qrwSTcdMXW6jBfrKQn1FtoasmBah2JYCy2X%2Bb07cLFKKNrwH7YLFlCoXiKWP6ntzrm0R2wSsr%2BTmn6RH8WoiXnnaC%2BagFiyUdqPlTdBGy2L%2BO2EKBNA0FnRx%2FnHR0rLG%2Fmcv7cSVcG%2Bsthv5nVSafUgDlQ8dLlcW%2FiGk10eSSAM2tMcUOPyoOLlsSJ%2B0RiSppJEhy3%2F5S63p7RT2fVdlSE1XH%2BnQYvIoUEw0uB3rVywTMknFFi7h8teki%2BGE%2BuONqcHwxSjfnDE53DzZZmmZ%2F5yW2TiK3KbbV974PVgGxA4epRwFCGKqmf7%2FeQCxjmOhPpvXL%2FWiltAIIxtUxlx6IL6IfTsf8i0bVU9cO4Jj14r5%2BdGt96h%2FHGftm%2BdBp1v7JD0vBIxiwFFgv073YjBBmg4gUoryI%2BKaWwf8ISc%2FNtJEg5e2ouslYm4GYYkDFosaYv1WIPztmWybvGeERGlpJg4apsJMEp2McrL7bT1dRPYRSiK9IZYSLXiW3gUnN3KpV62xDD6x5Y1ZZOw92dri5YdHu%2FyPUtn2JaZGDsgKNsSu2QuxFnDK5kiJJjeykTJPGEmqoP7EynzBnD3uCDGj3pH8GBseU8MR2fkHn%2F8ARqpwx%2F8U34XMTUgFX%2BQSCw6qEuFbDCMngyMAngFR2aF6aAqq7Oms1A61bKaGjHCzsEEyEhamLbpvqbGwjN7vrQhuiU6VsKLSnozfUI9eiGFGdko4xmXxbJkuPJHHiAEEtJoKSMPf8oJUGOqgB%2BS48LRDY83OvX5W2mX%2FTqb9UvUdwE9VB2HBQ11mkE36LD9BtOyLhxaFuMYFOUqk1JBKatWSV49UlQUZuRnEFnNmlB4v2zLX3qQ6WjbkZsGIGUynNugAGbtoxqHW9wWklimJKha1nDv3JmF4vDlfWMLfCe42Kth0gxEpXmMkdjY%2BSdJm%2FeELOzTqFrp3yMm%2FziDttBzx3mZnHgz0qaZIEp0BVmn4kSRH5&Expires=1655199524',
    # 'dataset/img1.jpg',
	# 'dataset/img5.jpg',
	# 'dataset/img6.jpg',
	# 'dataset/img8.jpg',
    # 'dataset/img7.jpg',
    # 'dataset/img9.jpg',
    # 'dataset/img11.jpg',
    # 'dataset/img11.jpg',
]

detectors = ['opencv', 'ssd', 'retinaface', 'mtcnn']  # dlib not tested


def test_gender_prediction():
    for detector in detectors:
        results = DeepFace.analyze(dataset, actions=('gender',), detector_backend=detector, prog_bar=False, enforce_detection=False)
        for result in results:
            assert 'gender' in result.keys()
            assert 'dominant_gender' in result.keys() and result["dominant_gender"] in ["Man", "Woman"]
            if result["dominant_gender"] == "Man":
                assert result["gender"]["Man"] > result["gender"]["Woman"]
            else:
                assert result["gender"]["Man"] < result["gender"]["Woman"]
        print(f'detector {detector} passed')


if __name__ == "__main__":
    test_gender_prediction()
