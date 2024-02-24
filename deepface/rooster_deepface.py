import os
from os import path
import pandas as pd
from tqdm import tqdm
import pickle
from deepface.DeepFace import functions, represent, dst
import time
import numpy as np


# These are thresholds that we have computed. Add to the list as more are
THRESHOLDS = {
    "VGG-Face": {
        "cosine": 0.34,
    },
    "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
}


def create_encodings_database(
    db_path,
    model_name="VGG-Face",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=True,
    force_recreate=False,
):
    """
    Create the necessary pkl files for a folder of images
    If a file is already there, it will use that unless force_recreate=True, then it will delete it and recreate it

    returns:
        reperesentations
    """
    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = functions.find_target_size(model_name=model_name)

    # ---------------------------------------

    file_name = f"representations_{model_name}_{detector_backend}.pkl"
    file_name = file_name.replace("-", "_").lower()

    df_cols = [
        "identity",
        f"{model_name}_representation",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    if path.exists(db_path + "/" + file_name) and not force_recreate:
        if not silent:
            print(
                f"Representations for images in {db_path} folder were previously stored"
                f" in {file_name}. If you added new instances after the creation, then please "
                "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

            if len(representations) > 0 and len(representations[0]) != len(df_cols):
                raise ValueError(
                    f"Seems existing {db_path}/{file_name} is out-of-the-date."
                    "Delete it and re-run."
                )

        if not silent:
            print(f"There are {len(representations)} representations found in {file_name}")

    else:  # create representation.pkl from scratch
        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )

        # ------------------------
        # find representations for db images

        representations = []

        # for employee in employees:
        pbar = tqdm(
            range(0, len(employees)),
            desc="Finding representations",
            disable=silent,
        )
        for index in pbar:
            employee = employees[index]

            img_objs = functions.extract_faces(
                img=employee,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )

            for img_content, img_region, _ in img_objs:
                embedding_obj = represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                instance.append(img_region["x"])
                instance.append(img_region["y"])
                instance.append(img_region["w"])
                instance.append(img_region["h"])
                representations.append(instance)

        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            print(
                f"Representations stored in {db_path}/{file_name} file."
                + "Please delete this file when you add new identities in your database."
            )

    return representations


def match_face(
    facial_data,
    db_path,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=True,
):
    """
    This is Rooster's adaptation of DeepFace.find

    Parameters:
            facial_data: from extract faces

            db_path (string): You should store some image files in a folder and pass the
            exact folder path to this. A database image can also have many faces.
            Then, all detected faces in db side will be considered in the decision.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
            Dlib, ArcFace, SFace or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (bool): The function throws exception if a face could not be detected.
            Set this to True if you don't want to get exception. This might be convenient for low
            resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

            silent (boolean): disable some logging and progress bars

    Returns:
            This function returns list of pandas data frame. Each item of the list corresponding to
            an identity in the img_path.
    """

    tic = time.time()

    representations = create_encodings_database(
        db_path=db_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        normalization=normalization,
        silent=silent,
        force_recreate=False,
    )

    # ----------------------------
    # now, we got representations for facial database
    df = pd.DataFrame(
        representations,
        columns=[
            "identity",
            f"{model_name}_representation",
            "target_x",
            "target_y",
            "target_w",
            "target_h",
        ],
    )
    # print(f"Data Frame Creating took {time.time()-tic}s")
    resp_obj = []
    target_img = facial_data["face"]
    target_region = facial_data["facial_area"]
    # confidence = facial_data["confidence"]
    # rep_time = time.time()

    # If the face is already embedded, skip that step to increase speed
    if "embedding" in facial_data.keys():
        print("match_face embedding mode on")
        target_representation = facial_data["embedding"]
    else:

        target_embedding_obj = represent(
            img_path=target_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )
        # print(f"Time to represent the face took {time.time()-rep_time}s")

        target_representation = target_embedding_obj[0]["embedding"]

    result_df = df.copy()  # df will be filtered in each img
    result_df["source_x"] = target_region["x"]
    result_df["source_y"] = target_region["y"]
    result_df["source_w"] = target_region["w"]
    result_df["source_h"] = target_region["h"]

    distances = []
    # dist_time = time.time()
    for index, instance in df.iterrows():
        source_representation = instance[f"{model_name}_representation"]

        if distance_metric == "cosine":
            distance = dst.findCosineDistance(source_representation, target_representation)
        elif distance_metric == "euclidean":
            distance = dst.findEuclideanDistance(source_representation, target_representation)
        elif distance_metric == "euclidean_l2":
            distance = dst.findEuclideanDistance(
                dst.l2_normalize(source_representation),
                dst.l2_normalize(target_representation),
            )
        else:
            raise ValueError(f"invalid distance metric passes - {distance_metric}")

        distances.append(distance)

        # ---------------------------
    # print(f"Time to calculate all of the distances {time.time()-dist_time}s")
    result_df[f"{model_name}_{distance_metric}"] = distances

    # df_time = time.time()
    threshold = THRESHOLDS[model_name][distance_metric]
    result_df = result_df.drop(columns=[f"{model_name}_representation"])
    result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
    result_df = result_df.sort_values(
        by=[f"{model_name}_{distance_metric}"], ascending=True
    ).reset_index(drop=True)

    resp_obj.append(result_df)

    # print(f"time to sort dataframe: {time.time()-df_time}s")

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return resp_obj


def get_embedding(
    img, model_name="ArcFace", enforce_detection=True, align=True, normalization="base"
):
    """
    Designed for rooster to return the embedding of the face, so it only has to be computed once
    """
    img1_embedding_obj = represent(
        img_path=img,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend="skip",
        align=align,
        normalization=normalization,
    )

    return img1_embedding_obj[0]["embedding"]


def verify(
    img1,
    img2,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    align=True,
    normalization="base",
    embedded_mode=False,
):
    """
    This is Rooster's adaptation of DeepFace.
    This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            img1_path, img2_path: exact image path as string. numpy array (BGR) or based64 encoded
            images are also welcome. If one of pair has more than one face, then we will compare the
            face pair with max similarity.

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
            , ArcFace and SFace

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

            embedded_mode (boolean): if True, assumes that img1 and img2 are embeddings,
            not images so it skips the embedding function

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    , "distance": 0.2563
                    , "max_threshold_to_verify": 0.40
                    , "model": "VGG-Face"
                    , "similarity_metric": "cosine"
                    , 'facial_areas': {
                            'img1': {'x': 345, 'y': 211, 'w': 769, 'h': 769},
                            'img2': {'x': 318, 'y': 534, 'w': 779, 'h': 779}
                    }
                    , "time": 2
            }

    """

    tic = time.time()
    # --------------------------------
    distances = []

    if not embedded_mode:
        # now we will find the face pair with minimum distance
        img1_embedding_obj = represent(
            img_path=img1,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        img2_embedding_obj = represent(
            img_path=img2,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        img1_representation = img1_embedding_obj[0]["embedding"]
        img2_representation = img2_embedding_obj[0]["embedding"]
    else:
        img1_representation = img1
        img2_representation = img2

    if distance_metric == "cosine":
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == "euclidean":
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == "euclidean_l2":
        distance = dst.findEuclideanDistance(
            dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    distances.append(distance)

    # -------------------------------
    threshold = dst.findThreshold(model_name, distance_metric)
    distance = min(distances)  # best distance

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "time": round(toc - tic, 2),
    }

    return resp_obj
