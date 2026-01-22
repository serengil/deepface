# built-in dependencies
import os
import shutil
import uuid
import unittest

# 3rd party dependencies
import pytest
from lightdsa import LightDSA

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

ALGORITHMS = ["ecdsa", "eddsa", "rsa", "dsa"]


# pylint: disable=line-too-long
class TestSignature(unittest.TestCase):
    def setUp(self):
        experiment_id = str(uuid.uuid4())
        self.db_path = f"/tmp/{experiment_id}"
        self.expected_ds = (
            "ds_model_vggface_detector_opencv_aligned_normalization_base_expand_0.pkl"
        )

        # create experiment folder
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)

        # copy some test files
        os.system(f"cp dataset/img1.jpg /tmp/{experiment_id}/")
        os.system(f"cp dataset/img2.jpg /tmp/{experiment_id}/")
        os.system(f"cp dataset/img3.jpg /tmp/{experiment_id}/")

    def tearDown(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def test_sign_and_verify_happy_path_with_obj(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)
            dfs_1st = DeepFace.find(
                img_path="dataset/img6.jpg", db_path=self.db_path, credentials=cs
            )
            dfs_2nd = DeepFace.find(
                img_path="dataset/img7.jpg", db_path=self.db_path, credentials=cs
            )

            assert isinstance(dfs_1st, list)
            assert isinstance(dfs_2nd, list)

            assert len(dfs_1st) > 0
            assert len(dfs_2nd) > 0

            assert dfs_1st[0].shape[0] == 2  # img1, img2
            assert dfs_2nd[0].shape[0] == 2  # img1, img2

            logger.info(
                f"✅ Signature test for happy path with LightDSA obj passed for {algorithm_name}"
            )
            self.__flush_datastore_and_signature()

    def test_sign_and_verify_happy_path_with_dict(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)
            _ = DeepFace.find(
                img_path="dataset/img6.jpg",
                db_path=self.db_path,
                credentials={**cs.dsa.keys, "algorithm_name": algorithm_name},
            )
            _ = DeepFace.find(
                img_path="dataset/img7.jpg",
                db_path=self.db_path,
                credentials={**cs.dsa.keys, "algorithm_name": algorithm_name},
            )

            logger.info(f"✅ Signature test for happy path with dict passed for {algorithm_name}")
            self.__flush_datastore_and_signature()

    def test_missing_algorithm_in_dict(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)

            with pytest.raises(
                ValueError, match="credentials dictionary must have 'algorithm_name' key"
            ):
                _ = DeepFace.find(
                    img_path="dataset/img6.jpg",
                    db_path=self.db_path,
                    credentials=cs.dsa.keys,
                )

            logger.info(f"✅ Signature test for missing algorithm name passed for {algorithm_name}")
            self.__flush_datastore_and_signature()

    def test_tampered_datastore_detection_with_type_error(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)

            # this will create and sign the datastore
            _ = DeepFace.find(img_path="dataset/img6.jpg", db_path=self.db_path, credentials=cs)

            # Tamper with the datastore file
            signature = f"{self.db_path}/{self.expected_ds}.ldsa"
            with open(signature, "w", encoding="utf-8") as f:
                f.write("'tampering with the datastore'")

            # signature type is not matching the algorithm
            with pytest.raises(ValueError, match="Verify the signature"):
                _ = DeepFace.find(img_path="dataset/img7.jpg", db_path=self.db_path, credentials=cs)

            self.__flush_datastore_and_signature()

            logger.info(
                f"✅ Tampered datastore detection test with type error passed for {algorithm_name}"
            )

    def test_tampered_datastore_detection_with_content(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)

            # this will create and sign the datastore
            _ = DeepFace.find(img_path="dataset/img6.jpg", db_path=self.db_path, credentials=cs)

            if algorithm_name == "rsa":
                new_signature = 319561459047296488548458984747399773018716548204273025089624526759359534233284312158510290754391934668693104185451914855617607653930118615834122905987992506015413484111459136235040703307837127330552076394553025514846602694994704058032032682011030896228476574896316764474080643444528752822215665326313975210266100821320428968057729348770126684036043834110914715739798738033680251895412183116783758569626527555756175521592665908984550792405972689418461489583818241720836275237261051794829129609867815663459783380179330918682830834361767346820728010180691232612809687266284664884281497246914633251532570093804727503221592140826938085233362518642240314192925852839183375057159735842181129571046919197169304114361287251975127762914060608489444548355191778055788828924190814939438198679453052886489889714657423399402932343101284126001466450432228046323891788753347814011641443220020734532039664233082527737624947853241639198217834
            elif algorithm_name == "dsa":
                new_signature = (
                    9100224601877825638014134863066256026676002678448729267282144204754,
                    7634803645147310159689393871148731810441463887021966655811033398889,
                )
            elif algorithm_name == "ecdsa":
                new_signature = (
                    21518513378698262440337632884706393342279436822165585485363750050247340191720,
                    41671206923584596559299832926426077520762049023469772372748973101822889226099,
                )
            elif algorithm_name == "eddsa":
                new_signature = (
                    (
                        10558823458062709006637242334064065742608369933510637863306601023456142118149,
                        35629342124183609337373813938440693584938877184586698416611230077459766071,
                    ),
                    616779964632213973552139846661614990419749880592583320329014277948539725318347638187634255022715560768868643524944455800527501125094189022366431963778314,
                )
            else:
                raise ValueError(f"Unsupported algorithm name: {algorithm_name}")

            # Tamper with the datastore file
            signature = f"{self.db_path}/{self.expected_ds}.ldsa"
            with open(signature, "w", encoding="utf-8") as f:
                f.write(str(new_signature))

            # signature type is not matching the algorithm
            with pytest.raises(ValueError, match=r"(Signature is invalid|Invalid signature)"):
                _ = DeepFace.find(img_path="dataset/img7.jpg", db_path=self.db_path, credentials=cs)

            self.__flush_datastore_and_signature()

            logger.info(
                f"✅ Tampered datastore detection test with content passed for {algorithm_name}"
            )

    def test_unsigned_datastore_detected(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)

            # this will create ds without signature
            _ = DeepFace.find(img_path="dataset/img6.jpg", db_path=self.db_path)

            with pytest.raises(
                ValueError,
                match=".ldsa not found.You may need to re-create the pickle by deleting the existing one.",
            ):
                _ = DeepFace.find(img_path="dataset/img7.jpg", db_path=self.db_path, credentials=cs)

            logger.info(
                f"✅ Signature test for happy path with LightDSA obj passed for {algorithm_name}"
            )
            self.__flush_datastore_and_signature()

    def test_signed_datastore_with_no_credentials(self):
        for algorithm_name in ALGORITHMS:
            cs = LightDSA(algorithm_name=algorithm_name)

            # this will create and sign the datastore
            _ = DeepFace.find(img_path="dataset/img6.jpg", db_path=self.db_path, credentials=cs)

            signature_path = f"{self.db_path}/{self.expected_ds}.ldsa"
            with pytest.raises(
                ValueError,
                match=f"Credentials not provided but signature file {signature_path} exists.",
            ):
                _ = DeepFace.find(img_path="dataset/img7.jpg", db_path=self.db_path)

            logger.info(f"✅ Signed datastore with no credentials test passed for {algorithm_name}")
            self.__flush_datastore_and_signature()

    def test_custom_curves(self):
        for algorithm_name, form_name, curve_name in [
            # default configurations
            # ("eddsa", "edwards", "ed25519"),
            # ("ecdsa", "weierstrass", "secp256k1"),
            # custom configurations
            ("eddsa", "weierstrass", "secp256k1"),
            ("eddsa", "koblitz", "k233"),
            ("eddsa", "edwards", "e521"),
            ("ecdsa", "edwards", "ed25519"),
            ("ecdsa", "koblitz", "k233"),
            ("ecdsa", "weierstrass", "bn638"),
        ]:
            cs = LightDSA(algorithm_name=algorithm_name, form_name=form_name, curve_name=curve_name)
            _ = DeepFace.find(
                img_path="dataset/img6.jpg",
                db_path=self.db_path,
                credentials={
                    **cs.dsa.keys,
                    "algorithm_name": algorithm_name,
                    "form_name": form_name,
                    "curve_name": curve_name,
                },
            )
            _ = DeepFace.find(
                img_path="dataset/img7.jpg",
                db_path=self.db_path,
                credentials={
                    **cs.dsa.keys,
                    "algorithm_name": algorithm_name,
                    "form_name": form_name,
                    "curve_name": curve_name,
                },
            )

            logger.info(
                f"✅ Signature test for custom curves passed for {algorithm_name}/{form_name}/{curve_name}"
            )
            self.__flush_datastore_and_signature()

    def __flush_datastore_and_signature(self):
        if os.path.exists(f"{self.db_path}/{self.expected_ds}"):
            os.remove(f"{self.db_path}/{self.expected_ds}")

        if os.path.exists(f"{self.db_path}/{self.expected_ds}.ldsa"):
            os.remove(f"{self.db_path}/{self.expected_ds}.ldsa")
