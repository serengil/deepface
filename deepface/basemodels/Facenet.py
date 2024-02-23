import os
import gdown
import tensorflow as tf
from deepface.commons import functions

# --------------------------------
# dependency configuration

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model
    from keras.layers import Activation
    from keras.layers import BatchNormalization
    from keras.layers import Concatenate
    from keras.layers import Conv2D
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import GlobalAveragePooling2D
    from keras.layers import Input
    from keras.layers import Lambda
    from keras.layers import MaxPooling2D
    from keras.layers import add
    from keras import backend as K
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import add
    from tensorflow.keras import backend as K

# --------------------------------


class InceptionResNetV2:
    """
    InceptionResNetV2 model
    Args:
        dimension (int): number of dimensions in the embedding layer
    Returns:
        model (Model)
    """

    def __new__(cls, dimension: int = 128):
        instance = super(InceptionResNetV2, cls).__new__(cls)
        model = instance._build(dimension)
        return model

    def _build(self, dimension):
        inputs = Input(shape=(160, 160, 3))
        x = self._input_block(inputs)
        for i in range(5):
            x = self._inception_block_a(x, i + 1)
        x = self._reduction_block_a(x)
        for i in range(10):
            x = self._inception_block_b(x, i + 1)
        x = self._reduction_block_b(x)
        for i in range(6):
            x = self._inception_block_c(x, i + 1)
        x = self._output_block(x, dimension)
        model = Model(inputs, x, name="inception_resnet_v2")
        return model

    def _scaling(self, x, scale):
        return x * scale

    def _input_block(self, inputs):
        x = Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Conv2d_1a_3x3",
        )(inputs)
        x = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Conv2d_1a_3x3_BatchNorm",
        )(x)
        x = Activation(
            "relu",
            name="Conv2d_1a_3x3_Activation",
        )(x)
        x = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="valid",
            use_bias=False,
            name="Conv2d_2a_3x3",
        )(x)
        x = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Conv2d_2a_3x3_BatchNorm",
        )(x)
        x = Activation(
            "relu",
            name="Conv2d_2a_3x3_Activation",
        )(x)
        x = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="Conv2d_2b_3x3",
        )(x)
        x = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Conv2d_2b_3x3_BatchNorm",
        )(x)
        x = Activation(
            "relu",
            name="Conv2d_2b_3x3_Activation",
        )(x)
        x = MaxPooling2D(
            pool_size=3,
            strides=2,
            name="MaxPool_3a_3x3",
        )(x)
        x = Conv2D(
            filters=80,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
            name="Conv2d_3b_1x1",
        )(x)
        x = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Conv2d_3b_1x1_BatchNorm",
        )(x)
        x = Activation(
            "relu",
            name="Conv2d_3b_1x1_Activation",
        )(x)
        x = Conv2D(
            filters=192,
            kernel_size=3,
            strides=1,
            padding="valid",
            use_bias=False,
            name="Conv2d_4a_3x3",
        )(x)
        x = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Conv2d_4a_3x3_BatchNorm",
        )(x)
        x = Activation(
            "relu",
            name="Conv2d_4a_3x3_Activation",
        )(x)
        x = Conv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Conv2d_4b_3x3",
        )(x)
        x = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Conv2d_4b_3x3_BatchNorm",
        )(x)
        x = Activation(
            "relu",
            name="Conv2d_4b_3x3_Activation",
        )(x)
        return x

    def _inception_block_a(self, x, i):
        branch_0 = Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block35_{i}_Branch_0_Conv2d_1x1",
        )(x)
        branch_0 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block35_{i}_Branch_0_Conv2d_1x1_BatchNorm",
        )(branch_0)
        branch_0 = Activation(
            "relu",
            name=f"Block35_{i}_Branch_0_Conv2d_1x1_Activation",
        )(branch_0)
        branch_1 = Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block35_{i}_Branch_1_Conv2d_0a_1x1",
        )(x)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block35_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block35_{i}_Branch_1_Conv2d_0a_1x1_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block35_{i}_Branch_1_Conv2d_0b_3x3",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block35_{i}_Branch_1_Conv2d_0b_3x3_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block35_{i}_Branch_1_Conv2d_0b_3x3_Activation",
        )(branch_1)
        branch_2 = Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block35_{i}_Branch_2_Conv2d_0a_1x1",
        )(x)
        branch_2 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block35_{i}_Branch_2_Conv2d_0a_1x1_BatchNorm",
        )(branch_2)
        branch_2 = Activation(
            "relu", name=f"Block35_{i}_Branch_2_Conv2d_0a_1x1_Activation"
        )(branch_2)
        branch_2 = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block35_{i}_Branch_2_Conv2d_0b_3x3",
        )(branch_2)
        branch_2 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block35_{i}_Branch_2_Conv2d_0b_3x3_BatchNorm",
        )(branch_2)
        branch_2 = Activation(
            "relu",
            name=f"Block35_{i}_Branch_2_Conv2d_0b_3x3_Activation",
        )(branch_2)
        branch_2 = Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block35_{i}_Branch_2_Conv2d_0c_3x3",
        )(branch_2)
        branch_2 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block35_{i}_Branch_2_Conv2d_0c_3x3_BatchNorm",
        )(branch_2)
        branch_2 = Activation(
            "relu",
            name=f"Block35_{i}_Branch_2_Conv2d_0c_3x3_Activation",
        )(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(
            axis=3,
            name=f"Block35_{i}_Concatenate",
        )(branches)
        up = Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name=f"Block35_{i}_Conv2d_1x1",
        )(mixed)
        up = Lambda(
            self._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17}
        )(up)
        x = add([x, up])
        x = Activation(
            "relu",
            name=f"Block35_{i}_Activation",
        )(x)
        return x

    def _reduction_block_a(self, x):
        branch_0 = Conv2D(
            filters=384,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Mixed_6a_Branch_0_Conv2d_1a_3x3",
        )(x)
        branch_0 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm",
        )(branch_0)
        branch_0 = Activation(
            "relu",
            name="Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation",
        )(branch_0)
        branch_1 = Conv2D(
            filters=192,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="Mixed_6a_Branch_1_Conv2d_0a_1x1",
        )(x)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name="Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=192,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="Mixed_6a_Branch_1_Conv2d_0b_3x3",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name="Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Mixed_6a_Branch_1_Conv2d_1a_3x3",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name="Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation",
        )(branch_1)
        branch_pool = MaxPooling2D(
            pool_size=3,
            strides=2,
            padding="valid",
            name="Mixed_6a_Branch_2_MaxPool_1a_3x3",
        )(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(
            axis=3,
            name="Mixed_6a",
        )(branches)
        return x

    def _inception_block_b(self, x, i):
        branch_0 = Conv2D(
            filters=128,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block17_{i}_Branch_0_Conv2d_1x1",
        )(x)
        branch_0 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block17_{i}_Branch_0_Conv2d_1x1_BatchNorm",
        )(branch_0)
        branch_0 = Activation(
            "relu",
            name=f"Block17_{i}_Branch_0_Conv2d_1x1_Activation",
        )(branch_0)
        branch_1 = Conv2D(
            filters=128,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block17_{i}_Branch_1_Conv2d_0a_1x1",
        )(x)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block17_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block17_{i}_Branch_1_Conv2d_0a_1x1_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=128,
            kernel_size=[1, 7],
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block17_{i}_Branch_1_Conv2d_0b_1x7",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block17_{i}_Branch_1_Conv2d_0b_1x7_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block17_{i}_Branch_1_Conv2d_0b_1x7_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=128,
            kernel_size=[7, 1],
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block17_{i}_Branch_1_Conv2d_0c_7x1",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block17_{i}_Branch_1_Conv2d_0c_7x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block17_{i}_Branch_1_Conv2d_0c_7x1_Activation",
        )(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(
            axis=3,
            name=f"Block17_{i}_Concatenate",
        )(branches)
        up = Conv2D(
            filters=896,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name=f"Block17_{i}_Conv2d_1x1",
        )(mixed)
        up = Lambda(
            self._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1}
        )(up)
        x = add([x, up])
        x = Activation(
            "relu",
            name=f"Block17_{i}_Activation",
        )(x)
        return x

    def _reduction_block_b(self, x):
        branch_0 = Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="Mixed_7a_Branch_0_Conv2d_0a_1x1",
        )(x)
        branch_0 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm",
        )(branch_0)
        branch_0 = Activation(
            "relu",
            name="Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation",
        )(branch_0)
        branch_0 = Conv2D(
            filters=384,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Mixed_7a_Branch_0_Conv2d_1a_3x3",
        )(branch_0)
        branch_0 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm",
        )(branch_0)
        branch_0 = Activation(
            "relu",
            name="Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation",
        )(branch_0)
        branch_1 = Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="Mixed_7a_Branch_1_Conv2d_0a_1x1",
        )(x)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name="Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Mixed_7a_Branch_1_Conv2d_1a_3x3",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name="Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation",
        )(branch_1)
        branch_2 = Conv2D(
            filters=256,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name="Mixed_7a_Branch_2_Conv2d_0a_1x1",
        )(x)
        branch_2 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm",
        )(branch_2)
        branch_2 = Activation(
            "relu",
            name="Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation",
        )(branch_2)
        branch_2 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="Mixed_7a_Branch_2_Conv2d_0b_3x3",
        )(branch_2)
        branch_2 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm",
        )(branch_2)
        branch_2 = Activation(
            "relu",
            name="Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation",
        )(branch_2)
        branch_2 = Conv2D(
            filters=256,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="Mixed_7a_Branch_2_Conv2d_1a_3x3",
        )(branch_2)
        branch_2 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm",
        )(branch_2)
        branch_2 = Activation(
            "relu",
            name="Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation",
        )(branch_2)
        branch_pool = MaxPooling2D(
            pool_size=3,
            strides=2,
            padding="valid",
            name="Mixed_7a_Branch_3_MaxPool_1a_3x3",
        )(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(
            axis=3,
            name="Mixed_7a",
        )(branches)
        return x

    def _inception_block_c(self, x, i):
        branch_0 = Conv2D(
            filters=192,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block8_{i}_Branch_0_Conv2d_1x1",
        )(x)
        branch_0 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block8_{i}_Branch_0_Conv2d_1x1_BatchNorm",
        )(branch_0)
        branch_0 = Activation(
            "relu",
            name=f"Block8_{i}_Branch_0_Conv2d_1x1_Activation",
        )(branch_0)
        branch_1 = Conv2D(
            filters=192,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block8_{i}_Branch_1_Conv2d_0a_1x1",
        )(x)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block8_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block8_{i}_Branch_1_Conv2d_0a_1x1_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=192,
            kernel_size=[1, 3],
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block8_{i}_Branch_1_Conv2d_0b_1x3",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block8_{i}_Branch_1_Conv2d_0b_1x3_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block8_{i}_Branch_1_Conv2d_0b_1x3_Activation",
        )(branch_1)
        branch_1 = Conv2D(
            filters=192,
            kernel_size=[3, 1],
            strides=1,
            padding="same",
            use_bias=False,
            name=f"Block8_{i}_Branch_1_Conv2d_0c_3x1",
        )(branch_1)
        branch_1 = BatchNormalization(
            axis=3,
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name=f"Block8_{i}_Branch_1_Conv2d_0c_3x1_BatchNorm",
        )(branch_1)
        branch_1 = Activation(
            "relu",
            name=f"Block8_{i}_Branch_1_Conv2d_0c_3x1_Activation",
        )(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(
            axis=3,
            name=f"Block8_{i}_Concatenate",
        )(branches)
        up = Conv2D(
            filters=1792,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name=f"Block8_{i}_Conv2d_1x1",
        )(mixed)
        up = Lambda(
            self._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2}
        )(up)
        x = add([x, up])
        x = Activation(
            "relu",
            name=f"Block8_{i}_Activation",
        )(x)
        return x

    def _output_block(self, x, dimension):
        x = GlobalAveragePooling2D(
            name="AvgPool",
        )(x)
        x = Dropout(
            1.0 - 0.8,
            name="Dropout",
        )(x)
        x = Dense(
            dimension,
            use_bias=False,
            name="Bottleneck",
        )(x)
        x = BatchNormalization(
            momentum=0.995,
            epsilon=0.001,
            scale=False,
            name="Bottleneck_BatchNorm",
        )(x)
        return x


# url = 'https://drive.google.com/uc?id=1971Xk5RwedbudGgTIrGAL4F7Aifu7id1'


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
):
    model = InceptionResNetV2()

    # -----------------------------------

    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/facenet_weights.h5") != True:
        print("facenet_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/facenet_weights.h5"
        gdown.download(url, output, quiet=False)

    # -----------------------------------

    model.load_weights(home + "/.deepface/weights/facenet_weights.h5")

    # -----------------------------------

    return model
