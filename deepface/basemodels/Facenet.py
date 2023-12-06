import os
import gdown
import tensorflow as tf
from deepface.commons import functions
from deepface.commons.logger import Logger

logger = Logger()

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


def scaling(x, scale):
    return x * scale


def InceptionResNetV2(dimension=128):

    inputs = Input(shape=(160, 160, 3))
    x = Conv2D(32, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_1a_3x3")(inputs)
    x = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_1a_3x3_BatchNorm"
    )(x)
    x = Activation("relu", name="Conv2d_1a_3x3_Activation")(x)
    x = Conv2D(32, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_2a_3x3")(x)
    x = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_2a_3x3_BatchNorm"
    )(x)
    x = Activation("relu", name="Conv2d_2a_3x3_Activation")(x)
    x = Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="Conv2d_2b_3x3")(x)
    x = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_2b_3x3_BatchNorm"
    )(x)
    x = Activation("relu", name="Conv2d_2b_3x3_Activation")(x)
    x = MaxPooling2D(3, strides=2, name="MaxPool_3a_3x3")(x)
    x = Conv2D(80, 1, strides=1, padding="valid", use_bias=False, name="Conv2d_3b_1x1")(x)
    x = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_3b_1x1_BatchNorm"
    )(x)
    x = Activation("relu", name="Conv2d_3b_1x1_Activation")(x)
    x = Conv2D(192, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_4a_3x3")(x)
    x = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_4a_3x3_BatchNorm"
    )(x)
    x = Activation("relu", name="Conv2d_4a_3x3_Activation")(x)
    x = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_4b_3x3")(x)
    x = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_4b_3x3_BatchNorm"
    )(x)
    x = Activation("relu", name="Conv2d_4b_3x3_Activation")(x)

    # 5x Block35 (Inception-ResNet-A block):
    branch_0 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_1_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block35_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_1_Conv2d_0b_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_1_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
    branch_2 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_2_Conv2d_0b_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_1_Branch_2_Conv2d_0c_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3, name="Block35_1_Concatenate")(branches)
    up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_1_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
    x = add([x, up])
    x = Activation("relu", name="Block35_1_Activation")(x)

    branch_0 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_2_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block35_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_2_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_1_Conv2d_0b_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_2_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
    branch_2 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_2_Conv2d_0b_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_2_Branch_2_Conv2d_0c_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3, name="Block35_2_Concatenate")(branches)
    up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_2_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
    x = add([x, up])
    x = Activation("relu", name="Block35_2_Activation")(x)

    branch_0 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_3_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block35_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_3_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_1_Conv2d_0b_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_3_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
    branch_2 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_2_Conv2d_0b_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_3_Branch_2_Conv2d_0c_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3, name="Block35_3_Concatenate")(branches)
    up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_3_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
    x = add([x, up])
    x = Activation("relu", name="Block35_3_Activation")(x)

    branch_0 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_4_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block35_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_4_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_1_Conv2d_0b_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_4_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
    branch_2 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_2_Conv2d_0b_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_4_Branch_2_Conv2d_0c_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3, name="Block35_4_Concatenate")(branches)
    up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_4_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
    x = add([x, up])
    x = Activation("relu", name="Block35_4_Activation")(x)

    branch_0 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_5_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block35_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_5_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_1_Conv2d_0b_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block35_5_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
    branch_2 = Conv2D(
        32, 1, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_2_Conv2d_0b_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
    branch_2 = Conv2D(
        32, 3, strides=1, padding="same", use_bias=False, name="Block35_5_Branch_2_Conv2d_0c_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3, name="Block35_5_Concatenate")(branches)
    up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name="Block35_5_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
    x = add([x, up])
    x = Activation("relu", name="Block35_5_Activation")(x)

    # Mixed 6a (Reduction-A block):
    branch_0 = Conv2D(
        384, 3, strides=2, padding="valid", use_bias=False, name="Mixed_6a_Branch_0_Conv2d_1a_3x3"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192, 3, strides=1, padding="same", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_0b_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
    branch_1 = Conv2D(
        256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_1a_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation")(branch_1)
    branch_pool = MaxPooling2D(
        3, strides=2, padding="valid", name="Mixed_6a_Branch_2_MaxPool_1a_3x3"
    )(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3, name="Mixed_6a")(branches)

    # 10x Block17 (Inception-ResNet-B block):
    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_1_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_1_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_1_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_1_Branch_1_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_1_Branch_1_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_1_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_1_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_1_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_2_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_2_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_2_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_2_Branch_2_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_2_Branch_2_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_2_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_2_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_2_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_3_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_3_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_3_Branch_3_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_3_Branch_3_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_3_Branch_3_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_3_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_3_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_3_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_4_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_4_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_4_Branch_4_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_4_Branch_4_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_4_Branch_4_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_4_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_4_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_4_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_5_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_5_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_5_Branch_5_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_5_Branch_5_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_5_Branch_5_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_5_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_5_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_5_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_6_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_6_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_6_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_6_Branch_6_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_6_Branch_6_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_6_Branch_6_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_6_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_6_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_6_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_7_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_7_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_7_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_7_Branch_7_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_7_Branch_7_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_7_Branch_7_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_7_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_7_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_7_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_8_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_8_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_8_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_8_Branch_8_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_8_Branch_8_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_8_Branch_8_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_8_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_8_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_8_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_9_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_9_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_9_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_9_Branch_9_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_9_Branch_9_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_9_Branch_9_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_9_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_9_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_9_Activation")(x)

    branch_0 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_10_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_10_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block17_10_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        128, 1, strides=1, padding="same", use_bias=False, name="Block17_10_Branch_10_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [1, 7],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_10_Branch_10_Conv2d_0b_1x7",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0b_1x7_Activation")(branch_1)
    branch_1 = Conv2D(
        128,
        [7, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block17_10_Branch_10_Conv2d_0c_7x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0c_7x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block17_10_Concatenate")(branches)
    up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name="Block17_10_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
    x = add([x, up])
    x = Activation("relu", name="Block17_10_Activation")(x)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = Conv2D(
        256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_0_Conv2d_0a_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation")(branch_0)
    branch_0 = Conv2D(
        384, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_0_Conv2d_1a_3x3"
    )(branch_0)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation")(branch_0)
    branch_1 = Conv2D(
        256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_1_Conv2d_1a_3x3"
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation")(branch_1)
    branch_2 = Conv2D(
        256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
    branch_2 = Conv2D(
        256, 3, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_0b_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
    branch_2 = Conv2D(
        256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_1a_3x3"
    )(branch_2)
    branch_2 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm",
    )(branch_2)
    branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation")(branch_2)
    branch_pool = MaxPooling2D(
        3, strides=2, padding="valid", name="Mixed_7a_Branch_3_MaxPool_1a_3x3"
    )(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3, name="Mixed_7a")(branches)

    # 5x Block8 (Inception-ResNet-C block):

    branch_0 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_1_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_1_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block8_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_1_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [1, 3],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_1_Branch_1_Conv2d_0b_1x3",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0b_1x3_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [3, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_1_Branch_1_Conv2d_0c_3x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0c_3x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block8_1_Concatenate")(branches)
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_1_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
    x = add([x, up])
    x = Activation("relu", name="Block8_1_Activation")(x)

    branch_0 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_2_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_2_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block8_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_2_Branch_2_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [1, 3],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_2_Branch_2_Conv2d_0b_1x3",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0b_1x3_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [3, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_2_Branch_2_Conv2d_0c_3x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0c_3x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block8_2_Concatenate")(branches)
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_2_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
    x = add([x, up])
    x = Activation("relu", name="Block8_2_Activation")(x)

    branch_0 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_3_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_3_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block8_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_3_Branch_3_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [1, 3],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_3_Branch_3_Conv2d_0b_1x3",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0b_1x3_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [3, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_3_Branch_3_Conv2d_0c_3x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0c_3x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block8_3_Concatenate")(branches)
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_3_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
    x = add([x, up])
    x = Activation("relu", name="Block8_3_Activation")(x)

    branch_0 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_4_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_4_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block8_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_4_Branch_4_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [1, 3],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_4_Branch_4_Conv2d_0b_1x3",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0b_1x3_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [3, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_4_Branch_4_Conv2d_0c_3x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0c_3x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block8_4_Concatenate")(branches)
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_4_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
    x = add([x, up])
    x = Activation("relu", name="Block8_4_Activation")(x)

    branch_0 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_5_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_5_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block8_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_5_Branch_5_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [1, 3],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_5_Branch_5_Conv2d_0b_1x3",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0b_1x3_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [3, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_5_Branch_5_Conv2d_0c_3x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0c_3x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block8_5_Concatenate")(branches)
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_5_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
    x = add([x, up])
    x = Activation("relu", name="Block8_5_Activation")(x)

    branch_0 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_6_Branch_0_Conv2d_1x1"
    )(x)
    branch_0 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_6_Branch_0_Conv2d_1x1_BatchNorm",
    )(branch_0)
    branch_0 = Activation("relu", name="Block8_6_Branch_0_Conv2d_1x1_Activation")(branch_0)
    branch_1 = Conv2D(
        192, 1, strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0a_1x1"
    )(x)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [1, 3],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_6_Branch_1_Conv2d_0b_1x3",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0b_1x3_Activation")(branch_1)
    branch_1 = Conv2D(
        192,
        [3, 1],
        strides=1,
        padding="same",
        use_bias=False,
        name="Block8_6_Branch_1_Conv2d_0c_3x1",
    )(branch_1)
    branch_1 = BatchNormalization(
        axis=3,
        momentum=0.995,
        epsilon=0.001,
        scale=False,
        name="Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm",
    )(branch_1)
    branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0c_3x1_Activation")(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3, name="Block8_6_Concatenate")(branches)
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_6_Conv2d_1x1")(
        mixed
    )
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 1})(up)
    x = add([x, up])

    # Classification block
    x = GlobalAveragePooling2D(name="AvgPool")(x)
    x = Dropout(1.0 - 0.8, name="Dropout")(x)
    # Bottleneck
    x = Dense(dimension, use_bias=False, name="Bottleneck")(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name="Bottleneck_BatchNorm")(
        x
    )

    # Create model
    model = Model(inputs, x, name="inception_resnet_v1")

    return model


# url = 'https://drive.google.com/uc?id=1971Xk5RwedbudGgTIrGAL4F7Aifu7id1'


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
):
    model = InceptionResNetV2()

    # -----------------------------------

    home = functions.get_deepface_home()

    if os.path.isfile(home + "/.deepface/weights/facenet_weights.h5") != True:
        logger.info("facenet_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/facenet_weights.h5"
        gdown.download(url, output, quiet=False)

    # -----------------------------------

    model.load_weights(home + "/.deepface/weights/facenet_weights.h5")

    # -----------------------------------

    return model
