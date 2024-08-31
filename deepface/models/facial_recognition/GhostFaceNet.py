# 3rd party dependencies
import tensorflow as tf

# project dependencies
from deepface.commons import package_utils, weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

tf_major = package_utils.get_tf_major_version()
if tf_major == 1:
    import keras
    from keras import backend as K
    from keras.models import Model
    from keras.layers import (
        Activation,
        Add,
        BatchNormalization,
        Concatenate,
        Conv2D,
        DepthwiseConv2D,
        GlobalAveragePooling2D,
        Input,
        Reshape,
        Multiply,
        ReLU,
        PReLU,
    )
else:
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Activation,
        Add,
        BatchNormalization,
        Concatenate,
        Conv2D,
        DepthwiseConv2D,
        GlobalAveragePooling2D,
        Input,
        Reshape,
        Multiply,
        ReLU,
        PReLU,
    )


# pylint: disable=line-too-long, too-few-public-methods, no-else-return, unsubscriptable-object, comparison-with-callable
PRETRAINED_WEIGHTS = "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5"


class GhostFaceNetClient(FacialRecognition):
    """
    GhostFaceNet model (GhostFaceNetV1 backbone)
    Repo: https://github.com/HamadYA/GhostFaceNets
    Pre-trained weights: https://github.com/HamadYA/GhostFaceNets/releases/tag/v1.2
        GhostFaceNet_W1.3_S1_ArcFace.h5 ~ 16.5MB
        Author declared that this backbone and pre-trained weights got 99.7667% accuracy on LFW
    """

    def __init__(self):
        self.model_name = "GhostFaceNet"
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.model = load_model()


def load_model():
    model = GhostFaceNetV1()

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="ghostfacenet_v1.h5", source_url=PRETRAINED_WEIGHTS
    )

    model.load_weights(weight_file)

    return model


def GhostFaceNetV1() -> Model:
    """
    Build GhostFaceNetV1 model. Refactored from
        github.com/HamadYA/GhostFaceNets/blob/main/backbones/ghost_model.py
    Returns:
        model (Model)
    """
    inputs = Input(shape=(112, 112, 3))

    out_channel = 20

    nn = Conv2D(
        out_channel,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        ),
    )(inputs)

    nn = BatchNormalization(axis=-1)(nn)
    nn = Activation("relu")(nn)

    dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
    exps = [20, 64, 92, 92, 156, 312, 260, 240, 240, 624, 872, 872, 1248, 1248, 1248, 664]
    outs = [20, 32, 32, 52, 52, 104, 104, 104, 104, 144, 144, 208, 208, 208, 208, 208]
    strides_set = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    reductions = [0, 0, 0, 24, 40, 0, 0, 0, 0, 156, 220, 220, 0, 312, 0, 168]

    pre_out = out_channel
    for dwk, stride, exp, out, reduction in zip(dwkernels, strides_set, exps, outs, reductions):
        shortcut = not (out == pre_out and stride == 1)
        nn = ghost_bottleneck(nn, dwk, stride, exp, out, reduction, shortcut)
        pre_out = out

    nn = Conv2D(
        664,
        (1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        kernel_initializer=keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        ),
    )(nn)
    nn = BatchNormalization(axis=-1)(nn)
    nn = Activation("relu")(nn)

    xx = Model(inputs=inputs, outputs=nn, name="GhostFaceNetV1")

    # post modelling
    inputs = xx.inputs[0]
    nn = xx.outputs[0]

    nn = keras.layers.DepthwiseConv2D(nn.shape[1], use_bias=False, name="GDC_dw")(nn)
    nn = keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, name="GDC_batchnorm")(nn)
    nn = keras.layers.Conv2D(
        512, 1, use_bias=True, kernel_initializer="glorot_normal", name="GDC_conv"
    )(nn)
    nn = keras.layers.Flatten(name="GDC_flatten")(nn)

    embedding = keras.layers.BatchNormalization(
        momentum=0.99, epsilon=0.001, scale=True, name="pre_embedding"
    )(nn)
    embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)

    model = keras.models.Model(inputs, embedding_fp32, name=xx.name)
    model = replace_relu_with_prelu(model=model)
    return model


def se_module(inputs, reduction):
    """
    Refactored from github.com/HamadYA/GhostFaceNets/blob/main/backbones/ghost_model.py
    """
    # get the channel axis
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = channel axis shape
    filters = inputs.shape[channel_axis]

    # from None x H x W x C to None x C
    se = GlobalAveragePooling2D()(inputs)

    # Reshape None x C to None 1 x 1 x C
    se = Reshape((1, 1, filters))(se)

    # Squeeze by using C*se_ratio. The size will be 1 x 1 x C*se_ratio
    se = Conv2D(
        reduction,
        kernel_size=1,
        use_bias=True,
        kernel_initializer=keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        ),
    )(se)
    se = Activation("relu")(se)

    # Excitation using C filters. The size will be 1 x 1 x C
    se = Conv2D(
        filters,
        kernel_size=1,
        use_bias=True,
        kernel_initializer=keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        ),
    )(se)
    se = Activation("hard_sigmoid")(se)

    return Multiply()([inputs, se])


def ghost_module(inputs, out, convkernel=1, dwkernel=3, add_activation=True):
    """
    Refactored from github.com/HamadYA/GhostFaceNets/blob/main/backbones/ghost_model.py
    """
    conv_out_channel = out // 2
    cc = Conv2D(
        conv_out_channel,
        convkernel,
        use_bias=False,
        strides=(1, 1),
        padding="same",
        kernel_initializer=keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        ),
    )(inputs)
    cc = BatchNormalization(axis=-1)(cc)
    if add_activation:
        cc = Activation("relu")(cc)

    nn = DepthwiseConv2D(
        dwkernel,
        1,
        padding="same",
        use_bias=False,
        depthwise_initializer=keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        ),
    )(cc)
    nn = BatchNormalization(axis=-1)(nn)
    if add_activation:
        nn = Activation("relu")(nn)
    return Concatenate()([cc, nn])


def ghost_bottleneck(inputs, dwkernel, strides, exp, out, reduction, shortcut=True):
    """
    Refactored from github.com/HamadYA/GhostFaceNets/blob/main/backbones/ghost_model.py
    """
    nn = ghost_module(inputs, exp, add_activation=True)
    if strides > 1:
        # Extra depth conv if strides higher than 1
        nn = DepthwiseConv2D(
            dwkernel,
            strides,
            padding="same",
            use_bias=False,
            depthwise_initializer=keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_out", distribution="truncated_normal"
            ),
        )(nn)
        nn = BatchNormalization(axis=-1)(nn)

    if reduction > 0:
        # Squeeze and excite
        nn = se_module(nn, reduction)

    # Point-wise linear projection
    nn = ghost_module(nn, out, add_activation=False)  # ghost2 = GhostModule(exp, out, relu=False)

    if shortcut:
        xx = DepthwiseConv2D(
            dwkernel,
            strides,
            padding="same",
            use_bias=False,
            depthwise_initializer=keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_out", distribution="truncated_normal"
            ),
        )(inputs)
        xx = BatchNormalization(axis=-1)(xx)
        xx = Conv2D(
            out,
            (1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_out", distribution="truncated_normal"
            ),
        )(xx)
        xx = BatchNormalization(axis=-1)(xx)
    else:
        xx = inputs
    return Add()([xx, nn])


def replace_relu_with_prelu(model) -> Model:
    """
    Replaces relu activation function in the built model with prelu.
        Refactored from github.com/HamadYA/GhostFaceNets/blob/main/backbones/ghost_model.py
    Args:
        model (Model): built model with relu activation functions
    Returns
        model (Model): built model with prelu activation functions
    """

    def convert_relu(layer):
        if isinstance(layer, ReLU) or (
            isinstance(layer, Activation) and layer.activation == keras.activations.relu
        ):
            layer_name = layer.name.replace("_relu", "_prelu")
            return PReLU(
                shared_axes=[1, 2],
                alpha_initializer=tf.initializers.Constant(0.25),
                name=layer_name,
            )
        return layer

    input_tensors = keras.layers.Input(model.input_shape[1:])
    return keras.models.clone_model(model, input_tensors=input_tensors, clone_function=convert_relu)
