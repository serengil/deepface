# These classes are the tensorflow counterparts of Minivision's Silent-Face-Anti-Spoofing
# Repo licensed under Apache License 2.0
# Ref: github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/model_lib/MiniFASNet.py

# built-in dependencies
from typing import Any, List, Tuple

# project dependencies
from deepface.commons import package_utils

# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model
    from keras.layers import (
        Input,
        ZeroPadding2D,
        Conv2D,
        DepthwiseConv2D,
        BatchNormalization,
        PReLU,
        Activation,
        Add,
        Multiply,
        GlobalAveragePooling2D,
        Reshape,
        Flatten,
        Dense,
        Dropout,
    )
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        ZeroPadding2D,
        Conv2D,
        DepthwiseConv2D,
        BatchNormalization,
        PReLU,
        Activation,
        Add,
        Multiply,
        GlobalAveragePooling2D,
        Reshape,
        Flatten,
        Dense,
        Dropout,
    )

# pytorch's batch normalization defaults, keras' momentum is the complement of pytorch's
BN_EPS = 1e-5
BN_MOMENTUM = 1 - 0.1

keep_dict = {
    "1.8M": [
        32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
        154, 154, 128, 52, 52, 128, 26, 26, 128, 52, 52, 128, 26, 26, 128, 26, 26, 128, 308,
        308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
    ],
    "1.8M_": [
        32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 231, 231, 128,
        231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26, 128, 308,
        308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
    ],
}


def conv_block(
    x: Any,
    out_c: int,
    kernel: Tuple[int, int] = (1, 1),
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    depthwise: bool = False,
    activation: bool = True,
    name: str = "",
) -> Any:
    """
    Convolution without bias, followed by a batch normalization and an optional prelu.
    Layers are named after their pytorch counterparts, dots replaced by underscores.
    Args:
        x: input tensor
        out_c (int): number of output channels, ignored when depthwise
        kernel (tuple): kernel size
        stride (tuple): stride
        padding (tuple): zero padding applied before the convolution, as pytorch does it
        depthwise (bool): convolve each channel on its own
        activation (bool): append a prelu, that is what tells a Conv_block from a Linear_block
        name (str): name prefix of the block
    Returns:
        output tensor
    """
    if padding != (0, 0):
        # pytorch pads symmetrically, whereas keras' "same" padding would not
        x = ZeroPadding2D(padding=padding, name=f"{name}_pad")(x)

    if depthwise:
        x = DepthwiseConv2D(kernel, strides=stride, use_bias=False, name=f"{name}_conv")(x)
    else:
        x = Conv2D(out_c, kernel, strides=stride, use_bias=False, name=f"{name}_conv")(x)

    x = BatchNormalization(epsilon=BN_EPS, momentum=BN_MOMENTUM, name=f"{name}_bn")(x)

    if activation:
        x = PReLU(shared_axes=[1, 2], name=f"{name}_prelu")(x)

    return x


def se_module(x: Any, channels: int, reduction: int, name: str = "") -> Any:
    """
    Squeeze and excitation block
    Args:
        x: input tensor
        channels (int): number of channels to gate
        reduction (int): squeeze ratio
        name (str): name prefix of the block
    Returns:
        output tensor
    """
    se = GlobalAveragePooling2D(name=f"{name}_pool")(x)
    se = Reshape((1, 1, channels), name=f"{name}_reshape")(se)

    se = Conv2D(channels // reduction, 1, use_bias=False, name=f"{name}_fc1")(se)
    se = BatchNormalization(epsilon=BN_EPS, momentum=BN_MOMENTUM, name=f"{name}_bn1")(se)
    se = Activation("relu", name=f"{name}_relu")(se)

    se = Conv2D(channels, 1, use_bias=False, name=f"{name}_fc2")(se)
    se = BatchNormalization(epsilon=BN_EPS, momentum=BN_MOMENTUM, name=f"{name}_bn2")(se)
    se = Activation("sigmoid", name=f"{name}_sigmoid")(se)

    return Multiply(name=f"{name}_scale")([x, se])


def depth_wise(
    x: Any,
    c1: Tuple[int, int],
    c2: Tuple[int, int],
    c3: Tuple[int, int],
    shortcut: bool = False,
    kernel: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (2, 2),
    padding: Tuple[int, int] = (1, 1),
    se_reduct: int = 0,
    name: str = "",
) -> Any:
    """
    Expanding pointwise convolution, depthwise convolution and a linear projection,
    optionally gated by a squeeze and excitation block and added to a shortcut.
    Args:
        x: input tensor
        c1 (tuple): in and out channels of the expanding convolution
        c2 (tuple): in and out channels of the depthwise convolution
        c3 (tuple): in and out channels of the projection
        shortcut (bool): add the input to the output
        kernel (tuple): kernel size of the depthwise convolution
        stride (tuple): stride of the depthwise convolution
        padding (tuple): zero padding of the depthwise convolution
        se_reduct (int): squeeze ratio, 0 disables the squeeze and excitation block
        name (str): name prefix of the block
    Returns:
        output tensor
    """
    short_cut = x

    x = conv_block(x, c1[1], kernel=(1, 1), stride=(1, 1), name=f"{name}_conv")
    x = conv_block(
        x,
        c2[1],
        kernel=kernel,
        stride=stride,
        padding=padding,
        depthwise=True,
        name=f"{name}_conv_dw",
    )
    x = conv_block(x, c3[1], kernel=(1, 1), stride=(1, 1), activation=False, name=f"{name}_project")

    if shortcut:
        if se_reduct > 0:
            x = se_module(x, c3[1], se_reduct, name=f"{name}_se_module")
        x = Add(name=f"{name}_add")([short_cut, x])

    return x


def residual(
    x: Any,
    c1: List[Tuple[int, int]],
    c2: List[Tuple[int, int]],
    c3: List[Tuple[int, int]],
    num_block: int,
    kernel: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (1, 1),
    se_reduct: int = 0,
    name: str = "",
) -> Any:
    """
    A set of residual blocks. When a squeeze ratio is given, only the last block is gated.
    Args:
        x: input tensor
        c1, c2, c3 (list): channel pairs of each block
        num_block (int): number of blocks
        kernel (tuple): kernel size of the depthwise convolutions
        stride (tuple): stride of the depthwise convolutions
        padding (tuple): zero padding of the depthwise convolutions
        se_reduct (int): squeeze ratio, 0 disables the squeeze and excitation blocks
        name (str): name prefix of the set
    Returns:
        output tensor
    """
    for i in range(num_block):
        x = depth_wise(
            x,
            c1[i],
            c2[i],
            c3[i],
            shortcut=True,
            kernel=kernel,
            stride=stride,
            padding=padding,
            se_reduct=se_reduct if i == num_block - 1 else 0,
            name=f"{name}_model_{i}",
        )
    return x


def MiniFASNet(
    keep: List[int],
    embedding_size: int = 128,
    conv6_kernel: Tuple[int, int] = (7, 7),
    drop_p: float = 0.0,
    num_classes: int = 3,
    img_channel: int = 3,
    se_reduct: int = 0,
) -> Model:
    """
    Build a MiniFASNet, the squeeze and excitation flavour of it when a squeeze ratio is given
    Args:
        keep (list): channel widths of the network
        embedding_size (int): size of the embedding layer
        conv6_kernel (tuple): kernel size of the global depthwise convolution
        drop_p (float): dropout ratio
        num_classes (int): number of classes to predict
        img_channel (int): number of channels of the input
        se_reduct (int): squeeze ratio, 0 builds a network without squeeze and excitation
    Returns:
        model (Model)
    """
    inputs = Input(shape=(80, 80, img_channel))

    x = conv_block(inputs, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), name="conv1")
    x = conv_block(
        x, keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), depthwise=True, name="conv2_dw"
    )

    x = depth_wise(
        x,
        (keep[1], keep[2]),
        (keep[2], keep[3]),
        (keep[3], keep[4]),
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        name="conv_23",
    )

    c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
    c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
    c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
    x = residual(x, c1, c2, c3, num_block=4, se_reduct=se_reduct, name="conv_3")

    x = depth_wise(
        x,
        (keep[16], keep[17]),
        (keep[17], keep[18]),
        (keep[18], keep[19]),
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        name="conv_34",
    )

    c1 = [
        (keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]),
        (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35]),
    ]
    c2 = [
        (keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]),
        (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36]),
    ]
    c3 = [
        (keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]),
        (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37]),
    ]
    x = residual(x, c1, c2, c3, num_block=6, se_reduct=se_reduct, name="conv_4")

    x = depth_wise(
        x,
        (keep[37], keep[38]),
        (keep[38], keep[39]),
        (keep[39], keep[40]),
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        name="conv_45",
    )

    c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
    c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
    c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
    x = residual(x, c1, c2, c3, num_block=2, se_reduct=se_reduct, name="conv_5")

    x = conv_block(x, keep[47], kernel=(1, 1), stride=(1, 1), name="conv_6_sep")
    x = conv_block(
        x,
        keep[48],
        kernel=conv6_kernel,
        stride=(1, 1),
        depthwise=True,
        activation=False,
        name="conv_6_dw",
    )

    # the feature maps are 1x1 sized here, so flattening them is free of any ordering issue
    x = Flatten(name="conv_6_flatten")(x)

    if embedding_size != 512:
        x = Dense(embedding_size, use_bias=False, name="linear")(x)

    x = BatchNormalization(epsilon=BN_EPS, momentum=BN_MOMENTUM, name="bn")(x)
    x = Dropout(drop_p, name="drop")(x)
    x = Dense(num_classes, use_bias=False, name="prob")(x)

    return Model(inputs=inputs, outputs=x, name="MiniFASNet")


def MiniFASNetV2(
    embedding_size: int = 128,
    conv6_kernel: Tuple[int, int] = (7, 7),
    drop_p: float = 0.2,
    num_classes: int = 3,
    img_channel: int = 3,
) -> Model:
    """
    MiniFASNetV2 model
    Returns:
        model (Model)
    """
    return MiniFASNet(
        keep_dict["1.8M_"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel
    )


def MiniFASNetV1SE(
    embedding_size: int = 128,
    conv6_kernel: Tuple[int, int] = (7, 7),
    drop_p: float = 0.75,
    num_classes: int = 3,
    img_channel: int = 3,
) -> Model:
    """
    MiniFASNetV1SE model - MiniFASNetV1 with squeeze and excitation blocks
    Returns:
        model (Model)
    """
    return MiniFASNet(
        keep_dict["1.8M"],
        embedding_size,
        conv6_kernel,
        drop_p,
        num_classes,
        img_channel,
        se_reduct=4,
    )
