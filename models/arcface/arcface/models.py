import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import (
    ResNet50
)
from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                        weights=weights)(x_in)
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = Backbone(use_pretrain=use_pretrain)(x)
    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        logist = ArcHead(num_classes=num_classes, margin=margin,
                         logist_scale=logist_scale)(embds, labels)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)
