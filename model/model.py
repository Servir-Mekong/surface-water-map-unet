# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend as K

conv_layer = partial(layers.Conv2D,
                     padding='same',
                     kernel_initializer='he_normal',
                     bias_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     bias_regularizer=keras.regularizers.l2(0.001)
                     )


def decoder_block(input_tensor, concat_tensor=None, n_filters=512, n_convs=2, i=0, rate=0.2,
                  name_prefix='decoder_block', noise=1, activation='relu', combo='add', **kwargs):
    deconv = input_tensor
    for j in range(n_convs):
        deconv = conv_layer(n_filters, (3, 3), name=f'{name_prefix}{i}_deconv{j + 1}')(deconv)
        deconv = layers.BatchNormalization(name=f'{name_prefix}{i}_batchnorm{j + 1}')(deconv)
        deconv = layers.Activation(activation, name=f'{name_prefix}{i}_activation{j + 1}')(deconv)
        deconv = layers.GaussianNoise(stddev=noise, name=f'{name_prefix}{i}_noise{j + 1}')(deconv)

        if j == 0 and concat_tensor is not None:
            deconv = layers.Dropout(rate=rate, name=f'{name_prefix}{i}_dropout')(deconv)
            if combo == 'add':
                deconv = layers.add([deconv, concat_tensor], name=f'{name_prefix}{i}_residual')
            elif combo == 'concat':
                deconv = layers.concatenate([deconv, concat_tensor], name=f'{name_prefix}{i}_concat')

    up = layers.UpSampling2D(interpolation='bilinear', name=f'{name_prefix}{i}_upsamp')(deconv)
    return up


def add_features(input_tensor):
    def normalized_difference(c1, c2, name='nd'):
        nd_f = layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1])), name=name)([c1, c2])
        # nd_inf = layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1] + 1e-7)), name=f'{name}_inf')([c1, c2])
        nd_inf = layers.Lambda(lambda x: (x[0] - x[1]), name=f'{name}_inf')([c1, c2])
        return tf.where(tf.math.is_finite(nd_f), nd_f, nd_inf)

    def ratio(c1, c2, name='ratio'):
        ratio_f = layers.Lambda(lambda x: x[0] / x[1], name=name)([c1, c2])
        # ratio_inf = layers.Lambda(lambda x: x[0] / (x[1] + 1e-7), name=f'{name}_inf')([c1, c2])
        ratio_inf = layers.Lambda(lambda x: x[0], name=f'{name}_inf')([c1, c2])
        return tf.where(tf.math.is_finite(ratio_f), ratio_f, ratio_inf)

    def nvi(c1, c2, name='nvi'):
        nvi_f = layers.Lambda(lambda x: x[0] / (x[0] + x[1]), name=name)([c1, c2])
        # nvi_inf = layers.Lambda(lambda x: x[0] / (x[0] + x[1] + 1e-7), name=f'{name}_inf')([c1, c2])
        nvi_inf = layers.Lambda(lambda x: x[0], name=f'{name}_inf')([c1, c2])
        return tf.where(tf.math.is_finite(nvi_f), nvi_f, nvi_inf)

    nd = normalized_difference(input_tensor[:, :, :, 0:1], input_tensor[:, :, :, 1:2])  # vh, vv
    ratio = ratio(input_tensor[:, :, :, 1:2], input_tensor[:, :, :, 0:1])  # vv, vh
    nvhi = nvi(input_tensor[:, :, :, 0:1], input_tensor[:, :, :, 1:2], name='nvhi')  # vh, vv
    nvvi = nvi(input_tensor[:, :, :, 1:2], input_tensor[:, :, :, 0:1], name='nvvi')  # vv, vh
    return layers.concatenate([input_tensor, nd, ratio, nvhi, nvvi], name='input_features')


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    true_sum = K.sum(K.square(y_true), -1)
    pred_sum = K.sum(K.square(y_pred), -1)
    return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))


def bce_dice_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2) + dice_loss(y_true, y_pred)


def bce_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)


def get_model(in_shape, out_classes, dropout_rate=0.2, noise=1,
              activation='relu', combo='add', **kwargs):
    in_tensor = layers.Input(shape=in_shape, name='input')
    in_tensor = add_features(in_tensor)

    vgg19 = keras.applications.VGG19(include_top=False, weights=None, input_tensor=in_tensor)

    base_in = vgg19.input
    base_out = vgg19.output
    concat_layers = ['block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2']
    concat_tensors = [vgg19.get_layer(layer).output for layer in concat_layers]

    decoder0 = decoder_block(
        base_out, n_filters=1028, n_convs=3, noise=noise,
        i=0, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )  # 64
    decoder1 = decoder_block(
        decoder0, concat_tensor=concat_tensors[0], n_filters=512, n_convs=3, noise=noise,
        i=1, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )
    decoder2 = decoder_block(
        decoder1, concat_tensor=concat_tensors[1], n_filters=512, n_convs=3, noise=noise,
        i=2, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )
    decoder3 = decoder_block(
        decoder2, concat_tensor=concat_tensors[2], n_filters=256, n_convs=2, noise=noise,
        i=3, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )
    decoder4 = decoder_block(
        decoder3, concat_tensor=concat_tensors[3], n_filters=128, n_convs=2, noise=noise,
        i=4, rate=dropout_rate, activation=activation, combo=combo, **kwargs
    )

    out_branch = conv_layer(64, (3, 3), name=f'out_block_conv1')(decoder4)
    out_branch = layers.BatchNormalization(name=f'out_block_batchnorm1')(out_branch)
    out_branch = layers.Activation(activation, name='out_block_activation1')(out_branch)
    if combo == 'add':
        out_branch = layers.add([out_branch, concat_tensors[4]], name='out_block_residual')
    elif combo == 'concat':
        out_branch = layers.concatenate([out_branch, concat_tensors[4]], name='out_block_concat')
    out_branch = layers.SpatialDropout2D(rate=dropout_rate, seed=0, name='out_block_spatialdrop')(out_branch)
    out_branch = conv_layer(64, (5, 5), name='out_block_conv2')(out_branch)
    out_branch = layers.BatchNormalization(name='out_block_batchnorm2')(out_branch)
    out_branch = layers.Activation(activation, name='out_block_activation2')(out_branch)

    out_activation = kwargs.get('out_activation', 'softmax')
    output = layers.Conv2D(out_classes, (1, 1), activation=out_activation, name='final_conv')(out_branch)
    model = models.Model(inputs=[base_in], outputs=[output], name='vgg19-unet')
    return model


def build(*args, optimizer=None, loss=None, metrics=None, distributed_strategy=None, **kwargs):
    learning_rate = kwargs.get('learning_rate', 0.001)
    if optimizer == 'sgd_momentum':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    if loss is None:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)

    if metrics is None:
        metrics = [
            keras.metrics.categorical_accuracy,
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            dice_coef,
            f1_m
        ]

    if distributed_strategy is not None:
        with distributed_strategy.scope():
            model = get_model(*args, **kwargs)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        model = get_model(*args, **kwargs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
