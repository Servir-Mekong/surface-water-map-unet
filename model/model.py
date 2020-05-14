from functools import partial

# Tensorflow setup.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


convLayer = partial(layers.Conv2D,
                    padding='same',
                    kernel_initializer='he_normal',
                    bias_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    bias_regularizer=keras.regularizers.l2(0.001)
                    )


def decoder_block(inputTensor, concatTensor=None, nFilters=512, nConvs=2, i=0, name_prefix="decoder_block", rate=0.2, noise=1, activation='relu', combo='add'):
    deconv = inputTensor
    for j in range(nConvs):
        deconv = convLayer(
            nFilters, (3, 3), name=f"{name_prefix}{i}_deconv{j+1}")(deconv)
        deconv = layers.BatchNormalization(
            name=f"{name_prefix}{i}_batchnorm{j+1}")(deconv)
        deconv = layers.Activation(
            activation, name=f"{name_prefix}{i}_activation{j+1}")(deconv)
        deconv = layers.GaussianNoise(
            stddev=noise, name=f"{name_prefix}{i}_noise{j+1}")(deconv)

        if j == 0 and concatTensor is not None:
            deconv = layers.Dropout(
                rate=rate, name=f'{name_prefix}{i}_dropout')(deconv)
            if combo == 'add':
                deconv = layers.add([deconv, concatTensor],
                                    name=f"{name_prefix}{i}_residual")
            elif combo == 'concat':
                deconv = layers.concatenate(
                    [deconv, concatTensor], name=f"{name_prefix}{i}_concat")

    up = layers.UpSampling2D(interpolation='bilinear',
                             name=f"{name_prefix}{i}_upsamp")(deconv)
    return up


def addFeatures(input_tensor):
    def normalizedDifference(c1, c2, name="nd"):
        return layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1] + 1e-7)), name=name)([c1, c2])

    def whiteness(r, g, b):
        mean = layers.average([r, g, b], name="vis_mean")
        rx = layers.Lambda(lambda x: K.abs(
            (x[0] - x[1]) / (x[1] + 1e-7)), name="r_centered")([r, mean])
        gx = layers.Lambda(lambda x: K.abs(
            (x[0] - x[1]) / (x[1] + 1e-7)), name="g_centered")([g, mean])
        bx = layers.Lambda(lambda x: K.abs(
            (x[0] - x[1]) / (x[1] + 1e-7)), name="b_centered")([b, mean])
        return layers.add([rx, gx, bx], name="whiteness")

    ndvi = normalizedDifference(
        input_tensor[:, :, :, 3:4], input_tensor[:, :, :, 2:3], name="ndvi_feature")
    mndwi = normalizedDifference(
        input_tensor[:, :, :, 1:2], input_tensor[:, :, :, 5:6], name="mndwi_feature")
    ndbi = normalizedDifference(
        input_tensor[:, :, :, 4:5], input_tensor[:, :, :, 3:4], name="ndbi_feature")
    white = whiteness(
        input_tensor[:, :, :, 2:3], input_tensor[:, :, :, 1:2], input_tensor[:, :, :, 0:1])

    return layers.concatenate([input_tensor, ndvi, mndwi, ndbi, white], name='input_features')


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



def getModel(inShape, outClasses, dropoutRate=0.2, noise=1, activation='relu', combo='add',regression=False):
    inTensor = layers.Input(shape=inShape, name='input')
    inTensor = addFeatures(inTensor)

    vgg19 = keras.applications.VGG19(
        include_top=False, weights=None, input_tensor=inTensor)

    base_in = vgg19.input
    base_out = vgg19.output
    concat_layers = ["block5_conv4", "block4_conv4",
                     "block3_conv4", "block2_conv2", "block1_conv2"]
    concat_tensors = [vgg19.get_layer(layer).output for layer in concat_layers]

    decoder0 = decoder_block(base_out, nFilters=1028, nConvs=3,
                             i=0, rate=rate, noise=noise, activation=activation)  # 64
    decoder1 = decoder_block(
        decoder0, concatTensor=concat_tensors[0], nFilters=512, nConvs=3, i=2, rate=rate, noise=noise, activation=activation)
    decoder2 = decoder_block(
        decoder1, concatTensor=concat_tensors[1], nFilters=512, nConvs=3, i=3, rate=rate, noise=noise, activation=activation)
    decoder3 = decoder_block(
        decoder2, concatTensor=concat_tensors[2], nFilters=256, nConvs=2, i=4, rate=rate, noise=noise, activation=activation)
    decoder4 = decoder_block(
        decoder3, concatTensor=concat_tensors[3], nFilters=128, nConvs=2, i=5, rate=rate, noise=noise, activation=activation)

    outBranch = convLayer(64, (3, 3), name=f"out_block_conv1")(decoder4)
    outBranch = layers.BatchNormalization(
        name="out_block_batchnorm1")(outBranch)
    outBranch = layers.Activation(
        activation, name="out_block_activation1")(outBranch)
    if combo == 'add':
        outBranch = layers.add(
            [outBranch, concat_tensors[4]], name="out_block_residual")
    elif combo == 'concat':
        outBranch = layers.concatenate(
            [outBranch, concat_tensors[4]], name="out_block_concat")
    outBranch = layers.SpatialDropout2D(
        rate=rate, seed=0, name="out_block_spatialdrop")(outBranch)
    outBranch = convLayer(64, (5, 5), name="out_block_conv2")(outBranch)
    outBranch = layers.BatchNormalization(
        name="out_block_batchnorm2")(outBranch)
    outBranch = layers.Activation(
        activation, name="out_block_activation2")(outBranch)

    output = layers.Conv2D(outClasses, (1, 1), name='final_conv')(outBranch)
    if regression:
        outActivation = "linear"
    else:
        if outClasses == 1:
            outActivation = "sigmoid"
        else:
            outActivation = "softmax"
    output = layers.Activation(outActivation, name='final_out')(output)

    model = models.Model(inputs=[base_in], outputs=[output], name="vgg19-unet")

    return model


def build(*args, optimizer=None, loss=None, metrics=None, distributedStrategy=None, **kwargs):

    if optimizer is None:
        optimizer = keras.optimizers.Adam()

    if loss is None:
        loss = bce_dice_loss

    if metrics is None:
        metrics = [keras.metrics.categorical_accuracy,
                   keras.metrics.Precision(),
                   keras.metrics.Recall(),
                   dice_coef
                   ]

    if distributedStrategy is not None:
        with strategy.scope():
            model = getModel(*args, **kwargs)

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics
                          )

    else:
        model = getModel(*args, **kwargs)

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics
                      )

    return model
