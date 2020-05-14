
from functools import partial

# Tensorflow setup.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


@tf.function
def parseTfrecord(example_proto, features=None, labels=None, patchShape=None):
    keys = features + labels
    columns = [
        tf.io.FixedLenFeature(shape=patchShape, dtype=tf.float32) for k in keys
    ]
    proto_struct = dict(zip(keys, columns))
    inputs = tf.io.parse_single_example(example_proto, proto_struct)
    inputsList = [inputs.get(key) for key in keys]
    stacked = tf.stack(inputsList, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    return tf.data.Dataset.from_tensors(stacked)


@tf.function
def toTuple(dataset, nFeatures=None):
    features = dataset[:, :, :,:nFeatures]
    labels = dataset[:, :, :, nFeatures:]
    return features, labels


@tf.function
def randomTransform(dataset):
    x = tf.random.uniform(())

    if x < 0.10:
        dataset = tf.image.flip_left_right(dataset)
    elif tf.math.logical_and(x >= 0.10, x < 0.20):
        dataset = tf.image.flip_up_down(dataset)
    elif tf.math.logical_and(x >= 0.20, x < 0.30):
        dataset = tf.image.flip_left_right(tf.image.flip_up_down(dataset))
    elif tf.math.logical_and(x >= 0.30, x < 0.40):
        dataset = tf.image.rot90(dataset, k=1)
    elif tf.math.logical_and(x >= 0.40, x < 0.50):
        dataset = tf.image.rot90(dataset, k=2)
    elif tf.math.logical_and(x >= 0.50, x < 0.60):
        dataset = tf.image.rot90(dataset, k=3)
    elif tf.math.logical_and(x >= 0.60, x < 0.70):
        dataset = tf.image.flip_left_right(tf.image.rot90(dataset, k=2))
    else:
        pass

    return dataset


def getDataset(files, features, labels, patchShape, batchSize, bufferSize=1000, training=False):
    parser = partial(parseTfrecord,
                     features=features,
                     labels=labels,
                     patchShape=patchShape
                     )

    splitData = partial(toTuple, nFeatures=len(features))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.interleave(
        parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(bufferSize, reshuffle_each_iteration=True).batch(batchSize)\
            .map(randomTransform, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(splitData, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(batchSize).map(
            splitData, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
