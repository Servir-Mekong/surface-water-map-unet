# -*- coding: utf-8 -*-

from functools import partial
import tensorflow as tf


@tf.function
def parse_tfrecord(example_proto, features=None, labels=None, patch_shape=None):
    keys = features + labels
    columns = [
        tf.io.FixedLenFeature(shape=patch_shape, dtype=tf.float32) for k in keys
    ]
    proto_struct = dict(zip(keys, columns))
    inputs = tf.io.parse_single_example(example_proto, proto_struct)
    inputs_list = [inputs.get(key) for key in keys]
    stacked = tf.stack(inputs_list, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    return tf.data.Dataset.from_tensors(stacked)


@tf.function
def to_tuple(dataset, n_features=None):
    features = dataset[:, :, :, :n_features]
    labels = dataset[:, :, :, n_features:]
    return features, labels


@tf.function
def random_transform(dataset):
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


def get_dataset(files, features, labels, patch_shape, batch_size, buffer_size=1000, training=False):
    parser = partial(parse_tfrecord,
                     features=features,
                     labels=labels,
                     patch_shape=patch_shape
                     )

    split_data = partial(to_tuple, n_features=len(features))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.interleave(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)\
            .batch(batch_size)\
            .map(random_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
