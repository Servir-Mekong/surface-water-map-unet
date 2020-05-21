# -*- coding: utf-8 -*-

import glob
import numpy as np
from pathlib import Path
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from model import dataio, model
from utils import file_utils

# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG/hydrafloods')
DATADIR = BASEDIR / 'data'
TRAINING_DIR = BASEDIR / 'training'
TESTING_DIR = BASEDIR / 'testing'
VALIDATION_DIR = BASEDIR / 'validation'
OUTPUT_DIR = BASEDIR / 'output'
MODEL_SAVE_DIR = OUTPUT_DIR / 'attempt1'
MODEL_NAME = 'vgg19_custom_unet_model'
MODEL_CHECKPOINT_NAME = 'bestModelWeights'

try:
    os.mkdir(MODEL_SAVE_DIR)
except FileExistsError:
    print(f'> {MODEL_SAVE_DIR} exists, skipping creation...')

# specify some data structure
FEATURES = ['VH', 'VV']
LABELS = ['class']

# patch size for training
KERNEL_SIZE = 256
PATCH_SHAPE = (KERNEL_SIZE, KERNEL_SIZE)

# Sizes of the training and evaluation datasets.
# based on sizes of exported data and spliting performed earlier
# ~13542 samples
# ~70% are training, ~20% are testing, ~10% are validation
TRAIN_SIZE = 9480
TEST_SIZE = 2709
VAL_SIZE = 1354

# Specify model training parameters.
BATCH_SIZE = 32
EPOCHS = 50
BUFFER_SIZE = 10000

# Rates
LEARNING_RATE = 0.003
DROPOUT_RATE = 0.15

# other params w/ notes
CALLBACK_PARAMETER = 'val_loss'
RANDOM_TRANSFORM = False
COMBINATION = 'concat'
NOTES = f'gaussian before activation\n \
concat_layers'

if os.path.exists(str(TRAINING_DIR)) and file_utils.file_num_in_folder(str(TRAINING_DIR)) > 1 and \
        os.path.exists(str(TESTING_DIR)) and file_utils.file_num_in_folder(str(TESTING_DIR)) > 1 and \
        os.path.exists(str(VALIDATION_DIR)) and file_utils.file_num_in_folder(str(VALIDATION_DIR)) > 1:
    training_files = glob.glob(str(TRAINING_DIR) + '/*')
    testing_files = glob.glob(str(TRAINING_DIR) + '/*')
    validation_files = glob.glob(str(TRAINING_DIR) + '/*')
else:
    files = glob.glob(str(DATADIR) + '/*')
    DATASET_SIZE = len(files)
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)
    test_size = int(0.1 * DATASET_SIZE)

    np.random.shuffle(files)
    training_files = files[:train_size]
    remaining = files[train_size:]
    np.random.shuffle(files)
    testing_files = remaining[:val_size]
    validation_files = remaining[val_size:]

    # except for folder exists but no training files
    try:
        os.mkdir(TRAINING_DIR)
    except FileExistsError:
        print(f'> {TRAINING_DIR} exists, skipping creation...')

    try:
        os.mkdir(TESTING_DIR)
    except FileExistsError:
        print(f'> {TESTING_DIR} exists, skipping creation...')

    try:
        os.mkdir(VALIDATION_DIR)
    except FileExistsError:
        print(f'> {VALIDATION_DIR} exists, skipping creation...')

    print('> splitting into TRAINING, TESTING and VALIDATION datasets')

    [shutil.copy(str(DATADIR / file), TRAINING_DIR) for file in training_files]
    [shutil.copy(str(DATADIR / file), TESTING_DIR) for file in testing_files]
    [shutil.copy(str(DATADIR / file), VALIDATION_DIR) for file in validation_files]

# get training, testing, and eval TFRecordDataset
# training is batched, shuffled, transformed, and repeated
training = dataio.get_dataset(training_files, FEATURES, LABELS, PATCH_SHAPE, BATCH_SIZE, buffer_size=BUFFER_SIZE,
                              training=True, apply_random_transform=RANDOM_TRANSFORM).repeat()
# testing is batched by 1 and repeated
testing = dataio.get_dataset(testing_files, FEATURES, LABELS, PATCH_SHAPE, 1).repeat()
# eval is batched by 1
eval = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

# get distributed strategy and apply distribute i/o and model build
strategy = tf.distribute.MirroredStrategy()

# define tensor input shape and number of classes
in_shape = PATCH_SHAPE + (len(FEATURES),)
out_classes = len(LABELS)

# build the model and compile
my_model = model.build(in_shape, out_classes, distributed_strategy=strategy, dropout_rate=DROPOUT_RATE,
                       learning_rate=LEARNING_RATE, combo=COMBINATION)

# define callbacks during training
model_checkpoint = callbacks.ModelCheckpoint(
    f'{str(MODEL_SAVE_DIR)}/{MODEL_CHECKPOINT_NAME}.h5',
    monitor=CALLBACK_PARAMETER, save_best_only=True,
    mode='min', verbose=1, save_weights_only=True
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor=CALLBACK_PARAMETER, patience=5, verbose=0,
    mode='auto', restore_best_weights=True
)
tensorboard = callbacks.TensorBoard(log_dir=str(MODEL_SAVE_DIR / 'logs'), write_images=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor=CALLBACK_PARAMETER, factor=0.1,
    patience=5, verbose=1
)

# fit the model
history = my_model.fit(
    x=training,
    epochs=EPOCHS,
    steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
    validation_data=testing,
    validation_steps=TEST_SIZE,
    callbacks=[model_checkpoint, tensorboard, early_stopping, reduce_lr],
)

# check how the model trained
my_model.evaluate(eval)

# save the parameters
with open(f'{str(MODEL_SAVE_DIR)}/parameters.txt', 'w') as f:
    f.write(f'TRAIN_SIZE: {TRAIN_SIZE}\n')
    f.write(f'TEST_SIZE: {TEST_SIZE}\n')
    f.write(f'VAL_SIZE: {VAL_SIZE}\n')
    f.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    f.write(f'EPOCHS: {EPOCHS}\n')
    f.write(f'BUFFER_SIZE: {BUFFER_SIZE}\n')
    f.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    f.write(f'DROPOUT_RATE: {DROPOUT_RATE}\n')
    f.write(f'FEATURES: {FEATURES}\n')
    f.write(f'LABELS: {LABELS}\n')
    f.write(f'PATCH_SHAPE: {PATCH_SHAPE}\n')
    f.write(f'CALLBACK_PARAMETER: {CALLBACK_PARAMETER}\n')
    f.write(f'MODEL_NAME: {MODEL_NAME}.h5\n')
    f.write(f'MODEL_CHECKPOINT_NAME: {MODEL_CHECKPOINT_NAME}.h5\n')
    f.write(f'RANDOM_TRANSFORM: {RANDOM_TRANSFORM}\n')
    f.write(f'COMBINATION: {COMBINATION}\n')
    f.write(f'NOTES: {NOTES}\n')

# save the model
my_model.save(f'{str(MODEL_SAVE_DIR)}/{MODEL_NAME}.h5')
