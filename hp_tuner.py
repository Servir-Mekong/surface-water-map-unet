# -*- coding: utf-8 -*-

from dotenv import load_dotenv
load_dotenv('.env')

import ast
import datetime
import glob
import os
import kerastuner as kt
import tensorflow as tf
from functools import partial
from pathlib import Path
from tensorflow import keras

from model import dataio, model


# specify directory as data io info
BASEDIR = Path(os.getenv('BASEDIR'))
TRAINING_DIR = BASEDIR / 'training_patches'
TESTING_DIR = BASEDIR / 'testing_patches'
VALIDATION_DIR = BASEDIR / 'validation_patches'
OUTPUT_DIR = BASEDIR / 'output'

today = datetime.date.today().strftime('%Y_%m_%d')
iterator = 1
while True:
    model_dir_name = f'hypertuner_{today}_V{iterator}'
    MODEL_SAVE_DIR = OUTPUT_DIR / model_dir_name
    try:
        os.mkdir(MODEL_SAVE_DIR)
    except FileExistsError:
        print(f'> {MODEL_SAVE_DIR} exists, creating another version...')
        iterator += 1
        continue
    break

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))
LABELS = ast.literal_eval(os.getenv('LABELS'))

# patch size for training
PATCH_SHAPE = ast.literal_eval(os.getenv('PATCH_SHAPE'))

# Sizes of the training and evaluation datasets.
TRAIN_SIZE = int(os.getenv('TRAIN_SIZE'))
TEST_SIZE = int(os.getenv('TEST_SIZE'))
VAL_SIZE = int(os.getenv('VAL_SIZE'))

# Specify model training parameters.
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
EPOCHS = int(os.getenv('EPOCHS'))
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE'))

# get list of files for training, testing and eval
training_files = glob.glob(str(TRAINING_DIR) + '/*')
testing_files = glob.glob(str(TRAINING_DIR) + '/*')
validation_files = glob.glob(str(TRAINING_DIR) + '/*')

# get training, testing, and eval TFRecordDataset
# training is batched, shuffled, transformed, and repeated
training = dataio.get_dataset(training_files, FEATURES, LABELS, PATCH_SHAPE, BATCH_SIZE,
                              buffer_size=BUFFER_SIZE, training=True).repeat()
# testing is batched by 1 and repeated
testing = dataio.get_dataset(testing_files, FEATURES, LABELS, PATCH_SHAPE, 1).repeat()
# eval is batched by 1
validation = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

# get distributed strategy and apply distribute i/o and model build
strategy = tf.distribute.MirroredStrategy()

# define tensor input shape and number of classes
in_shape = (None, None) + (len(FEATURES),)
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

# partial function to pass keywords to for hyper-parameter tuning
get_model = partial(model.get_model, in_shape=in_shape, out_classes=out_classes)


# function to build model for hp tuning
# accepts hp tuning instance to select variables
def build_tuner(hp):
    with strategy.scope():
        # build the model with parameters
        # dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.05),
        # noise = hp.Float('noise', min_value=0.2, max_value=2.0, step=0.2),
        # activation = hp.Choice('activation', values=['relu', 'elu']),
        # out_activation = hp.Choice('out_activation', values=['sigmoid', 'softmax']),
        # combo = hp.Choice('combo', values=['add', 'concat']),

        my_model = get_model()

        loss_options = {
            'bce': model.bce_loss,
            'dice': model.dice_loss,
            'bce_dice': model.bce_dice_loss,
        }

        optimizer_options = {
            'sgd_momentum': keras.optimizers.SGD(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5]), momentum=0.9),
            'adam': keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])),
        }

        # compile model
        my_model.compile(
            optimizer=optimizer_options[hp.Choice('optimizer', ['sgd_momentum', 'adam'])],
            loss=loss_options[hp.Choice('loss', ['bce', 'dice', 'bce_dice'])],
            metrics=[
                keras.metrics.categorical_accuracy,
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                model.dice_coef,
                model.f1_m
            ]
        )
    return my_model


tuner = kt.RandomSearch(
    build_tuner,
    objective=kt.Objective('val_f1_m', direction='max'),
    max_trials=500,
    executions_per_trial=2,
    seed=0,
    directory=str(MODEL_SAVE_DIR / 'model'),
    project_name='sentinel1-surface-water'
)

# set early stopping to prevent long running models
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=0,
    mode='auto', restore_best_weights=True
)
tensorboard = keras.callbacks.TensorBoard(log_dir=str(MODEL_SAVE_DIR / 'logs'), write_images=True)

# fit the model
tuner.search(
        x=training,
        epochs=EPOCHS,
        steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
        validation_data=testing,
        validation_steps=TEST_SIZE,
        callbacks=[early_stopping, tensorboard]
)

tuner.results_summary()
