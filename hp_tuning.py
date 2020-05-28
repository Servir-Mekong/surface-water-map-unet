# -*- coding: utf-8 -*-

import glob
from pathlib import Path
from functools import partial
from model import dataio, model
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt


# specify directory as data io info
BASEDIR = Path('/home/ubuntu/hydrafloods')
TRAINING_DIR = BASEDIR / 'training_patches'
TESTING_DIR = BASEDIR / 'testing_patches'
VALIDATION_DIR = BASEDIR / 'validation_patches'
OUTPUT_DIR = BASEDIR / 'output'
MODEL_SAVE_DIR = OUTPUT_DIR / 'hyperopt-attempt1'

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
BATCH_SIZE = 64
EPOCHS = 50
BUFFER_SIZE = 9500

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
eval = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

# get distributed strategy and apply distribute i/o and model build
strategy = tf.distribute.MirroredStrategy()

# define tensor input shape and number of classes
in_shape = PATCH_SHAPE + (len(FEATURES),)
out_classes = len(LABELS)

# partial function to pass keywords to for hyper-parameter tuning
get_model = partial(model.get_model, in_shape=in_shape, out_classes=2)


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
    max_trials=10,
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
