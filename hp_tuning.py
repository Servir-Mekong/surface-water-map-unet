# -*- coding: utf-8 -*-

import glob
from pathlib import Path
from functools import partial
from model import dataio, model
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt


# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG/hydrafloods')
TRAINING_DIR = BASEDIR / 'training'
TESTING_DIR = BASEDIR / 'testing'
VALIDATION_DIR = BASEDIR / 'validation'
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
BATCH_SIZE = 32
EPOCHS = 50
BUFFER_SIZE = 10000

# other params w/ notes
CALLBACK_PARAMETER = 'val_loss'
RANDOM_TRANSFORM = True

# get list of files for training, testing and eval
training_files = glob.glob(str(TRAINING_DIR) + '/*')
testing_files = glob.glob(str(TRAINING_DIR) + '/*')
validation_files = glob.glob(str(TRAINING_DIR) + '/*')

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

# partial function to pass keywords to for hyper-parameter tuning
get_model = partial(model.get_model, in_shape=in_shape, out_classes=out_classes)


# function to build model for hp tuning
# accepts hp tuning instance to select variables
def build_tuner(hp):
    with strategy.scope():
        # build the model with parameters
        my_model = get_model(dropout_rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.05),
                             noise=hp.Float('noise', min_value=0.2, max_value=2.0, step=0.2),
                             activation=hp.Choice('activation', values=['relu', 'elu']),
                             out_activation=hp.Choice('out_activation', values=['sigmoid', 'softmax']),
                             combo=hp.Choice('combo', values=['add', 'concat']),
                             apply_random_transform=hp.Boolean('apply_random_transform', default=RANDOM_TRANSFORM),
                             )

        # compile model
        my_model.compile(
            optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=model.bce_dice_loss,
            metrics=[
                keras.metrics.categorical_accuracy,
                keras.metrics.Accuracy(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                model.dice_coef,
            ]
        )
    return my_model


tuner = kt.RandomSearch(
    build_tuner,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    seed=0,
    directory=str(MODEL_SAVE_DIR),
    project_name='sentinel1-surface-water'
)

# set early stopping to prevent long running models
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=CALLBACK_PARAMETER, patience=5, verbose=0,
    mode='auto', restore_best_weights=True
)

# fit the model
tuner.search(
        x=training,
        epochs=EPOCHS,
        steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
        validation_data=testing,
        validation_steps=TEST_SIZE,
        callbacks=[early_stopping]
)

tuner.results_summary()
