# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv('.env')

import ast
import glob
import os
import tensorflow as tf

from tensorflow import keras
from pathlib import Path
from model import dataio, model


# specify directory as data io info
DATADIR = Path('/home/ubuntu/hydrafloods')
BASEDIR = Path('/mnt/hydrafloods/output/jrc_adjusted_LR_2020_07_13_V1/model/sentinel1-surface-water')
MODEL_DIR = BASEDIR / 'trial_6ba0bc0ef8458bf43280b5814775bd2b'
CHECKPOINT = MODEL_DIR / 'checkpoints' / 'epoch_0'/ 'checkpoint'

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))
LABELS = ast.literal_eval(os.getenv('LABELS'))
PATCH_SHAPE = ast.literal_eval(os.getenv('PATCH_SHAPE'))
in_shape = (None, None) + (len(FEATURES),)
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

VALIDATION_DIR = DATADIR / 'validation_patches_jrc'
validation_files = glob.glob(str(VALIDATION_DIR) + '/*')
# eval is batched by 1
validation = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

this_model = model.get_model(in_shape, out_classes)

# open and save model
this_model.load_weights(f'{str(CHECKPOINT)}')

# compile the model
this_model.compile(
    optimizer='adam',
    loss=model.bce_loss,
    metrics=[
        keras.metrics.categorical_accuracy,
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        model.dice_coef,
        model.f1_m
    ]
)

# check how the model trained
score = this_model.evaluate(validation)
print(score)
