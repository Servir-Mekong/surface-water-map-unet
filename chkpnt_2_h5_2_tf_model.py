# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv('.env')

import ast
import os
import tensorflow as tf

from pathlib import Path
from model import model

# specify directory as data io info
BASEDIR = Path('/mnt/hydrafloods/output/jrc_adjusted_LR_2020_07_13_V1/model/sentinel1-surface-water')
MODEL_DIR = BASEDIR / 'trial_6ba0bc0ef8458bf43280b5814775bd2b'
CHECKPOINT_DIR = MODEL_DIR / 'checkpoints' / 'epoch_0' / 'checkpoint'
H5_MODEL = MODEL_DIR / 'tf-model-h5'
TF_MODEL_DIR = MODEL_DIR / 'tf-model'

try:
    os.mkdir(TF_MODEL_DIR)
except FileExistsError:
    print(f'> {TF_MODEL_DIR} exists, skipping..')

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))
in_shape = (None, None) + (len(FEATURES),)
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

this_model = model.get_model(in_shape, out_classes)

# open and save model
this_model.load_weights(f'{str(H5_MODEL)}')
print(this_model.summary())
tf.keras.models.save_model(this_model, str(TF_MODEL_DIR), save_format='tf')
