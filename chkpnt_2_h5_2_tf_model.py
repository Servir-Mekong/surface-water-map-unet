# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv('.env')

import ast
import os
import tensorflow as tf

from pathlib import Path
from model import model

###############################################################################
# DONT FORGET TO CHANGE THE LEARNING_RATE AND THE LOSS FUNCTION BELOW #########
###############################################################################

print('###############################################################################')
print('# DONT FORGET TO CHANGE THE LEARNING_RATE AND THE LOSS FUNCTION BELOW #########')
print('###############################################################################')

# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG/hydrafloods/output')
MODEL_DIR = BASEDIR / 'trial_469ae9d7b6c82488deb9be9c0a0a25e7'
CHECKPOINT_DIR = MODEL_DIR / '469ae9d7b6c82488deb9be9c0a0a25e7' / 'checkpoint'
H5_MODEL = MODEL_DIR / 'tf-model-h5'
TF_MODEL_DIR = MODEL_DIR / 'tf-model'

try:
    os.mkdir(TF_MODEL_DIR)
except FileExistsError:
    print(f'> {TF_MODEL_DIR} exists, skipping..')

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))

LEARNING_RATE = 0.0001
LOSS = model.bce_loss

in_shape = (None, None) + (len(FEATURES),)
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

# optimizer refers to adam
this_model = model.build(in_shape, out_classes, learning_rate=LEARNING_RATE, optimizer=None, loss=LOSS)

# open and save model
this_model.load_weights(f'{str(H5_MODEL)}')
print(this_model.summary())
tf.keras.models.save_model(this_model, str(TF_MODEL_DIR), save_format='tf')
