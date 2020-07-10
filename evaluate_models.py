# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv('.env')

import ast
import os
import tensorflow as tf

from pathlib import Path
from model import model


# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG/hydrafloods/output')
MODEL_DIR = BASEDIR / 'trial_469ae9d7b6c82488deb9be9c0a0a25e7'
CHECKPOINT_DIR = MODEL_DIR / '469ae9d7b6c82488deb9be9c0a0a25e7' / 'checkpoint'
H5_MODEL = MODEL_DIR / 'tf-model-h5'
TF_MODEL_DIR = MODEL_DIR / 'tf-model'

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))
in_shape = (None, None) + (len(FEATURES),)
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

this_model = model.get_model(in_shape, out_classes)

# open and save model
this_model.load_weights(f'{str(H5_MODEL)}')

# check how the model trained
this_model.evaluate(validation)
