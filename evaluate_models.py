# -*- coding: utf-8 -*-

from dotenv import load_dotenv

load_dotenv('.env')

import ast
import glob
import os
import tensorflow as tf

from pathlib import Path
from model import dataio, model


# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG/hydrafloods/output')
MODEL_DIR = BASEDIR / 'trial_469ae9d7b6c82488deb9be9c0a0a25e7'
H5_MODEL = MODEL_DIR / 'tf-model-h5'

# specify some data structure
FEATURES = ast.literal_eval(os.getenv('FEATURES'))
LABELS = ast.literal_eval(os.getenv('LABELS'))
PATCH_SHAPE = ast.literal_eval(os.getenv('PATCH_SHAPE'))
in_shape = (None, None) + (len(FEATURES),)
out_classes = int(os.getenv('OUT_CLASSES_NUM'))

VALIDATION_DIR = BASEDIR / 'validation_patches'
validation_files = glob.glob(str(VALIDATION_DIR) + '/*')
validation = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

this_model = model.get_model(in_shape, out_classes)

# open and save model
this_model.load_weights(f'{str(H5_MODEL)}')

# check how the model trained
score = this_model.evaluate(validation)
print(score)
