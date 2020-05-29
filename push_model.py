# -*- coding: utf-8 -*-

from dotenv import load_dotenv
load_dotenv('.env')

import ast
import os
import subprocess
from pathlib import Path
from tensorflow.python.tools import saved_model_utils

MODEL_CREATE = False

PROJECT = os.getenv('GCS_PROJECT')
BUCKET = os.getenv('GCS_BUCKET')

MODEL_NAME = 'mekong_sentinel1-surface-water_vgg16-unet'

# specify directory as data io info
BASEDIR = Path(os.getenv('BASEDIR'))
OUTPUT_DIR = BASEDIR / 'output'
model_dir_name = '2020_05_29_V4'
MODEL_SAVE_DIR = OUTPUT_DIR / model_dir_name
EEIFIED_DIR = MODEL_SAVE_DIR / 'eeified'
GCS_EEIFIED_DIR = os.getenv('GCS_EEIFIED_DIR')

VERSION_NAME = f'v{model_dir_name}'
print('Creating version: ' + VERSION_NAME)

LABELS = ast.literal_eval(os.getenv('LABELS'))

try:
    os.mkdir(EEIFIED_DIR)
except FileExistsError:
    print(f'> {EEIFIED_DIR} exists, skipping creation...')

meta_graph_def = saved_model_utils.get_meta_graph_def(str(MODEL_SAVE_DIR), 'serve')
inputs = meta_graph_def.signature_def['serving_default'].inputs
outputs = meta_graph_def.signature_def['serving_default'].outputs

# Just get the first thing(s) from the serving signature def.  i.e. this
# model only has a single input and a single output.
input_name = None
for k, v in inputs.items():
    input_name = v.name
    break

output_name = None
for k, v in outputs.items():
    output_name = v.name
    break

# Set the project before using the model prepare command.
set_project = f'earthengine set_project {PROJECT}'
result = subprocess.check_output(set_project, shell=True)
print(result)

model_prepare = f'earthengine model prepare --source_dir {MODEL_SAVE_DIR} --dest_dir {EEIFIED_DIR} ' \
                f'--input "{{\\"{input_name}\\":\\"array\\"}}" --output "{{\\"{output_name}\\":\\"{LABELS[0]}\\"}}"'
result = subprocess.check_output(model_prepare, shell=True)
print(result)

# this creates an temp directory inside the eeified dir
files = [name for name in os.listdir(str(EEIFIED_DIR)) if os.path.isfile(os.path.join(str(EEIFIED_DIR), name))]
if len(files) == 0:
    temp_dir = os.listdir(str(EEIFIED_DIR))[0]
    EEIFIED_DIR = EEIFIED_DIR / temp_dir

# copy the eeified to the google bucket
GS_EEIFIED_PATH = f'gs://{BUCKET}/{GCS_EEIFIED_DIR}/{model_dir_name}'
copy = f'gsutil -m cp -R {EEIFIED_DIR} {GS_EEIFIED_PATH}'
result = subprocess.check_output(copy, shell=True)
print(result)

if MODEL_CREATE:
    # create a model
    ai_model_create = f'gcloud ai-platform models create {MODEL_NAME} --project {PROJECT}'
    result = subprocess.check_output(ai_model_create, shell=True)
    print(result)

# provide staging-bucket if pushing from local directory
# be careful with the python version
ai_version_create = f'gcloud ai-platform versions create {VERSION_NAME} --project {PROJECT} --model {MODEL_NAME} ' \
                    f'--origin {str(GS_EEIFIED_PATH)} --runtime-version=2.1  --python-version=3.7'
result = subprocess.check_output(ai_version_create, shell=True)
print(result)
