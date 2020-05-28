# -*- coding: utf-8 -*-

import os
import subprocess
import time
from pathlib import Path
from tensorflow.python.tools import saved_model_utils

PROJECT = 'servir-ee'
BUCKET = 'mekong-tf'

MODEL_NAME = 'test_model'
VERSION_NAME = f'v{int(time.time())}'
print('Creating version: ' + VERSION_NAME)

# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG')
OUTPUT_DIR = BASEDIR / 'hydrafloods'
MODEL_DIR = OUTPUT_DIR / 'to_push'
EEIFIED_DIR_NAME = 'eeified-temp'
EEIFIED_DIR = MODEL_DIR / EEIFIED_DIR_NAME

LABEL = 'class'

try:
    os.mkdir(EEIFIED_DIR)
except FileExistsError:
    print(f'> {EEIFIED_DIR} exists, skipping creation...')

meta_graph_def = saved_model_utils.get_meta_graph_def(str(MODEL_DIR), 'serve')
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

model_prepare = f'earthengine model prepare --source_dir {MODEL_DIR} --dest_dir {EEIFIED_DIR} ' \
                f'--input "{{\\"{input_name}\\":\\"array\\"}}" --output "{{\\"{output_name}\\":\\"{LABEL}\\"}}"'
result = subprocess.check_output(model_prepare, shell=True)
print(result)

# this creates an temp directory inside the eeified dir
files = [name for name in os.listdir(str(EEIFIED_DIR)) if os.path.isfile(os.path.join(str(EEIFIED_DIR), name))]
if len(files) == 0:
    temp_dir = os.listdir(str(EEIFIED_DIR))[0]
    EEIFIED_DIR = EEIFIED_DIR / temp_dir

# copy the eeified to the google bucket
GS_EEIFIED_PATH = f'gs://{BUCKET}/{EEIFIED_DIR_NAME}'
copy = f'gsutil -m cp -R {EEIFIED_DIR} {GS_EEIFIED_PATH}'
result = subprocess.check_output(copy, shell=True)
print(result)

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
