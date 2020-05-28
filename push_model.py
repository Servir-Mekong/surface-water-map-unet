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
MODEL_DIR = OUTPUT_DIR / 'model_push'
EEIFIED_DIR = MODEL_DIR / 'eeified'

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

ai_model_create = f'gcloud ai-platform models create {MODEL_NAME} --project {PROJECT}'
result = subprocess.check_output(ai_model_create, shell=True)
print(result)

# provide staging-bucket if pushing from local directory
# be careful with the python version
ai_version_create = f'gcloud ai-platform versions create {VERSION_NAME} --project {PROJECT} --model {MODEL_NAME} ' \
                    f'--origin {str(EEIFIED_DIR)} --staging-bucket gs://{BUCKET} --runtime-version=2.0 ' \
                    f'--framework "TENSORFLOW" --python-version=3.6'
result = subprocess.check_output(ai_version_create, shell=True)
print(result)
