import glob
from pathlib impor Path
from model import dataio, model
import tensorflow as tf


# specify directory as data io info
BASEDIR = Path('/data/kmarkert/ls_cloud_qa/')
DATADIR = BASEDIR / 'training_data'
TRAINING = 'training_patches'
TESTING = 'testing_patches'
EVAL = 'val_patches'

# specify some data structure
FEATURES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
LABLES = ['cloud', 'shadow', 'snow', 'water', 'land', 'nodata']

# patch size for training
KERNEL_SIZE = 256
PATCH_SHAPE = (KERNEL_SIZE, KERNEL_SIZE)

# Sizes of the training and evaluation datasets.
TRAIN_SIZE = 12672
TEST_SIZE = 5030
VAL_SIZE = 0  # really the validation data...

# Specify model training parameters.
BATCH_SIZE = 64
EPOCHS = 50
BUFFER_SIZE = 12750

# get list of files for training, testing and eval
trainingFiles = glob.glob(str(DATADIR / TRAINING) + '*')
testingFiles = glob.glob(str(DATADIR / TESTING) + '*')
evalFiles = glob.glob(str(DATADIR / EVAL) + '*')

# get training, testing, and eval TFRecordDataset
# training is batched, shuffled, transformed, and repeated
training = dataio.get_dataset(trainingFiles, FEATURES, LABLES, PATCH_SHAPE,
                       BATCH_SIZE, bufferSize=BUFFER_SIZE, training=True).repeat()
# testing is batched by 1 and repeated
testing = dataio.get_dataset(testingFiles, FEATURES,
                      LABLES, PATCH_SHAPE, 1).repeat()
# eval is batched by 1
eval = dataio.get_dataset(evalFiles, FEATURES,
                      LABLES, PATCH_SHAPE, 1)

# get distributed strategy and apply distribute i/o and model build
strategy = tf.distribute.MirroredStrategy()

# define tensor inputshape and number of classes
inShape = PATCH_SHAPE + (len(FEATURES))
outClasses = len(LABELS)

# build the model and compile
myModel = model.build(inShape, outClasses, distributedStrategy=strategy)

# define callbacks during training
modelCheckpnt = tf.keras.callbacks.ModelCheckpoint(
    'bestModelWeights.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1, save_weights_only=True)
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs', write_images=True)

# fit the model
history = myModel.fit(
    x=training,
    epochs=EPOCHS,
    steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
    validation_data=testing,
    validation_steps=TEST_SIZE,
    callbacks=[modelCheckpnt, tensorboard, earlyStop],
)

# check how the model trained
myModel.evaluate(eval)
