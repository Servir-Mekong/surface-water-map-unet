import glob
from pathlib impor Path
from functools import partial
from model import dataio, model
import kerastuner as kt

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

# partial function to pass keywords to for hyperparameter tuning
modelBuilder = partial(model.getModel,inShape=inShape, outClasses=outClasses)

# function to build model for hp tuning
# accepts hp tuning instance to select variables
def buildTuner(hp):
    with strategy.scope():
        # build the model with parameters
        myModel = modelBuilder(
                rate = hp.Float('rate',min_value=0.1,max_value=0.5,step=0.05),
                noise = hp.Float('noise',min_value=0.2,max_value=2.0,step=0.2),
                activation = hp.Choice('activation',values=['relu','elu']),
                combo = hp.Choice('combo',values=['add','concat']),
            )

        # compile model
        myModel.compile(optimizer = keras.optimizers.Adam(
                        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss = bce_dice_loss,
            metrics = [keras.metrics.categorical_accuracy,
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                dice_coef
            ]
        )
    return myModel



tuner = kt.RandomSearch(
    buildTuner,
    objective=kt.Objective("val_recall", direction="max"), # adjust objective to what is important for model metrics
    max_trials=10,
    executions_per_trial=2,
    seed =0,
    directory='hyperopt',
    project_name='ls_qa')

# set early stopping to prevent long running models
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto',restore_best_weights=True)

# fit the model
tuner.search(
        x = training,
        epochs=EPOCHS,
        steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
        validation_data=testing,
        validation_steps=TEST_SIZE,
        callbacks=[earlyStop]
)


tuner.results_summary()
