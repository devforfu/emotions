import math
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from IPython.display import Image, display
import pandas as pd
import numpy as np

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from emotions import FER2013Dataset, load, load_all
from utils import plot_metrics, plot_roc_curve
from settings import DATA_FOLDER


TARGET_NAMES = FER2013Dataset.VERBOSE_EMOTION
N_CLASSES = len(TARGET_NAMES)
SIZE = FER2013Dataset.IMAGE_SIDE
N_PIXELS = FER2013Dataset.IMAGE_SIZE


def data_augmentation():
    return ImageDataGenerator(
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        featurewise_center=True,
        featurewise_std_normalization=True,
        zca_whitening=True)


def step_decay(base=0.1, drop=0.5, epochs_drop=10):
    """Learning rate step-wise decay schedule."""

    return lambda epoch: base * math.pow(
        drop, math.floor((1 + epoch)/epochs_drop))


def main():
    X, y = load_all(DATA_FOLDER, 'emotions')
    X = X.reshape(X.shape[0], 1, SIZE, SIZE).astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    y_hot_train = np_utils.to_categorical(y_train)

    kfold = StratifiedKFold(n_splits=10, random_state=1)
    train, val = next(kfold.split(X_train, y_train))
    train_gen, val_gen = data_augmentation(), data_augmentation()
    train_gen.fit(X_train[train])
    val_gen.fit(X_train[val])

    model = Sequential()
    model.add(Convolution2D(96, 5, 5,
                            input_shape=(1, SIZE, SIZE),
                            activation='relu',
                            ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(N_CLASSES, activation='softmax'))

    nb_epoch = 20
    template = "model_{epoch:02d}_[val={val_loss:.2f}].hdf5"
    checkpoint_file = os.path.join(DATA_FOLDER, 'checkpoints', template)
    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(checkpoint_file, save_best_only=True),
        LearningRateScheduler(step_decay())
    ]
    sgd = SGD(lr=0.0, decay=0.0, nesterov=True)

    compile_params = {
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy',
                    'categorical_accuracy',
                    'top_k_categorical_accuracy',
                    'fmeasure'],
        'optimizer': sgd
    }

    model.compile(**compile_params)
    h = model.fit_generator(
        train_gen.flow(X_train[train], y_hot_train[train], batch_size=32),
        samples_per_epoch=1000,
        validation_data=val_gen.flow(X_train[val], y_hot_train[val]),
        nb_val_samples=300,
        callbacks=callbacks,
        nb_epoch=nb_epoch,
        verbose=1)

    y_pred = model.predict_classes(X_test)
    target_names = FER2013Dataset.VERBOSE_EMOTION
    print(classification_report(y_test, y_pred, target_names=target_names))

    scores = pd.DataFrame(h.history)
    fig = plot_metrics(model.metrics_names, scores, 'CNN from paper')
    fig.savefig('cnn_paper.png', format='png')
    np.save('cnn_paper_y_pred', y_pred)
    np.save('cnn_paper_y_test', y_test)


if __name__ == '__main__':
    main()
