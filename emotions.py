import os

import numpy as np
import pandas as pd

from basedir import data


VERBOSE_EMOTION = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

IMAGE_SIDE = 48
IMAGE_SIZE = IMAGE_SIDE ** 2


def prepare_data(path):

    def to_pixels(row):
        return np.array([float(px) for px in row.split()])

    emotions_data = pd.read_csv(path)
    labels = emotions_data['emotion'].apply(int)
    labels.name = 'label'
    verbose = labels.apply(lambda emotion_label: VERBOSE_EMOTION[emotion_label])
    verbose.name = 'verbose'
    usage = emotions_data.Usage.map({
        'Training': 'train',
        'PublicTest': 'valid',
        'PrivateTest': 'test'
    })
    usage.name = 'subset'

    flat_images = pd.DataFrame(
        np.r_[[to_pixels(row) for row in emotions_data.pixels.values]],
        columns=['p' + str(i) for i in range(1, IMAGE_SIZE + 1)])

    processed_data = pd.DataFrame(
        pd.concat([labels, verbose, usage, flat_images], axis=1))
    prepared_path = os.path.join(os.path.dirname(path), 'emotions.dat')
    processed_data.to_pickle(prepared_path)

    return processed_data


def to_image(pixels, size=48):
    """Converts flat image representation into (size, size) shape."""

    if isinstance(pixels, str):
        pixels = pixels.split()
    flat = np.array([float(px) for px in pixels])
    return flat.reshape((size, size))


def main():
    dataset_path = data('fer2013', 'emotions.dat')
    if not dataset_path:
        dataset_path = data('fer2013', 'fer2013.csv')
        dataset = prepare_data(dataset_path)
    else:
        dataset = pd.read_pickle(dataset_path)
    print("Emotions dataset:")
    print(dataset.head())
    print("Done!")


if __name__ == '__main__':
    main()
