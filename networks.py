from keras.layers import Dense, Activation, Dropout
from keras.utils.visualize_util import plot
from keras.models import Sequential

from emotions import FER2013Dataset


_deep_models = {}


def deep_model(model_name):
    def wrapper(cls):
        _deep_models[model_name] = cls
        return cls
    return wrapper


def get_model(model_name):
    if model_name not in _deep_models:
        available_models = ", ".join(_deep_models.keys())
        raise ValueError(
            "Model '%s' not found. Available models are: %s"
            % (model_name, available_models))
    return _deep_models[model_name]


def init_model(name, *args, **kwargs):
    return get_model(name)(*args, **kwargs)


class DeepModel:

    image_size = 48
    n_pixels = image_size ** 2
    n_classes = len(FER2013Dataset.VERBOSE_EMOTION)

    def __init__(self, *args, **kwargs):
        self.model = None

    @property
    def name(self):
        return 'Deep Model'

    def build(self, **params):
        raise NotImplementedError()

    def show_structure(self, filename=None):
        if not filename:
            filename = self.name + '.png'
        plot(self.model, to_file=filename)


@deep_model('simple')
class SimpleFeedforwardModel(DeepModel):

    def build(self, init='normal', optimizer='adam', activation='relu',
              output_activation='sigmoid'):

        model = Sequential()
        model.add(Dense(self.n_pixels, input_dim=self.n_pixels, init=init))
        model.add(Activation(activation))
        model.add(Dense(self.n_pixels * 2, init=init))
        model.add(Activation(activation))
        model.add(Dense(self.n_classes, init=init))
        model.add(Activation(output_activation))
        self.model = model

        return model


@deep_model('dropout')
class DropoutFeedforwardModel(DeepModel):

    def build(self, init='normal', optimizer='adam', activation='relu',
              output_activation='sigmoid', dropout=0.2):

        model = Sequential()
        model.add(Dense(self.n_pixels, input_dim=self.n_pixels, init=init))
        model.add(Activation(activation))
        model.add(Dense(self.n_pixels * 2, init=init))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        model.add(Dense(self.n_pixels * 4, init=init))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        model.add(Dense(self.n_classes, init=init))
        model.add(Activation(output_activation))
        self.model = model

        return model
