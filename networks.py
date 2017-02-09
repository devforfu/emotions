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
        return self.__class__.__name__

    def build(self, **params):
        raise NotImplementedError()

    def show_structure(self, filename=None):
        if not filename:
            filename = self.name + '.png'
        plot(self.model, to_file=filename)


@deep_model('trivial')
class DummyModel(DeepModel):

    def build(self, **params):
        model = Sequential()
        model.add(Dense(self.n_pixels, input_dim=self.n_pixels, init='normal'))
        model.add(Activation('relu'))
        model.add(Dense(self.n_classes, input_dim=self.n_pixels, init='normal'))
        model.add(Activation('softmax'))
        self.model = model
        return model


@deep_model('simple')
class SimpleFeedforwardModel(DeepModel):

    def build(self, init='normal', activation='relu',
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


@deep_model('dropout_const')
class DropoutFeedforwardModel(DeepModel):

    def build(self, init='normal', activation='relu',
              output_activation='sigmoid', dropout=0.2):

        model = Sequential()
        model.add(Dense(self.n_pixels,
                        input_dim=self.n_pixels,
                        activation=activation,
                        init=init))

        one_third_more = int(self.n_pixels * 1.3)
        twice_more = self.n_pixels * 2

        model.add(Dense(one_third_more, init=init, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(one_third_more, init=init, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(twice_more, init=init, activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(self.n_classes, init=init))
        model.add(Activation(output_activation))
        self.model = model

        return model


