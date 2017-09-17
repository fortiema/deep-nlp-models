import abc
from abc import abstractmethod
import os

from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model, model_from_json
from keras.optimizers import Adadelta


class ClassificationModel(abc.ABC):
    """Base class for classification task model
    """

    def __init__(self, nb_labels=2):
        self._model = None
        self._nb_labels = nb_labels

    def save(self, dirname):
        model_json = self._model.to_json()
        with open(os.path.join(dirname, 'model.json'), 'w') as fout:
            fout.write(model_json)
        self._model.save_weights(os.path.join(dirname, 'weights.h5'))

    def load(self, dirname, label):
        model_file = open(os.path.join(dirname, 'model.json'), 'r')
        model_json = model_file.read()
        model_file.close()
        self._model = model_from_json(model_json)
        self._model.load_weights(os.path.join(dirname, 'weights.h5'))

    @abstractmethod
    def compile(self):
        pass

    def train(self, x, y, epochs=10, batch_size=8, early_stop=True):
        board = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=50,
                            write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)

        early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=1)

        self._model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[board, early_stop])

    def train_gen(self, gen, epochs=20, steps_per_epoch=10000, early_stop=True):
        board = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=50,
                            write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)

        early_stop = EarlyStopping(monitor='loss', min_delta=0.0, patience=1)

        self._model.fit_generator(
            gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[board, early_stop])

    def evaluate(self, x, y):
        return self._model.evaluate(x, y)

    def evaluate_gen(self, gen, steps=1000):
        return self._model.evaluate_generator(gen, steps)

    def predict(self, x):
        return self._model.predict(x)
