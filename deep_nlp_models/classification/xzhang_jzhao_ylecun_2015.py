# -*- coding: utf-8 -*-
import logging

from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import plot_model

from deep_nlp_models.classification.models import ClassificationModel


class XZhangJZhaoYLecun2015Model(ClassificationModel):
    """
    Character-level Convolutional Networks for Text Classification
    X. Zhang, J. Zhao, Y. LeCun - 2015

    'Small Features' Variant

    https://arxiv.org/abs/1509.01626
    """

    def __init__(self, alphabet_size, max_seq=1014, nb_labels=None):
        super().__init__()
        self.logger = logging.getLogger(XZhangJZhaoYLecun2015Model.__name__)

        imput_layer = Input(shape=(max_seq,), dtype='int32')

        # trick to simulate a one-hot encoding step using the embedding layer
        embed = Embedding(
            input_dim=alphabet_size,
            output_dim=alphabet_size-1,
            input_length=max_seq,
            init='uniform',
            trainable=False)(imput_layer)

        conv1 = Conv1D(256, 7, padding='valid', activation='relu')(embed)
        conv1 = MaxPooling1D(pool_size=3, strides=3)(conv1)

        conv2 = Conv1D(256, 7, padding='valid', activation='relu')(conv1)
        conv2 = MaxPooling1D(pool_size=3, strides=3)(conv2)

        conv3 = Conv1D(256, 3, padding='valid', activation='relu')(conv2)

        conv4 = Conv1D(256, 3, padding='valid', activation='relu')(conv3)

        conv5 = Conv1D(256, 3, padding='valid', activation='relu')(conv4)

        conv6 = Conv1D(256, 3, padding='valid', activation='relu')(conv5)
        conv6 = MaxPooling1D(pool_size=3, strides=3)(conv6)

        flat = Flatten()(conv6)

        dense1 = Dense(1024, activation='relu')(flat)
        dense1 = Dropout(0.5)(dense1)

        dense2 = Dense(1024, activation='relu')(dense1)
        dense2 = Dropout(0.5)(dense2)

        out = Dense(nb_labels, activation='softmax')(dense2)

        self._model = Model(inputs=imput_layer, outputs=out)

        self.compile()

    def compile(self):
        optimizer = Adadelta(lr=0.01, decay=1e-5)

        self._model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['acc'])
