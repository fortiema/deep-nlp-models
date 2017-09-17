# -*- coding: utf-8 -*-
import logging
import os

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Activation, concatenate, Dense, Dropout, Input, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.models import Model, model_from_json
from keras.optimizers import Adadelta

from deep_nlp_models.classification.models import ClassificationModel
from deep_nlp_models.utils.embeddings import load_pretrained_embeddings


class YKim2014Model(ClassificationModel):
    """
    Convolutional Neural Networks for Sentence Classification
    Y. Kim - 2014

    Uses basic convolutions with varying filter size to capture localized semantics.

    http://www.aclweb.org/anthology/D14-1181
    """

    def __init__(self, word_index, embed_fname, max_seq=1024, nb_labels=None):
        self.logger = logging.getLogger(YKim2014Model.__name__)

        self.logger.debug('Populating embedding layer with pre-trained gloVe vectors...')
        embed_matrix = load_pretrained_embeddings(embed_fname, word_index)
        self.logger.debug('Success - Dimentionality: {}'.format(embed_matrix.shape[1]))

        imput_layer = Input(shape=(max_seq,))

        embedding_layer = Embedding(
            input_dim=len(word_index) + 1,
            output_dim=embed_matrix.shape[1],
            weights=[embed_matrix],
            input_length=max_seq,
            trainable=False)(imput_layer)

        conv1 = Conv1D(100, 3, padding='valid', activation='relu')(embedding_layer)
        conv1 = GlobalMaxPooling1D()(conv1)

        conv2 = Conv1D(100, 4, padding='valid', activation='relu')(embedding_layer)
        conv2 = GlobalMaxPooling1D()(conv2)

        conv3 = Conv1D(100, 5, padding='valid', activation='relu')(embedding_layer)
        conv3 = GlobalMaxPooling1D()(conv3)

        out = concatenate([conv1, conv2, conv3])
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(nb_labels, activation='softmax')(out)

        self.model = Model(inputs=imput_layer, outputs=out)

        self.compile()

    def compile(self):
        optimizer = Adadelta(lr=0.01, clipnorm=3.0, decay=1e-5)

        self._model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['acc'])
