# -*- coding: utf-8 -*-
import logging
import os

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Activation, concatenate, Dense, Dropout, Input, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.models import Model, model_from_json
from keras.optimizers import Adadelta

from deep_nlp_models.utils.embeddings import load_pretrained_embeddings


class CNNYKim2014Model:
    """
    Convolutional Neural Networks for Sentence Classification
    Y. Kim - 2014

    http://www.aclweb.org/anthology/D14-1181
    """

    def __init__(self, word_index, embed_fname, max_seq=1024, nb_labels=None):
        self.logger = logging.getLogger(CNNYKim2014Model.__name__)

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

        optimizer = Adadelta(lr=0.01, clipnorm=3.0, decay=1e-5)

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['acc'])

    def save(self, dirname):
        model_json = self.model.to_json()
        with open(os.path.join(dirname, 'model.json'), 'w') as fout:
            fout.write(model_json)
        self.model.save_weights(os.path.join(dirname, 'weights.h5'))

    def load(self, dirname):
        model_file = open(os.path.join(dirname, 'model.json'), 'r')
        model_json = model_file.read()
        model_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights(os.path.join(dirname, 'weights.h5'))

        optimizer = Adadelta(lr=0.01, clipnorm=3.0, decay=1e-5)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['acc'])

    def train(self, x, y, epochs=10, batch_size=8, early_stop=True):
        board = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=50,
                            write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None)

        early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=1)

        self.model.fit(
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

        self.model.fit_generator(
            gen, 
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[board, early_stop])
    
    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def evaluate_gen(self, gen, steps=1000):
        return self.model.evaluate_generator(gen, steps)

    def predict(self, x):
        return self.model.predict(x)
