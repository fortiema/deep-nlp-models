import numpy as np


def load_pretrained_embeddings(fname, word_index):
    """Loads pre-trained GloVe model
    """
    embeddings_index = {}
    embedding_dim = None
    with open(fname) as fin:
        for i, line in enumerate(fin):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if not i:
                embedding_dim = len(values[1:])
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
