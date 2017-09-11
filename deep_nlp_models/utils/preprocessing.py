from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class TextPreprocessor():
    """Basic text preprocessor

    Basic tokenizer, retains only top N words in its vocabulary.
    Sequences smaller than the target length are padded.
    """

    def __init__(self, vocab_size=100000):
        self.tokenizer = Tokenizer(vocab_size)
        self.word_index = None

    def fit_dict(self, documents):
        self.tokenizer.fit_on_texts(documents)
        self.word_index = self.tokenizer.word_index

    def get_docs_gen(self, source, seq_len=1024):
        if not self.word_index:
            raise Exception('Need to instantiate a dictionary by calling "fit_dict" first')

        def _prep_text(text):
            return pad_sequences(self.tokenizer.texts_to_sequences(text),
                                 maxlen=seq_len)

        yield from (_prep_text([s]) for s in source)

    def get_docs(self, source, seq_len=1024):
        if not self.word_index:
            raise Exception('Need to instantiate a dictionary by calling "fit_dict" first')

        def _prep_text(text):
            return pad_sequences(self.tokenizer.texts_to_sequences(text),
                                 maxlen=seq_len)

        return np.asarray([_prep_text([s]).flatten() for s in source])
