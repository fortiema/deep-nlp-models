from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class TextPreprocessor:
    """Basic text preprocessor

    Basic tokenizer, retains only top N words in its vocabulary.
    Sequences smaller than the target length are padded.
    """

    def __init__(self, vocab_size=100000, char_level=False):
        if char_level:
            self.tokenizer = Tokenizer(vocab_size, filters='', char_level=char_level)
        else:
            self.tokenizer = Tokenizer(vocab_size, char_level=char_level)
        self.word_index = None

    def fit_dict(self, documents):
        self.tokenizer.fit_on_texts(documents)
        self.word_index = self.tokenizer.word_index

    def _prep_text(self, text, seq_len=1024):
        return pad_sequences(
            self.tokenizer.texts_to_sequences(text),
            padding='post', truncating='post',
            maxlen=seq_len)

    def get_docs_gen(self, source, seq_len=1024):
        if not self.word_index:
            raise Exception('Need to instantiate a dictionary by calling "fit_dict" first')

        yield from (self._prep_text([s], seq_len=seq_len) for s in source)

    def get_docs(self, source, seq_len=1024):
        if not self.word_index:
            raise Exception('Need to instantiate a dictionary by calling "fit_dict" first')

        return np.asarray([self._prep_text([s], seq_len=seq_len).flatten() for s in source])
