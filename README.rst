===============
deep-nlp-models
===============

Various Keras DeepNet implementations for NLP tasks.

* Free software: MIT license
* Documentation: https://deep-nlp-models.readthedocs.io.


Models
------

Classification
^^^^^^^^^^^^^^
* **ykim2014** - *Convolutional Neural Networks for Sentence Classification*, Y. Kim (2014)


Remarks
-------

* All models were trained/tested using Tensorflow backend on a GTX1070.
* Some models make use of pretrained GloVe vectors, you can download those `here <https://nlp.stanford.edu/projects/glove/>`_.
 * You can store then in the ``models/glove``, otherwise you can pass the full path to the model when you instantiate it.