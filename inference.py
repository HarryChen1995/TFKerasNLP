import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
import numpy
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
imdb, info = tfds.load("imdb_reviews", with_info= True, as_supervised= True)

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for s, l in imdb["train"]:
    train_sentences.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s, l in imdb["test"]:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

train_labels = numpy.array(train_labels)
test_labels = numpy.array(test_labels)

vocal_size = 10000
embded_dim = 16
maxlength = 120
ovv_tok = "<OOV>"

from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tockenizer = Tokenizer(num_words=vocal_size, oov_token=ovv_tok)
tockenizer.fit_on_texts(train_sentences)
sequence = tockenizer.texts_to_sequences(train_sentences)
padded = pad_sequences( sequence, maxlen=maxlength, truncating="post")

test_sentences = ["this movie is bac"]
test_sentences = tockenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sentences, maxlen=maxlength, truncating="post")
model = tf.keras.models.load_model("NLP_Model")
pred = model.predict(test_padded)
print(pred)