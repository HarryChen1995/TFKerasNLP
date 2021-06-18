import numpy
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
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

test_sequences = tockenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences( test_sequences, maxlen = maxlength, truncating="post")

reverse_word_index = dict([(val, key) for key, val in tockenizer.word_index.items()])
def decode(text):
    return " ".join([reverse_word_index.get(c, "?") for c in text])




print(decode(padded[0]))
print(train_sentences[0])



model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocal_size, embded_dim, input_length=maxlength),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences= True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss = "binary_crossentropy", optimizer= "adam", metrics=["acc"])
model.summary()
history = model.fit(padded, train_labels, epochs= 50, validation_data=(test_padded, test_labels))
model.save("NLP_Model")
