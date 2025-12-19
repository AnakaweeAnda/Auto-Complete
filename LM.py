import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class ModelLM :
    def __init__(self):
        self.vectorizer = None
        self.model = None
    def tokenize_text(self,texts,max_tokens=10000,sequence_length=50) :
        text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                            output_sequence_length=sequence_length,
                                                            standardize=None
                                                            )
        text_vectorizer.adapt(texts)
        self.vectorizer = text_vectorizer
        return self.vectorizer(texts)
    def create_dataset(self,sequences,batch_size=64) :
        X = sequences[:,:-1]
        y = sequences[:,-1]

        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    def build_model(self,sequence_length,embedding_dims=256,lstm_units=128) :
        vocab_size = self.vectorizer.vocabulary_size()

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,embedding_dims,input_length=sequence_length-1),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(vocab_size,activation='softmax')
        ])

        model.compile(
            loss='sparse_categorial_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model = model
        return model
    def predict(self,texts,num_words=5) :
        result = texts
        for _ in range(num_words) :
            sequence = self.vectorizer([texts])
            pred = self.model.predict(sequence)
            pred_idx = np.argmax(pred[0])
            vocab = self.vectorizer.get_vocabulary()
            pred_word = vocab[pred_idx]
            result + ' ' + pred_word
        return result



