import unittest
import numpy as np
import tensorflow as tf

import preprocess
from data_windowing import WindowGenerator
from Baseline import Baseline
from MLmodel import Encoder, GRUNetwork, BahdanauAttention, CrossAttention


class TestMLModel(unittest.TestCase):

    def setUp(self) -> None:
        self.filename = "data/Solar_Wind_Dst_1997_2016_shifted_forward.csv"
        self.train_col = ["Bx", "By", "Bz", "Sv", "Den", "Dst"]

        train_ratio = 0.7
        val_ratio = 0.2

        input_width = 30
        label_width = 30
        shift = 1
        label_columns = ["Dst"]

        trainALL, valALL, testALL = preprocess.split_train_test(self.filename, train_ratio, val_ratio, self.train_col)

        self.traindf, self.valdf, self.testdf = preprocess.normalize(trainALL, valALL, testALL)



        self.wg = WindowGenerator(input_width, label_width, shift,
                             self.traindf, self.valdf, self.testdf, label_columns)

    def test_BaselineModel_compile(self):

        col_indcs = tf.constant(self.wg.label_indices, dtype=np.int64)

        baseline = Baseline()
        baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

        val_performance = {}
        performance = {}

        val_performance['Baseline'] = baseline.evaluate(self.wg.validation)
        performance['Baseline'] = baseline.evaluate(self.wg.test, verbose = 0)


    def test_GRULayer(self):

        inputs = tf.random.normal([32, 10, 8])
        gru = tf.keras.layers.GRU(4, return_state=True)

        whole_sequence_output, final_state = gru(inputs)
        print(whole_sequence_output)
        print(whole_sequence_output.shape)
        print(final_state.shape)

    def test_Bidirectional(self):

        #inputs = tf.random.normal([32, 10, 8])
        biGRU = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(32,
                                      # Return the sequence and state

                                      return_sequences=True,
                                      recurrent_initializer='glorot_uniform'))

        for inputs, labels in self.wg.train.take(1):
            whole_sequence_output = biGRU(inputs)

            print(whole_sequence_output.shape)


    def test_Encoder(self):

        encoder = Encoder(32)
        for (batch_n, (inputs, labels)) in enumerate(self.wg.train.take(1)):
            encode_outputs = encoder(inputs)


            print(encode_outputs.shape)



    def test_BahdanauAttention(self):
        encoder = Encoder(32)
        att = BahdanauAttention(32)

        for inputs, labels in self.wg.train.take(1):
            encode_outputs, states = encoder(inputs)

            context_vector, weights = att(states, encode_outputs)

            print(f"enc_output shape: {encode_outputs.shape}")
            print(f"context_vector shape: {context_vector.shape}")

    def test_CrossArrention(self):

        att = CrossAttention(32)

        embed = tf.keras.layers.Embedding(50,
                                          output_dim=6, mask_zero=True)
        encoder = Encoder(32)

        for inputs, labels in self.wg.train.take(1):
            #inputs_embed = embed(inputs)
            enc_output = encoder(inputs)
            att_output = att(inputs, enc_output)


            print(f"Attention Output shape: {att_output.shape}")







    def test_ModelBuild(self):

        encoder = Encoder(32)
        model = GRUNetwork(32)

        for (batch_n, (inputs, labels)) in enumerate(self.wg.train.take(1)):
            encode_outputs, states = encoder(inputs)
            pred = model(labels, states, encode_outputs)

        print(model.summary())










