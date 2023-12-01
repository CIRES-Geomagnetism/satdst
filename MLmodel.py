import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_units):

        super(Encoder, self).__init__()

        self.gru = tf.keras.layers.GRU(rnn_units,
                                      # Return the sequence and state
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
    def call(self, inputs):

        x = inputs


        encode_outputs, states = self.gru(x)

        return encode_outputs, states

class GRUNetwork(tf.keras.Model):
    def __init__(self, rnn_units):
        super().__init__(self)

        encoder = Encoder(rnn_units)

        self.encoder = encoder

        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, trainSet, encode_states):

        inputs, labels = trainSet
        encode_inputs, encode_states = self.encoder(inputs)

        x = labels
        x, states = self.gru(x, initial_states=encode_states)

        output = self.dense(x)

        return output




















