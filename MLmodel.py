import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units):

        super(Encoder, self).__init__()

        self.gru = tf.keras.layers.GRU(units,
                                      # Return the sequence and state
                                      return_state=True,
                                      return_sequences=True,
                                      recurrent_initializer='glorot_uniform')
    def call(self, inputs, states=None, training=False):

        x = inputs

        if states == None:
            states = self.gru.get_initial_state(x)

        x, states = self.gru(x, initial_states=states, training=training)

        return x, states
















