import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_units):

        super(Encoder, self).__init__()

        self.gru = tf.keras.layers.GRU(rnn_units,
                                      # Return the sequence and state
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(rnn_units,
                                      # Return the sequence and state
                                      return_sequences=True,
                                      recurrent_initializer='glorot_uniform'))
    def call(self, inputs):

        x = inputs

        #output, state = self.gru(x)
        encode_outputs = self.rnn(x)

        return encode_outputs

class CrossAttention(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):

        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def __call__(self, x, context):

        att_output, att_weights = self.mha(query=x, value=context, return_attention_scores=True)

        x = self.add([x, att_output])
        x = self.layernorm(x)

        return x

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values):

        #We are doing this to broadcast addition along the time axis to calcualte the score
        #query_with_time_axis = tf.expand_dims(query, 1)



        score = self.V(tf.nn.tanh(self.W1(query)) + self.W2(values))

        weights = tf.nn.softmax(score, axis = 1)

        context_vector = weights*values

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, weights


class GRUNetwork(tf.keras.Model):
    def __init__(self, rnn_units):
        super().__init__(self)

        encoder = Encoder(rnn_units)

        self.encoder = encoder

        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.attention = BahdanauAttention(rnn_units)
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, x, enc_states, enc_output):

        context_vector, att_weights = self.attention(enc_states, enc_output)

        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(context_vector)

        #output = tf.reshape(output, (-1, output.shape[2]))




        x = self.dense(output)

        return x, state, att_weights




















