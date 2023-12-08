import tensorflow as tf
from MLmodel import Encoder, Decoder


########### ML Model ###############################
#input_seq_len = 30
#output_seq_len = 1
#batch_size = 1
#learning_rate = 0.00005
#lambda_l2_reg = 0.0003
#total_iterations = 11000
#hidden_dim = 200
#num_stacked_layers = 1
#GRADIENT_CLIPPING = 0.5
######################################################




class GRUNetwork():

    def __init__(self, units, optimizer):

        self.encoder = Encoder(units)
        self.decoder = Decoder(units)

        self.optimizer = optimizer

    @tf.function
    def train_step(self, inputs, labels):

        with tf.GradientTape as tape:

            enc_output = self.encoder(inputs)
            logits = self.decoder(labels, enc_output)

            loss = self.loss_function(labels, logits)




        trained_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, trained_variables)
        self.optimizer.apply_gradients(zip(grads, trained_variables))


    def loss_function(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        loss = loss_fn(y_true, y_pred)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, loss.dtype)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


