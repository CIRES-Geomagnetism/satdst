import time

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
        self.loss = tf.keras.losses.MeanSquaredError()


    def train_step(self, inputs, labels):

        with tf.GradientTape() as tape:

            enc_output = self.encoder(inputs)
            logits = self.decoder(labels, enc_output)



            loss = self.loss(labels, logits)

        trained_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, trained_variables)
        self.optimizer.apply_gradients(zip(grads, trained_variables))

        return loss


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

    def train(self, epochs, trainset):



        for epoch in range(epochs):

            start = time.time()

            batch_loss = 0


            for (batch_n, (inputs, labels)) in enumerate(trainset):

                loss = self.train_step(inputs, labels)




                if batch_n % 100 == 0:
                    print(f"Epoch {epoch+1} Batch {batch_n} Loss {loss:.4f}")
            print()
            #print(f"Epoch {epoch+1} Loss: {mean.results.numpy():.4f}")
            print(f"Time taken from 1 epoch {time.time() - start:.2f} sec")
            print("_"*50)





