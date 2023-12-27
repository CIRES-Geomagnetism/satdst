import os
import time
from collections import deque
import pandas as pd

import tensorflow as tf
from MLmodel import Encoder, Decoder, GRUNetwork


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

def save_weights(model, checkpoint_dir, epoch):

    # Directory where the checkpoints will be saved
    checkpoint_dir = checkpoint_dir
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    model.save_weights(checkpoint_prefix.format(epoch=epoch))


def gradientTape_train(inputs, labels, model, loss_fn, optimizer):



    with tf.GradientTape() as tape:


        logits = model(inputs, labels)
        loss = loss_fn(labels, logits)


    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def valid_model(model, validset, loss_fn):
    valid_loss = tf.metrics.Mean()

    for (batch_n, (inputs, labels)) in enumerate(validset):
        logits = model(inputs, labels)
        loss = loss_fn(labels, logits)

        valid_loss.update_state(loss)

    return valid_loss

def early_stopping(loss_history, delta):

    last_loss = loss_history.popleft()
    min_loss = min(loss_history)
    ratio = (min_loss - last_loss)/min_loss

    if ratio < delta:
        return True
    else:
        return False




def train(epochs, dataset, optimizer, checkpoint_dir):

    patience = 3
    delta = 0.001

    trainset = dataset.train
    validset = dataset.validation

    train_loss = tf.metrics.Mean()
    loss_fn = tf.keras.losses.MeanSquaredError()

    model= GRUNetwork(32)
    loss_history = deque(maxlen=patience+1)



    for epoch in range(epochs):

        start = time.time()

        train_loss.reset_states()

        for (batch_n, (inputs, labels)) in enumerate(trainset):

            loss = gradientTape_train(inputs, labels, model, loss_fn, optimizer)
            train_loss.update_state(loss)

            if batch_n % 100 == 0:
                print(f"Epoch {epoch+1} Batch {batch_n} Loss {loss:.4f}")

            if epoch % 5 == 0:
                save_weights(model, checkpoint_dir, epoch)


        valid_loss = valid_model(model, validset, loss_fn)
        loss_history.append(valid_loss)

        if len(loss_history) > patience:
            if early_stopping(loss_history, delta):
                print(f"Early Stopping. No improvement of more than {delta:.5%} in validation loss in the last {patience} epochs")
                break

        print(f"Epoch {epoch + 1} Loss: {valid_loss.result():.4f}")
        print(f"Time taken from 1 epoch {time.time() - start:.2f} sec")
        print("_"*50)
        save_weights(model, checkpoint_dir, epoch)

def compile_and_fit(model, dataset, epochs):

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=3,
                                                      mode='min')

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])


    history = model.fit(dataset.train.repeat(), epochs=epochs, steps_per_epoch = 100, validation_data=dataset.validation, callbacks=[early_stopping], batch_size=30)

    return history

def evaluate(model: tf.keras.Model, inputs, true_df: pd.DataFrame):


    predictions = model(inputs)

    true_dst = true_df["DST"].values






#def loss_fn():
    # L2 loss





