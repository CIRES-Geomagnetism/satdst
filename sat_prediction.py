## Predicting Disturbance-storm-time index using tensorflow-based algorithm
# Uses coefficients previously developed using ML training
# August 2018
# Manoj.C.Nair@Noaa.gov

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import copy
import os
import pickle as pickle
from datetime import timedelta, datetime
from scipy import signal

# Load session variables used in training
with open('./SatDst/session_variables.pkl', 'rb') as f:  
 input_seq_len, output_seq_len, batch_size , total_iterations , \
 hidden_dim , num_stacked_layers, learning_rate, lambda_l2_reg , \
 GRADIENT_CLIPPING, cols_to_train, cols_to_train_array, output_data , \
 TestYear , rms_error , mean_error, elapsed_time, \
 X_train_mean, X_train_std, y_mean, y_std = pickle.load(f)

# Load solarwind and Dst data
df = pd.read_csv('./SatDst/Solar_Wind_Dst_2016_extra_short.csv')
#print(df.head())

# plot data
#===============================================================================
# cols_to_plot = ["Dst", "Bx", "By", "Bz", "Sv", "Den"]
# i = 1
# #plot each column
# plt.figure(figsize = (10,12))
# for col in cols_to_plot:
#     plt.subplot(len(cols_to_plot), 1, i)
#     plt.plot(df["Decyear"],df[col])
#     plt.title(col, y=0.5, loc='left')
#     i += 1
# plt.show()
#===============================================================================

# Extract input data
X_test = df.loc[:, cols_to_train_array].values.copy()

# Extract output data
y_test = df['Dst'].values.copy().reshape(-1, 1)

# z-score transform of input data
for i in range(X_test.shape[1]):
    X_test[:, i] = (X_test[:, i] - X_train_mean[i]) / X_train_std[i]    

# z-score transform output data
y_test = (y_test - y_mean) / y_std


# convert decimal year to date
def convert_partial_year(number):

 year = np.asarray(number).astype(int)
 date = []
 for i in range(len(number)): 
  d = timedelta(days=(number[i] - year[i])* (365 ))# Leapyear not considered !!
  day_one = datetime(year[i],1,1)
  date.append(d + day_one)
 return date

# Generate data vector for RNN prediction
def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
    
    total_samples = x.shape[0]
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq


# Get the data vector and print the dimensions
test_x, test_y = generate_test_samples()
print(test_x.shape, test_y.shape)

# num of input signals
input_dim = X_test.shape[1]
# num of output signals
output_dim = y_test.shape[1]
# num of stacked lstm layers 

# Build ML graph
def build_graph(feed_previous = False):
    
    tf.reset_default_graph()
    
    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    weights = {
        'out': tf.get_variable('Weights_out', \
                               shape = [hidden_dim, output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', \
                               shape = [output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.constant_initializer(0.)),
    }
                                          
    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder. 
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the 
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'): 
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)
         
        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):
          """RNN decoder for the sequence-to-sequence model.
          Args:
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            initial_state: 2D Tensor with shape [batch_size x cell.state_size].
            cell: rnn_cell.RNNCell defining the cell function and size.
            loop_function: If not None, this function will be applied to the i-th output
              in order to generate the i+1-st input, and decoder_inputs will be ignored,
              except for the first element ("GO" symbol). This can be used for decoding,
              but also for training to emulate http://arxiv.org/abs/1506.03099.
              Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
              state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                 states can be the same. They are different for LSTM cells though.)
          """
          with variable_scope.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
              if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
              output, state = cell(inp, state)
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=dtypes.float32,
                              scope=None):
          """Basic RNN sequence-to-sequence model.
          This model first runs an RNN to encode encoder_inputs into a state vector,
          then runs decoder, initialized with the last encoder state, on decoder_inputs.
          Encoder and decoder use the same RNN cell type, but don't share parameters.
          Args:
            encoder_inputs: A list of 2D Tensors [batch_size x input_size].
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            feed_previous: Boolean; if True, only the first of decoder_inputs will be
              used (the "GO" symbol), all other inputs will be generated by the previous 
              decoder output using _loop_function below. If False, decoder_inputs are used 
              as given (the standard decoder case).
            dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
              state: The state of each decoder cell in the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
          """
          with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
          '''Naive implementation of loop function for _rnn_decoder. Transform prev from 
          dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
          used as decoder input of next time step '''
          return tf.matmul(prev, weights['out']) + biases['out']
        
        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp, 
            dec_inp, 
            cell, 
            feed_previous = feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]
        
    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)
        
    saver = tf.train.Saver
    
    return dict(
        enc_inp = enc_inp, 
        target_seq = target_seq, 
        train_op = optimizer, 
        loss=loss,
        saver = saver, 
        reshaped_outputs = reshaped_outputs,
        )
    
    



rnn_model = build_graph(feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    
    saver = rnn_model['saver']().restore(sess,  os.path.join('./SatDst/', 'multivariate_ts_Dst'))
    
    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    
    final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
    final_preds = np.concatenate(final_preds, axis = 1)
    

    
    print("Test RMS error is: ", np.sqrt(((final_preds * y_std - test_y * y_std) ** 2).mean()))
    print("Test Mean error is: ", np.mean((final_preds * y_std - test_y * y_std) ) )

#x_axis
date_x = df["Decyear"].values.copy()
date_x = convert_partial_year(date_x[0:X_test.shape[0]-input_seq_len-output_seq_len])

## remove duplicate hours and concatenate into one long array
test_y_expand = np.concatenate([test_y[i].reshape(-1) for i in range(0, test_y.shape[0], test_y.shape[1])], axis = 0)
final_preds_expand = np.concatenate([final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], final_preds.shape[1])], axis = 0)

original_rms = np.sqrt(((signal.detrend(test_y_expand)* y_std) ** 2).mean())
ml_rms = np.sqrt((( signal.detrend(final_preds_expand) * y_std - signal.detrend(test_y_expand)  * y_std) ** 2).mean())
    
print("Original RMS error is: ", original_rms)
print("ML RMS error is: ", ml_rms)


plt.figure(figsize=(12,4))
plt.plot_date(date_x,test_y_expand * y_std + y_mean,'b-', label = 'observed', linewidth = 2)
plt.plot_date(date_x,final_preds_expand * y_std + y_mean, 'r-', label = 'predicted', linewidth = 2)

plt.title("Model vs prediction on test data")
plt.legend(loc="upper left")
plt.title("ML Error %5.2fnT, Original Error %5.2fnT" % (
          ml_rms,
          original_rms))
plt.savefig("./SatDst/prediction_plot.png")
plt.show()