
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import copy
import os
import time
import pickle as pickle
from datetime import timedelta, datetime

#Input data set
cols_to_train_array = ["Bx","By","Bz","Sv","Den"]

#cols_to_train_array = ["Bx","By","Bz","Sv"]
#cols_to_train = "Dst,Bx,By,Bz,Sv,Den"
#cols_to_train_array = ["Dst","Bx","By","Bz","Sv","Den"]
#cols_to_train = "Dst,Bx,By,Bz,Sv,Den"
#cols_to_train_array = ["Dst","Bx","By","Bz","Sv","Den"]
#cols_to_train = "Dst"
#cols_to_train_array = ["Dst"]
cols_to_train =','.join(cols_to_train_array)
output_data = "Dst"
reference_data = "LASP"



# Testing year data
#TestYear = "1997-1998"
#TestYear = "2015-2016"
TestYear = "2002-2003"
#TestYear = "2014-2016"
# close all figures
#plt.close("all")

# Read data. Columns No,year,month,day,hour,Dst,Bx,By,Bz,Sv,Den
#df = pd.read_csv('./SatDst/Solar_Wind_Dst_1997_2016.csv')
#df = pd.read_csv('./SatDst/Solar_Wind_Dst_1997_2016_shifted_back.csv')
df = pd.read_csv('./SatDst/Solar_Wind_Dst_1997_2016_shifted_forward.csv')
#df = pd.read_csv('./SatDst/Solar_Wind_Dst_2013_2016_1_minute.csv') # USGS 1 minute Dst and ACE 1 minute data

# Averaging Interval 
averaging_bin_size = 15
# averaging (Note disable this for hourly data !!)
#df = df.groupby(np.arange(len(df))//averaging_bin_size).mean();
#print(df.head())

# plot data

#===============================================================================
#cols_to_plot = ["Dst", "Bx", "By", "Bz", "Sv","Den","RC"]
#cols_to_plot = ["Dst", "Bx", "By", "Bz", "Sv"] # For minute data file, no density data
#i = 1
# plot each column
#===============================================================================
# plt.figure(figsize = (10,12))
# for col in cols_to_plot:
#      plt.subplot(len(cols_to_plot), 1, i)
#      plt.plot(df["Decyear"],df[col])
#      plt.title(col, y=0.5, loc='left')
#      i += 1
# plt.show()
#===============================================================================
#===============================================================================
## Fill NA with 0 
#print(df.isnull().sum())
#df.fillna(0, inplace = True)
## One-hot encode 'cbwd'
#temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
#df = pd.concat([df, temp], axis = 1)
#del df['cbwd'], temp

## Split into train and test - I am using 2008 (last year) as test data
# The original data is 11 years long 1997 to 2016
# 2003 52585:61344
# 2008 96409 105192
# 2015 157777:166536
# 2014 149018
# 2002 43800

#df_train = df.T[list(df.T.columns[0:157776]) + list(df.T.columns[166537:df.shape[0]])].T  # Except 2015 
#df_test = df.iloc[157777:166536, :].copy() # 2015

#df_train = df.iloc[0:149017, :].copy() # Except 2014-2016
#df_test = df.iloc[149018:df.shape[0], :].copy() # 2014-2016

#df_train = df.iloc[17521:df.shape[0], :].copy() # 1997-1998
#df_test = df.iloc[0:17520, :].copy()

#df_train = df.iloc[0:int(round(262080/averaging_bin_size)), :].copy() # 2013 to 2015 Dec 31 1 minute data
#df_test = df.iloc[262081:df.shape[0], :].copy() # 2016-1-1 to 2016-7-1 test data 1 minute
#df_test = df.iloc[int(round(262080/averaging_bin_size)) : int(round(262080/averaging_bin_size)) + int(round(1440*150/averaging_bin_size)) , :].copy() # 2016-1-1 to 2016-7-1 test data 1 minute

#df_train = df.iloc[np.r_[1:96408,105193:], :].copy() # Except 2008
#df_train = df[list(df.columns[0:96408]) + list(df.columns[105193:df.shape[0]])]
#df_test = df.iloc[96409:105192, :].copy() # 2008

#df_train = df.T[list(df.T.columns[0:52584]) + list(df.T.columns[61345:df.shape[0]])].T # Except 2003
#df_test = df.iloc[52585:61344, :].copy() # 2003

df_train = df.T[list(df.T.columns[0:43800]) + list(df.T.columns[61345:df.shape[0]])].T # Except 2002-2003
df_test = df.iloc[43800:61344, :].copy() # 2002-2003

start_time = time.time()
# Set sequence lengths
input_seq_len = 30
# Number of preceeding data (hours)

output_seq_len = 1# Number of succeeding data (prediction)
#output_seq_len = 5 # Number of succeeding data (prediction)

batch_size = 1 # Number of TS pieces used in one training session

## Learning Parameters

learning_rate = 0.00005
lambda_l2_reg = 0.0003

#lambda_l2_reg = 0.00003  
## Network Parameters
# length of input signals
input_seq_len = input_seq_len
# length of output signals
output_seq_len = output_seq_len
# Number of iterations

total_iterations = 1000

#total_iterations = 7000
# size of LSTM Cell

hidden_dim = 200
#hidden_dim = 60
# number of hidden layers

num_stacked_layers = 1
#num_stacked_layers = 1
# gradient clipping - to avoid gradient exploding

GRADIENT_CLIPPING = 0.5



## take out the useful columns for modeling - you may also keep 'hour', 'day' or 'month' and to see if that will improve your accuracy
#X_train = df_train.loc[:, ['Dst', 'Bx', 'By', 'Bz', 'Sv', 'Den']].values.copy()
#X_test = df_test.loc[:, ['Dst', 'Bx', 'By', 'Bz', 'Sv', 'Den']].values.copy()

#X_train = df_train.loc[:, ['Dst','Bx','By','Bz','Sv','Den']].values.copy()
#X_test = df_test.loc[:, ['Dst','Bx','By','Bz','Sv','Den']].values.copy()

#X_train = df_train.loc[:, ['Dst','Bz','Sv','Den']].values.copy()
#X_test = df_test.loc[:, ['Dst','Bz','Sv','Den']].values.copy()

X_train = df_train.loc[:, cols_to_train_array].values.copy()
X_test = df_test.loc[:, cols_to_train_array].values.copy()

y_train = df_train[output_data].values.copy().reshape(-1, 1)
y_test = df_test[output_data].values.copy().reshape(-1, 1)
yy_test = df_test[reference_data].values.copy().reshape(-1, 1)

## z-score transform x 
X_train_mean = np.zeros(X_train.shape[1])
X_train_std = np.zeros(X_train.shape[1])
    ## z-score transform x 
for i in range(X_train.shape[1]):

    #print(X_train.shape[1])
    #X_train[:, i] = signal.detrend( X_train[:, i]) # detrending data
    X_train_mean[i] = X_train[:, i].mean()
    X_train_std[i] = X_train[:, i].std()

    X_train[:, i] = (X_train[:, i] - X_train_mean[i]) / X_train_std[i]
    X_test[:, i] = (X_test[:, i] - X_train_mean[i]) /  X_train_std[i]
    
## z-score transform y
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

yy_std = yy_test.std()
yy_mean = yy_test.mean()
yy_test = (yy_test - yy_mean) / yy_std



def convert_partial_year(number):

 year = np.asarray(number).astype(int)
 date = []
 for i in range(len(number)): 
  d = timedelta(days=(number[i] - year[i])* (365 ))# Leapyear not considered !!
  day_one = datetime(year[i],1,1)
  date.append(d + day_one)
 return date

def generate_train_samples(x = X_train, y = y_train, batch_size = 30, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)

def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
    
    total_samples = x.shape[0]
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq

x, y = generate_train_samples(batch_size=1)


test_x, test_y = generate_test_samples()
test_xx, test_yy = generate_test_samples(X_test,yy_test,input_seq_len,output_seq_len)



# num of input signals
input_dim = X_train.shape[1]
# num of output signals
output_dim = y_train.shape[1]
# num of stacked lstm layers 

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
    
    


rnn_model = build_graph(feed_previous=False)

saver = tf.train.Saver()



init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    

    for i in range(total_iterations):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)
        
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        if i%100 == 0:
            print("Training losses at iteration ",i ,": ", loss_t*y_std)
        
    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./SatDst/', 'multivariate_ts_Dst'))
        
print("Checkpoint saved at: ", save_path)
elapsed_time = time.time() - start_time
print('Execution time in minutes:', (elapsed_time/60))
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



    print("Final Preds: ", final_preds)
    print("Final Preds Shape: ", final_preds.shape)

    final_preds_expand = np.concatenate(
        [final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], final_preds.shape[1])], axis=0)
    print("final_preds_expand: ", final_preds_expand)
    print("shape: ", final_preds_expand.shape)
    # mean error
    test_mean_error = np.mean(final_preds * y_std - test_y * y_std ) 
    # mean error
    ref_mean_error = np.mean(test_yy * yy_std - test_y * y_std ) 
    
      #mean removed RMS error
    rms_error = np.sqrt((((final_preds * y_std - test_y * y_std) - np.mean(final_preds * y_std - test_y * y_std ) ) ** 2).mean())
   #mean removed rms erro e.r.t reference
    rms_error_yy = np.sqrt((((test_yy * yy_std - test_y * y_std) - np.mean(test_yy * yy_std - test_y * y_std) ) ** 2).mean())

      #mean removed RMS error
    rms_error_with_mean = np.sqrt((((final_preds * y_std - test_y * y_std)  ) ** 2).mean())
   #mean removed rms erro e.r.t reference
    rms_error_yy_with_mean= np.sqrt((((test_yy * yy_std - test_y * y_std)  ) ** 2).mean())


    print("Test RMS error is: ", rms_error)
    print("Reference RMS error is: ", rms_error_yy)
    print("Test RMS error and mean error is: ", rms_error_with_mean)
    print("Reference RMS error and mean erroris: ", rms_error_yy_with_mean)
    print("Test Mean error is: ", test_mean_error )
    print("reference Mean error is: ", ref_mean_error )
    # Bin data according to activity levels
    bins = np.array([-300,-250,-200,-150,-100,-50,0,50,100])
    bins = np.array([-125,-100,-75,-50,-25,-20,-15,-10,-5,0,5,25])
    #bins = np.array([-300,-200,-100,-50,50,100])
    binned_observations = np.digitize(test_y * y_std + y_mean, bins)
    binned_rms = np.zeros(bins.__len__() + 1)
    binned_observations_ref = np.digitize(test_yy * yy_std + yy_mean, bins)
    binned_rms_ref = np.zeros(bins.__len__() + 1)
    for i in range(0, len(binned_rms) - 1):
        if len(final_preds[binned_observations == i]) > 0:
            binned_rms[i] = np.sqrt((((final_preds[binned_observations == i] * y_std - test_y[binned_observations == i] * y_std)  ) ** 2).mean())
            binned_rms_ref[i] = np.sqrt((((test_yy[binned_observations == i] * yy_std - test_y[binned_observations == i] * y_std)  ) ** 2).mean())
            # bin[i] prints RMS error between bins[i-1] and bins[i]. For the first case, it is
            # all values less than bin[i]. For last case
            print(bins[i],len(final_preds[binned_observations == i]), binned_rms[i], binned_rms_ref[i])
