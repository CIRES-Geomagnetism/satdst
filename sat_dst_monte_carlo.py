
import pandas as pd
import numpy as np 
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import copy
import os
import time
from quickstart import main
from random import randrange
import random
import pickle as pickle


input_seq_len_array = [40,60,70,80]
#output_seq_len_array = [1,2,3,4,5]
output_seq_len_array = [1,2]
batch_size_array = [40,45,50]
#learning_rate_array = [0.001,0.005,0.0001,0.0005]
#lambda_l2_reg_array = [0.005,0.0001,0.0003,0.003,0.0005]
learning_rate_array = [0.0001,0.00005]
lambda_l2_reg_array = [0.0003,0.0005]
#total_iterations_array = [100,200,500,700,1000,1500,2000,2500,3000,7000,10000]
total_iterations_array = [7000,8500,10000,12000,15000]
hidden_dim_array = [80,90,100,110,120]
num_stacked_layers_array = [1,2]
GRADIENT_CLIPPING_ARRAY = [0.5]


#Input data set
cols_to_train_array_all = ["Dst","Bx","By","Bz","Sv","Den"]
output_data = "Dst"


#cols_to_train = "Dst,Bz,Sv,Den"
#cols_to_train_array = ["Dst","Bz","Sv","Den"]
#cols_to_train = "Dst,Bx,By,Bz,Sv,Den"
#cols_to_train_array = ["Dst","Bx","By","Bz","Sv","Den"]
# Testing year data
TestYear = "2003"

# close all figures
#plt.close("all")

# Read data. Columns No,year,month,day,hour,Dst,Bx,By,Bz,Sv,Den
#df = pd.read_csv('./SatDst/Solar_Wind_Dst_1997_2008.csv')
#df = pd.read_csv('./SatDst/Solar_Wind_Dst_1997_2016.csv')
df = pd.read_csv('./SatDst/Solar_Wind_Dst_1997_2016_shifted_back.csv')
print(df.head())

# plot data

#cols_to_plot = ["Dst", "Bx", "By", "Bz", "Sv", "Den"]
#i = 1
#plot each column
#plt.figure(figsize = (10,12))
#for col in cols_to_plot:
#    plt.subplot(len(cols_to_plot), 1, i)
#    plt.plot(df["Decyear"],df[col])
#    plt.title(col, y=0.5, loc='left')
#    i += 1
#plt.show()
## Fill NA with 0 
#print(df.isnull().sum())
#df.fillna(0, inplace = True)
## One-hot encode 'cbwd'
#temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
#df = pd.concat([df, temp], axis = 1)
#del df['cbwd'], temp

## Split into train and test - I am using 2008 (last year) as test data
# The original data is 11 years long 1997 to 2016

#df_train = df.iloc[np.r_[1:157776,166537:], :].copy() # Except 2015
#df_test = df.iloc[np.r_[157777:166536], :].copy() # 2015

#df_train = df.iloc[0:157776, :].copy() # 1997-2014
#df_test = df.iloc[157777:df.shape[0], :].copy()

#df_train = df.iloc[17521:df.shape[0], :].copy() # 1997-1998
#df_test = df.iloc[0:17520, :].copy()

#df_train = df.iloc[61345:df.shape[0], :].copy() #  2003 +
#df_test = df.iloc[52585:61344, :].copy() # 2003
#df_train = df.T[list(df.T.columns[0:52584]) + list(df.T.columns[61345:df.shape[0]])].T # Except 2003
#df_test = df.iloc[52585:61344, :].copy() # 2003

df_train = df.T[list(df.T.columns[0:43800]) + list(df.T.columns[61345:df.shape[0]])].T # Except 2002-2003
df_test = df.iloc[43800:61344, :].copy() # 2002-2003

#df_train = df.iloc[np.r_[1:96408,105193:], :].copy() # Except 2008
#df_test = df.iloc[np.r_[96409:105192], :].copy() # 2008

#df_train = df.iloc[np.r_[1:52584,61345:], :].copy() # Except 2003
#df_test = df.iloc[np.r_[52585:61344], :].copy() # 2003

while True:


    start_time = time.time()
# Set sequence lengths
    random_index = randrange(0,len(input_seq_len_array))
    input_seq_len = input_seq_len_array[random_index]
# Number of preceeding data (hours)
    random_index = randrange(0,len(output_seq_len_array))
    output_seq_len = output_seq_len_array[random_index]# Number of succeeding data (prediction)
    #output_seq_len = 5 # Number of succeeding data (prediction)
    random_index = randrange(0,len(batch_size_array))
    batch_size = batch_size_array[random_index] # Number of TS pieces used in one training session

## Learning Parameters
    random_index = randrange(0,len(learning_rate_array))
    learning_rate = learning_rate_array[random_index]
    
    random_index = randrange(0,len(lambda_l2_reg_array))
    lambda_l2_reg = lambda_l2_reg_array[random_index]
    
    random_index = randrange(1,len(cols_to_train_array_all))
    random.shuffle(cols_to_train_array_all)
    cols_to_train_array = cols_to_train_array_all[0:random_index]
    cols_to_train =','.join(cols_to_train_array)
#lambda_l2_reg = 0.00003  
## Network Parameters
# length of input signals
    input_seq_len = input_seq_len
# length of output signals
    output_seq_len = output_seq_len
# Number of iterations
    random_index = randrange(0,len(total_iterations_array))
    total_iterations = total_iterations_array[random_index]

#total_iterations = 7000
# size of LSTM Cell
    random_index = randrange(0,len(hidden_dim_array))
    hidden_dim = hidden_dim_array[random_index]
    #hidden_dim = 60
    # number of hidden layers
    random_index = randrange(0,len(num_stacked_layers_array))
    num_stacked_layers = num_stacked_layers_array[random_index]
#num_stacked_layers = 1
# gradient clipping - to avoid gradient exploding
    random_index = randrange(0,len(GRADIENT_CLIPPING_ARRAY))
    GRADIENT_CLIPPING = GRADIENT_CLIPPING_ARRAY[random_index]
   # GRADIENT_CLIPPING = 0.5 

# 2003 52585:61344
# 2008 96409 105192
# 2015 157777:166536

## take out the useful columns for modeling - you may also keep 'hour', 'day' or 'month' and to see if that will improve your accuracy
#X_train = df_train.loc[:, ['Dst', 'Bx', 'By', 'Bz', 'Sv', 'Den']].values.copy()
#X_test = df_test.loc[:, ['Dst', 'Bx', 'By', 'Bz', 'Sv', 'Den']].values.copy()

#X_train = df_train.loc[:, ['Dst','Bx','By','Bz','Sv','Den']].values.copy()
#X_test = df_test.loc[:, ['Dst','Bx','By','Bz','Sv','Den']].values.copy()

#X_train = df_train.loc[:, ['Dst','Bz','Sv','Den']].values.copy()
#X_test = df_test.loc[:, ['Dst','Bz','Sv','Den']].values.copy()

    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"),
     input_seq_len,
         output_seq_len,
         batch_size ,
         total_iterations ,
         hidden_dim ,
         num_stacked_layers,
         learning_rate,
         lambda_l2_reg ,
         GRADIENT_CLIPPING,
         cols_to_train,
         TestYear 
         )

    X_train = df_train.loc[:, cols_to_train_array].values.copy()
    X_test = df_test.loc[:, cols_to_train_array].values.copy()
    
    y_train = df_train['Dst'].values.copy().reshape(-1, 1)
    y_test = df_test['Dst'].values.copy().reshape(-1, 1)

X_train_mean = np.zeros(X_train.shape[1])
X_train_std = np.zeros(X_train.shape[1])
    ## z-score transform x 
for i in range(X_train.shape[1]):
    print(i)
    #print(X_train.shape[1])
    #X_train[:, i] = signal.detrend( X_train[:, i]) # detrending data
    X_train_mean[i] = X_train[:, i].mean()
    X_train_std[i] = X_train[:, i].std()
    print(X_train_mean[i], X_train_std[i])
    X_train[:, i] = (X_train[:, i] - X_train_mean[i]) / X_train_std[i]
    X_test[:, i] = (X_test[:, i] - X_train_mean[i]) /  X_train_std[i]
        
    ## z-score transform y
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    #print(y_mean,y_std)
    
    
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
    
    x, y = generate_train_samples()
    print(x.shape, y.shape)
    
    test_x, test_y = generate_test_samples()
    print(test_x.shape, test_y.shape)
    
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
        
        print("Training losses: ")
        for i in range(total_iterations):
            batch_input, batch_output = generate_train_samples(batch_size=batch_size)
            
            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
            #print(loss_t)
            
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
        #mean removed RMS error
        rms_error = np.sqrt((((final_preds * y_std - test_y * y_std) - np.mean(final_preds * y_std - test_y * y_std ) ) ** 2).mean())
        # mean error
        mean_error = np.mean(final_preds * y_std - test_y * y_std ) 
        print("Test RMS error is: ", rms_error)
        print("Test Mean error is: ", mean_error )
        
    bins = np.array([-300,-200,-100,-50,50,100])
    binned_observations = np.digitize(test_y * y_std + y_mean, bins)
    binned_rms = np.array([0.0,0.0,0.0,0.0,0.0])
    for i in range(1, len(bins)-1):
        if len(final_preds[binned_observations == i]) > 0:
            binned_rms[i] = np.sqrt((((final_preds[binned_observations == i] * y_std - test_y[binned_observations == i] * y_std)  ) ** 2).mean())
            #print(bins[i]-25,len(final_preds[binned_observations == i]), binned_rms[i])
 ## Update Google Spreadsheet with the training results
    main(time.strftime("%m/%d/%Y"),
     input_seq_len,
         output_seq_len,
         batch_size ,
         total_iterations ,
         hidden_dim ,
         num_stacked_layers,
         learning_rate,
         lambda_l2_reg ,
         GRADIENT_CLIPPING,
         cols_to_train,
         TestYear ,
         rms_error ,
         mean_error,
         elapsed_time/60,
         binned_rms[1],
         binned_rms[2],
         binned_rms[3],
         binned_rms[4])
    # save the non ann variables 

# Saving the objects:

with open('session_variables.pkl', 'wb') as f:  
    pickle.dump((input_seq_len,
          output_seq_len,
          batch_size ,
          total_iterations ,
          hidden_dim ,
          num_stacked_layers,
          learning_rate,
          lambda_l2_reg ,
          GRADIENT_CLIPPING,
          cols_to_train,
          cols_to_train_array,
          output_data ,
          TestYear ,
          rms_error ,
          mean_error,
          elapsed_time/60,
          X_train_mean,
          X_train_std,
          y_mean,
          y_std),
          f)
    print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"),
     input_seq_len,
         output_seq_len,
         batch_size ,
         total_iterations ,
         hidden_dim ,
         num_stacked_layers,
         learning_rate,
         lambda_l2_reg ,
         GRADIENT_CLIPPING,
         cols_to_train,
         TestYear,rms_error ,
         mean_error,
         elapsed_time/60,
         binned_rms[1],
         binned_rms[2],
         binned_rms[3],
         binned_rms[4] 
         )
       
        
    ## remove duplicate hours and concatenate into one long array
#test_y_expand = np.concatenate([test_y[i].reshape(-1) for i in range(0, test_y.shape[0], test_y.shape[1])], axis = 0)
#final_preds_expand = np.concatenate([final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], final_preds.shape[1])], axis = 0)
    #===============================================================================
    # plt.plot(final_preds_expand * y_std + y_mean, color = 'red', label = 'predicted')
    # plt.plot(test_y_expand * y_std + y_mean, color = 'blue', label = 'observed')
    # plt.title("Model vs prediction on test data")
    # plt.legend(loc="upper left")
    # plt.show()
    #===============================================================================
    
    
    
