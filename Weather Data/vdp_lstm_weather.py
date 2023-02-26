import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
import wandb
#os.environ["WANDB_API_KEY"] = ""
# import seaborn as sns
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def x_Sigma_w_x_T(x, W_Sigma):
    batch_sz = x.shape[0]
    xx_t = tf.reduce_sum(tf.multiply(x, x), axis=1, keepdims=True)
    xx_t_e = tf.expand_dims(xx_t, axis=2)
    return tf.multiply(xx_t_e, W_Sigma)

def w_t_Sigma_i_w(w_mu, in_Sigma):
    Sigma_1_1 = tf.matmul(tf.transpose(w_mu), in_Sigma)
    Sigma_1_2 = tf.matmul(Sigma_1_1, w_mu)
    return Sigma_1_2

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    Sigma_3_1 = tf.linalg.trace(in_Sigma)
    Sigma_3_2 = tf.expand_dims(Sigma_3_1, axis=1)
    Sigma_3_3 = tf.expand_dims(Sigma_3_2, axis=1)
    return tf.multiply(Sigma_3_3, W_Sigma)

def activation_Sigma(gradi, Sigma_in):
    grad1 = tf.expand_dims(gradi, axis=2)
    grad2 = tf.expand_dims(gradi, axis=1)
    return tf.multiply(Sigma_in, tf.matmul(grad1, grad2))

def Hadamard_sigma(sigma1, sigma2, mu1, mu2):
    sigma_1 = tf.multiply(sigma1, sigma2)
    sigma_2 = tf.matmul(tf.matmul(tf.linalg.diag(mu1), sigma2), tf.linalg.diag(mu1))
    sigma_3 = tf.matmul(tf.matmul(tf.linalg.diag(mu2), sigma1), tf.linalg.diag(mu2))
    return sigma_1 + sigma_2 + sigma_3

def grad_sigmoid(mu_in):
    with tf.GradientTape() as g:
        g.watch(mu_in)
        out = tf.sigmoid(mu_in)
    gradi = g.gradient(out, mu_in)
    return gradi

def grad_tanh(mu_in):
    with tf.GradientTape() as g:
        g.watch(mu_in)
        out = tf.tanh(mu_in)
    gradi = g.gradient(out, mu_in)
    return gradi

def mu_muT(mu1, mu2):
    mu11 = tf.expand_dims(mu1, axis=2)
    mu22 = tf.expand_dims(mu2, axis=1)
    return tf.matmul(mu11, mu22)

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels

BATCH_SIZE = 200
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=BATCH_SIZE, )
    ds = ds.map(self.split_window)
    return ds
@property
def train(self):
    return self.make_dataset(self.train_df)
@property
def val(self):
    return self.make_dataset(self.val_df)
@property
def test(self):
    return self.make_dataset(self.test_df)
@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result
    
def sigma_regularizer1(x):
    input_size = 19.
    f_s = tf.math.softplus(x)  # tf.math.log(1. + tf.math.exp(x))
    return  -input_size * tf.reduce_mean(1. + tf.math.log(f_s) - f_s, axis=-1)

def sigma_regularizer2(x):
    f_s = tf.math.softplus(x)  # tf.math.log(1. + tf.math.exp(x))
    return  -tf.reduce_mean(1. + tf.math.log(f_s) - f_s, axis=-1)
    
class densityPropLSTMCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units]), tf.TensorShape([units, units]),
                           tf.TensorShape([units, units])]
        super(densityPropLSTMCell, self).__init__(**kwargs)
    def build(self, input_shape):        
        input_size = input_shape[-1]        
        ini_sigma = -4.6
        #min_sigma = -2.2
        init_mu = 0.05
        seed_ = None
        tau1 = 100.# / input_size
        tau2 = 100. / self.units
        # Forget Gate
        self.U_f =self.add_weight(name='U_f', shape=(input_size, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_),  trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=tau1, l2=tau1))
        self.uf_sigma = self.add_weight(name='uf_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_f =self.add_weight(name='W_f', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=tau2, l2=tau2))
        self.wf_sigma = self.add_weight(name='wf_sigma', shape=(self.units,),initializer=tf.constant_initializer(ini_sigma), trainable=True,regularizer=sigma_regularizer2)
        # Input Gate
        self.U_i = self.add_weight(name='U_i', shape=(input_size, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True,regularizer=tf.keras.regularizers.l1_l2(l1=tau1, l2=tau1))
        self.ui_sigma = self.add_weight(name='ui_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_i = self.add_weight(name='W_i', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_),trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=tau2, l2=tau2))
        self.wi_sigma = self.add_weight(name='wi_sigma', shape=(self.units,),initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer2)
        # Output Gate
        self.U_o = self.add_weight(name='U_o', shape=(input_size, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=tau1, l2=tau1))
        self.uo_sigma = self.add_weight(name='uo_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_o = self.add_weight(name='W_o', shape=(self.units, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=tau2, l2=tau2))
        self.wo_sigma = self.add_weight(name='wo_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer2)
        # Gate Gate
        self.U_g =self.add_weight(name='U_g', shape=(input_size, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l1_l2(l1=tau1, l2=tau1))
        self.ug_sigma = self.add_weight(name='ug_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_g = self.add_weight(name='W_g', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True,regularizer=tf.keras.regularizers.l1_l2(l1=tau2, l2=tau2))
        self.wg_sigma = self.add_weight(name='wg_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True,regularizer=sigma_regularizer2)
        self.built = True 
    def call(self, inputs, states):
        # state should be in [(batch, units), (batch, units, units)], mean vector and covaraince matrix
        prev_state, prev_istate, Sigma_state, Sigma_istate = states

        ## Forget Gate
        f = tf.sigmoid(tf.matmul(inputs, self.U_f) + tf.matmul(prev_state, self.W_f))
        Uf_Sigma = tf.linalg.diag(tf.math.softplus(self.uf_sigma))
        Sigma_Uf = x_Sigma_w_x_T(inputs, Uf_Sigma)
        ################
        Wf_Sigma = tf.linalg.diag(tf.math.softplus(self.wf_sigma))
        Sigma_f1 = w_t_Sigma_i_w(self.W_f, Sigma_state)
        Sigma_f2 = x_Sigma_w_x_T(prev_state, Wf_Sigma)
        Sigma_f3 = tr_Sigma_w_Sigma_in(Sigma_state, Wf_Sigma)
        Sigma_out_ff = Sigma_f1 + Sigma_f2 + Sigma_f3 + Sigma_Uf
        ################
        gradi_f = grad_sigmoid(tf.matmul(inputs, self.U_f) + tf.matmul(prev_state, self.W_f))
        Sigma_out_f = activation_Sigma(gradi_f, Sigma_out_ff)
        ###################################
        ###################################
        ## Input Gate
        i = tf.sigmoid(tf.matmul(inputs, self.U_i) + tf.matmul(prev_state, self.W_i))
        Ui_Sigma = tf.linalg.diag(tf.math.softplus(self.ui_sigma))
        Sigma_Ui = x_Sigma_w_x_T(inputs, Ui_Sigma)
        ################
        Wi_Sigma = tf.linalg.diag(tf.math.softplus(self.wi_sigma))
        Sigma_i1 = w_t_Sigma_i_w(self.W_i, Sigma_state)
        Sigma_i2 = x_Sigma_w_x_T(prev_state, Wi_Sigma)
        Sigma_i3 = tr_Sigma_w_Sigma_in(Sigma_state, Wi_Sigma)
        Sigma_out_ii = Sigma_i1 + Sigma_i2 + Sigma_i3 + Sigma_Ui
        ################
        gradi_i = grad_sigmoid(tf.matmul(inputs, self.U_i) + tf.matmul(prev_state, self.W_i))
        Sigma_out_i = activation_Sigma(gradi_i, Sigma_out_ii)
        ###################################
        ###################################
        ## Output Gate
        o = tf.sigmoid(tf.matmul(inputs, self.U_o) + tf.matmul(prev_state, self.W_o))
        Uo_Sigma = tf.linalg.diag(tf.math.softplus(self.uo_sigma))
        Sigma_Uo = x_Sigma_w_x_T(inputs, Uo_Sigma)
        ################
        Wo_Sigma = tf.linalg.diag(tf.math.softplus(self.wo_sigma))
        Sigma_o1 = w_t_Sigma_i_w(self.W_o, Sigma_state)
        Sigma_o2 = x_Sigma_w_x_T(prev_state, Wo_Sigma)
        Sigma_o3 = tr_Sigma_w_Sigma_in(Sigma_state, Wo_Sigma)
        Sigma_out_oo = Sigma_o1 + Sigma_o2 + Sigma_o3 + Sigma_Uo
        ################
        gradi_o = grad_sigmoid(tf.matmul(inputs, self.U_o) + tf.matmul(prev_state, self.W_o))
        Sigma_out_o = activation_Sigma(gradi_o, Sigma_out_oo)
        ###################################
        ###################################
        ## Gate Gate
        g = tf.tanh(tf.matmul(inputs, self.U_g) + tf.matmul(prev_state, self.W_g))
        Ug_Sigma = tf.linalg.diag(tf.math.softplus(self.ug_sigma))
        Sigma_Ug = x_Sigma_w_x_T(inputs, Ug_Sigma)
        ################
        Wg_Sigma = tf.linalg.diag(tf.math.softplus(self.wg_sigma))
        Sigma_g1 = w_t_Sigma_i_w(self.W_g, Sigma_state)
        Sigma_g2 = x_Sigma_w_x_T(prev_state, Wg_Sigma)
        Sigma_g3 = tr_Sigma_w_Sigma_in(Sigma_state, Wg_Sigma)
        Sigma_out_gg = Sigma_g1 + Sigma_g2 + Sigma_g3 + Sigma_Ug
        ################
        gradi_g = grad_tanh(tf.matmul(inputs, self.U_g) + tf.matmul(prev_state, self.W_g))
        Sigma_out_g = activation_Sigma(gradi_g, Sigma_out_gg)
        ###################################
        ###################################
        ## Current Internal State
        c = tf.multiply(prev_istate, f) + tf.multiply(i, g)
        ################
        sigma_cf = Hadamard_sigma(Sigma_istate, Sigma_out_f, prev_istate, f)
        sigma_ig = Hadamard_sigma(Sigma_out_i, Sigma_out_g, i, g)
        Sigma_out_c = sigma_cf + sigma_ig
        ###################################
        ###################################
        ## Current State
        mu_out = tf.multiply(tf.tanh(c), o)
        ################
        gradi_tanhc = grad_tanh(c)
        Sigma_out_tanhc = activation_Sigma(gradi_tanhc, Sigma_out_c)
        Sigma_out = Hadamard_sigma(Sigma_out_tanhc, Sigma_out_o, tf.tanh(c), o)
        Sigma_out_c = tf.where(tf.math.is_nan(Sigma_out_c), tf.zeros_like(Sigma_out_c), Sigma_out_c)
        Sigma_out_c = tf.where(tf.math.is_inf(Sigma_out_c), tf.zeros_like(Sigma_out_c), Sigma_out_c)
        Sigma_out_c = tf.linalg.set_diag(Sigma_out_c, tf.abs(tf.linalg.diag_part(Sigma_out_c)))
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        output = mu_out
        new_state = (mu_out, c, Sigma_out, Sigma_out_c)
        return output, new_state

class VDP_Dense(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units):
        super(VDP_Dense, self).__init__()
        self.units = units
    def build(self, input_shape):
        ini_sigma = 0.542
      #  min_sigma = -6.9
        tau = 100. / input_shape[-1]
       # print(input_shape[-1])
        self.w_mu = self.add_weight(name='w_mu', shape=(input_shape[-1], self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=.05 , seed=None), regularizer=tf.keras.regularizers.l1_l2(l1=tau, l2=tau),trainable=True, )#, l2=tau), #tf.keras.regularizers.l1_l2
                                    
        self.w_sigma = self.add_weight(name='w_sigma',shape=(self.units,),
                                       initializer= tf.constant_initializer(ini_sigma), #tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,seed=None),
                                       regularizer=sigma_regularizer2, #  
                                       trainable=True, ) 
    def call(self, mu_in, Sigma_in):
        mu_out = tf.matmul(mu_in, self.w_mu)  
        W_sigma = tf.math.log(1. + tf.math.exp(self.w_sigma))       
        Sigma_1 = w_t_Sigma_i_w(self.w_mu, Sigma_in)  # shape=[batch_size]
        Sigma_2 = tf.expand_dims(tf.multiply(tf.reduce_sum(tf.multiply(mu_in, mu_in), axis=1, keepdims=True), W_sigma), axis=2)  # shape=[batch_size]
        Sigma_3 = tf.expand_dims(tf.expand_dims(tf.multiply(tf.linalg.trace(Sigma_in), W_sigma), axis=1), axis=2)  # shape=[batch_size]
        Sigma_out = (Sigma_1 + Sigma_2 + Sigma_3)  # shape=[batch_size]
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        return mu_out, Sigma_out

class Density_prop_LSTM(tf.keras.Model):
  def __init__(self, units, name=None): 
    super(Density_prop_LSTM, self).__init__()
    self.units = units
    self.cell = densityPropLSTMCell(self.units)
    self.rnn = tf.keras.layers.RNN(self.cell, return_state=True)#, stateful=True)
    self.linear_1 = VDP_Dense(1)
  def call(self, inputs, training=True):
    xx = self.rnn(inputs)
    x, mu_state, c_state, sigma_state, sigma_cstate = xx   
    outputs, Sigma = self.linear_1(mu_state, sigma_state)   
    Sigma = tf.where(tf.math.is_nan(Sigma), tf.zeros_like(Sigma), Sigma)
    Sigma = tf.where(tf.math.is_inf(Sigma), tf.zeros_like(Sigma), Sigma)
    return outputs, Sigma    

def nll_gaussian(y_test, y_pred_mean, y_pred_sd):
    loss1 = tf.math.reduce_mean(tf.math.divide_no_nan(tf.square(y_test - y_pred_mean) , y_pred_sd)  )
    loss2 = tf.math.reduce_mean(tf.math.log(y_pred_sd ) )
    loss2 =  tf.where(tf.math.is_nan(loss2), tf.zeros_like(loss2), loss2)
    loss = tf.reduce_mean(tf.math.add(loss1, loss2 )) 
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss) 
    return loss 
       
def main_function(time_step=24, input_dim=19, units=30, output_size=1, batch_size=200, epochs=50, lr=0.001, lr_end = 0.000001, kl_factor=0.001,
          Random_noise=True, gaussain_noise_std=0.5, Adversarial_noise=False, epsilon=0.3, 
          BIM_adversarial=False, alpha= 1, maxAdvStep=100,  
          Training=False, Testing=True, continue_training=False, saved_model_epochs=50):
    PATH = './vdp_lstm2/saved_models_with_hidden_unit_{}_lr_{}/VDP_lstm_epoch_{}/'.format(units, lr, epochs)
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    # slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[2::3]
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0
    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0
    # The above inplace edits are reflected in the DataFrame
    df['wv (m/s)'].min()
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')
    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180
    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)
    timestamp_s = date_time.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    fft = tf.signal.rfft(df['T (degC)'])
    f_per_dataset = np.arange(0, len(fft))
    n_samples_h = len(df['T (degC)'])
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples_h / (hours_per_year)
    f_per_year = f_per_dataset / years_per_dataset
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    num_features = df.shape[1]
#    train_mean = train_df.mean()
#    train_std = train_df.std()#    
#    train_df = (train_df - train_mean) / train_std
#    val_df = (val_df - train_mean) / train_std
#    test_df = (test_df - train_mean) / train_std    
    train_df = (train_df - train_df.min()) / (train_df.max() - train_df.min())
    val_df = (val_df - val_df.min()) / (val_df.max() - val_df.min())
    test_df = (test_df - test_df.min()) / (test_df.max() - test_df.min())
    WindowGenerator.split_window = split_window
    # WindowGenerator.plot = plot
    WindowGenerator.make_dataset = make_dataset
    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.test = test
    WindowGenerator.example = example

    CONV_WIDTH = time_step
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH, 
        label_width=1,
        shift=1,
        label_columns=['T (degC)'], train_df=train_df, val_df=val_df, test_df=test_df)
    # conv_window
    # conv_window.plot()
    # plt.title("Given 3h as input, predict 1h into the future.")
    tr_dataset = conv_window.train
    val_dataset = conv_window.val
    test_dataset = conv_window.test    
#    print(tr_dataset.shape)
#    print(tr_dataset.dtype)
#    print(val_dataset.shape)
#    print(val_dataset.shape)
#    tr_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(batch_size)
#    val_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(batch_size)
      
    num_train_steps = epochs * int(train_df.shape[0] /batch_size)
#    step = min(step, decay_steps)
#    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,  decay_steps=num_train_steps,  end_learning_rate=lr_end, power=10.)    
    lstm_model = Density_prop_LSTM(units, name='vdp_lstm')
    #mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)#reduction=tf.keras.losses.Reduction.SUM)
    #test_loss = tf.keras.metrics.Mean(name='test_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn )#, clipnorm=1.0)
    #MAE = tf.keras.metrics.MeanAbsoluteError()
    
    @tf.function()# Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            logits, sigma = lstm_model(x)
            sigma = tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1.0e-6), clip_value_max=tf.constant(1.0e+5))
            loss_final = nll_gaussian(y, tf.expand_dims(logits, axis=2 ), sigma ) #tf.math.reduce_mean(tf.math.divide_no_nan(mse(y, tf.expand_dims(logits, axis=2 )), (sigma**2)) )+ tf.math.reduce_mean(tf.math.log(sigma**2 +1.0e-4))           
            regularization_loss=tf.math.add_n(lstm_model.losses)
            loss = 0.5 * (loss_final  + kl_factor*regularization_loss )
            loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
            loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)  
        gradients = tape.gradient(loss, lstm_model.trainable_weights) 
#            for g,v in zip(gradients, lstm_model.trainable_weights):
#                tf.print(v.name, tf.reduce_mean(g))          

        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, lstm_model.trainable_weights))                
        return loss, logits, gradients  , regularization_loss, loss_final, sigma
    @tf.function()# Make it fast.
    def valid_on_batch(x, y):          
        logits, sigma = lstm_model(x)
        sigma = tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1.0e-6), clip_value_max=tf.constant(1.0e+5))            
        loss_final = nll_gaussian(y, tf.expand_dims(logits, axis=2 ), sigma ) #tf.math.reduce_mean(tf.math.divide_no_nan(mse(y, tf.expand_dims(logits, axis=2 )), (sigma**2)) )+ tf.math.reduce_mean(tf.math.log(sigma**2 +1.0e-4))               
        regularization_loss=tf.math.add_n(lstm_model.losses)
        loss = 0.5 * (loss_final  + kl_factor*regularization_loss )        
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)        
        return loss, logits, regularization_loss, loss_final, sigma
        
    @tf.function
    def test_on_batch(x):  
        lstm_model.trainable = False                    
        mu_out, sigma = lstm_model(x, training=False)            
        return mu_out, sigma    
       
#    def create_adversarial_pattern(input_image, input_label):
#          with tf.GradientTape() as tape: 
#            tape.watch(input_image)
#            prediction, sigma = lstm_model(input_image) 
#            loss = 0.5* (mse(y, logits)/ tf.math.reduce_mean(sigma**2+1.0e-10) + tf.math.reduce_mean(tf.math.log(sigma**2 +1.0e-6))   )
##            loss_final, loss1, loss2 = nll_gaussian(input_label, prediction,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1.0e-10),
##                                   clip_value_max=tf.constant(1.0e+5)))                         
##            loss = 0.5 * loss_final 
#          # Get the gradients of the loss w.r.t to the input image.
#          gradient = tape.gradient(loss, input_image)
#          # Get the sign of the gradients to create the perturbation
#          signed_grad = tf.sign(gradient)
#          return signed_grad
    @tf.function()# Make it fast.        
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
       #     lstm_model.trainable = False 
            prediction, sigma = lstm_model(input_image, training=False) 
            sigma = tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1.0e-6), clip_value_max=tf.constant(1.0e+5))   
            loss_final = nll_gaussian(input_label, tf.expand_dims(logits, axis=2 ), sigma )            
            regularization_loss=tf.math.add_n(lstm_model.losses)
            loss = 0.5 * (loss_final  - kl_factor*regularization_loss )  
            loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
            loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
          #  loss = tf.clip_by_value(t=loss, clip_value_min=tf.constant(-1.0e+4),  clip_value_max=tf.constant(1.0e+4))
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          gradient = tf.where(tf.math.is_nan(gradient),  tf.constant(1.0e-6, shape=gradient.shape), gradient)
          gradient = tf.where(tf.math.is_inf(gradient),  tf.constant(1.0e-6, shape=gradient.shape), gradient)
         # print('gradient', gradient)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
        #  print("signed_grad shape", signed_grad.shape)
          return signed_grad           
                     
    if Training:
       # wandb.init(entity = "", project="VDP_LSTM_weather_with_hidden_units_{}_epochs_{}_lr_{}_latest".format(units, epochs, lr))
        if continue_training:
            saved_model_path = './vdp_lstm2/saved_models_with_hidden_unit_{}_lr_{}/VDP_lstm_epoch_{}/'.format(units,lr,  saved_model_epochs)
            lstm_model.load_weights(saved_model_path + 'vdp_lstm_model')
        train_mae = np.zeros(epochs)
        valid_mae = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs):
            print('Epoch: ', epoch + 1, '/', epochs)
            mae1 = 0
            mae_valid1 = 0
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0
            # -------------Training--------------------            
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step / int(train_df.shape[0] / (batch_size)))
                if (x.shape[0] == batch_size):
                    loss, logits, gradients, regularization_loss, loss_final, sigma = train_on_batch(x, y)  
#                    print( logits.shape)   
#                    print( y.shape)  
#                    print( sigma.shape)                
                    err1 += loss.numpy()                   
                    mae_err = tf.math.reduce_mean(tf.abs(y - tf.expand_dims(logits, axis=2 )))#  .update_state(y, tf.expand_dims(logits, axis=2 ))                    
                    mae1 += mae_err.numpy() 
                    tr_no_steps += 1
                    if step % 500 == 0:                        
                        print("Step:", step, "Loss:", float(err1 / (tr_no_steps )))
                        print("Training MAE: %.6f" % float(mae1 / (tr_no_steps )))
                        print('regularization_loss', regularization_loss.numpy())
                        print('gradient', np.mean(gradients[0]))
                        print('Average Variance value', tf.reduce_mean(sigma).numpy())
                    # wandb.log({"Average Variance value": tf.reduce_mean(sigma).numpy(),
                    #            "Regularization_loss": regularization_loss.numpy(),
                    #            "Log-Likelihood Loss": np.mean(loss_final.numpy()),
                    #            "Total Training Loss": loss.numpy() ,
                    #            "Training MAE per minibatch": mae_err.numpy() ,
                    #            "gradient per minibatch": np.mean(gradients[0]),
                    #            'epoch': epoch
                    # })
            train_mae[epoch] = mae1 / tr_no_steps 
            train_err[epoch] = err1 / tr_no_steps
            print('Training MAE  ', train_mae[epoch])
            print('Training Loss  ', train_err[epoch])                      
            # ---------------Validation----------------------           
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(val_df.shape[0] / (batch_size)))
                if (x.shape[0] == batch_size):
                    total_vloss, logits, regularization_loss, vloss, sigma = valid_on_batch(x, y)                    
                    err_valid1 += total_vloss.numpy()                                       
                    valid_mae_err = tf.math.reduce_mean(tf.abs(y - tf.expand_dims(logits, axis=2 )))                    
                    mae_valid1 += valid_mae_err.numpy()   
                    va_no_steps += 1   
                    if step % 400 == 0:
                        print("Step:", step, "Loss:", float(err_valid1/va_no_steps ))
                        print("Total validation MAE so far: %.6f" % (mae_valid1/va_no_steps))
                    
            #         wandb.log({"Average Variance value (validation Set)": tf.reduce_mean(sigma).numpy(),
            #                    "Regularization_loss (validation Set)": regularization_loss.numpy(),
            #                    "Log-Likelihood Loss (validation Set)": np.mean(vloss.numpy() ),
            #                    "Total Validation Loss": total_vloss.numpy() ,
            #                    "Validation MAE per minibatch": valid_mae_err.numpy()
            #                     })
            # wandb.log({"Training Loss": (err1 / tr_no_steps),
            #             "Training MAE": (mae1 / tr_no_steps),
            #             "Validation Loss": (err_valid1 / va_no_steps),
            #             "Validation MAE": (mae_valid1 / va_no_steps),
            #             'epoch': epoch
            #            })
            lstm_model.save_weights(PATH + 'vdp_lstm_model') 
            valid_mae[epoch] = mae_valid1 / va_no_steps
            valid_error[epoch] = err_valid1 / va_no_steps
            stop = timeit.default_timer()
            print('Total Training Time: ', stop - start)
            print('Training MAE  ', train_mae[epoch])
            print('Validation MAE  ', valid_mae[epoch])
            print('------------------------------------')
            print('Training Loss  ', train_err[epoch])
            print('Validation Loss  ', valid_error[epoch])
            # -----------------End Training--------------------------
        lstm_model.save_weights(PATH + 'vdp_lstm_model')
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_mae, 'b', label='Training MAE')
            plt.plot(valid_mae,'r' , label='Validation MAE')
           # plt.ylim(0, 1.1)
            plt.title("Density Propagation LSTM on time series forecasting Data")
            plt.xlabel("Epochs")
            plt.ylabel("Mean Absolute Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_LSTM_on_time_series_Data_mae.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training Loss')
            plt.plot(valid_error,'r' , label='Validation Loss')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation LSTM on time series forecasting Data")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_LSTM_on_time_series_Data_loss.png')
            plt.close(fig)
        
        f = open(PATH + 'training_validation_mae_loss.pkl', 'wb')         
        pickle.dump([train_mae, valid_mae, train_err, valid_error], f)                                                   
        f.close()                  
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w') 
        textfile.write(' Input Sequence Length : ' +str(time_step))           
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr)) 
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))         
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Training  MAE : "+ str( train_mae))
                textfile.write("\n Validation MAE : "+ str(valid_mae ))
                    
                textfile.write("\n Training  Loss : "+ str( train_err))
                textfile.write("\n Validation Loss : "+ str(valid_error ))
            else:
                textfile.write("\n Training  MAE : "+ str(np.mean(train_mae[epoch])))
                textfile.write("\n Validation MAE : "+ str(np.mean(valid_mae[epoch])))
                
                textfile.write("\n Training  Loss : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Validation Loss : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    if(Testing):
        test_path = 'test_results/'
        if Random_noise:
            test_path = 'test_random_noise_{}/'.format(gaussain_noise_std)               
        os.makedirs(PATH + test_path) 
        lstm_model.load_weights(PATH + 'vdp_lstm_model')
        
        rmse_test = np.zeros(int(test_df.shape[0] /batch_size))     
        mae_test = np.zeros(int(test_df.shape[0] /batch_size)) 
        true_x = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size])
        logits_ = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, output_size])
        sigma_ = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, output_size, output_size]) 
        test_start = timeit.default_timer()
        test_no_steps = 0          
        for step, (x, y) in enumerate(test_dataset):
            update_progress(step / int(test_df.shape[0] / (batch_size)) ) 
            if (x.shape[0] == batch_size):            
                true_x[test_no_steps, :, :, :] = x
                true_y[test_no_steps, :] = np.squeeze(y)
                if Random_noise:
                    noise = tf.random.normal(shape = [batch_size, time_step, input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                    x = x +  noise         
                logits, sigma = test_on_batch(x)  
                logits_[test_no_steps,:,:] =logits
                sigma_[test_no_steps, :, :, :]= sigma                                 
                mae_ = tf.math.reduce_mean(tf.abs(y - tf.expand_dims(logits, axis=2 ))) 
                rmse_ = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - tf.expand_dims(logits, axis=2 ))))                                         
                mae_test[test_no_steps] = mae_.numpy()
                rmse_test[test_no_steps] = rmse_.numpy()
                if step % 100 == 0:
                    print("Test MAE: %.6f" % mae_.numpy())     
                    print("Test RMSE: %.6f" % rmse_.numpy())        
                test_no_steps+=1       
        test_stop = timeit.default_timer()
        print('Total Test Time: ', test_stop - test_start) 
        test_mae = np.mean(mae_test) 
        test_rmse = np.mean(rmse_test)          
        print('Average Test MAE : ', test_mae )        
        print('Average Test RMSE : ', test_rmse )
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_, sigma_, true_y, mae_test, rmse_test ], pf)                                                   
        pf.close()
        
        if Random_noise:
            snr_signal = np.zeros([int(test_df.shape[0]/(batch_size)) ,batch_size])
            for i in range(int(test_df.shape[0]/(batch_size))):
                for j in range(batch_size):
                    if(i<test_no_steps):  
                       noise = tf.random.normal(shape = [time_step, input_dim ], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ).numpy()   
                       snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/(np.sum( np.square(noise))+1e-5 ))
            print('SNR', np.mean(snr_signal))
        
        sigma_1 = np.reshape(tf.squeeze(tf.squeeze(sigma_)), int(test_df.shape[0]/(batch_size))* batch_size)
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(sigma_1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()
        
            
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write('\n Total test time in sec : ' +str(test_stop - test_start))    
        textfile.write("\n Average Test MAE : "+ str( test_mae))  
        textfile.write("\n Average Test RMSE : "+ str( test_rmse))  
        textfile.write("\n Output Variance: "+ str(np.mean(sigma_)))                 
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std )) 
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))                           
        textfile.write("\n---------------------------------")    
        textfile.close()
        
    if(Adversarial_noise):
        test_path = 'test_adversarial_noise_{}_latest/'.format(epsilon)  
        os.makedirs(PATH + test_path)       
        lstm_model.load_weights(PATH + 'vdp_lstm_model')
        lstm_model.trainable = False               
        
        rmse_test = np.zeros(int(test_df.shape[0] /batch_size))     
        mae_test = np.zeros(int(test_df.shape[0] /batch_size))         
        true_x = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        adv_perturbations = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size])
        logits_ = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, output_size])
        sigma_ = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, output_size, output_size])  
        adv_test_start = timeit.default_timer()
        test_no_steps = 0         
        for step, (x, y) in enumerate(test_dataset):
            update_progress(step / int(test_df.shape[0] / (batch_size)) ) 
            if (x.shape[0] == batch_size): 
                true_x[test_no_steps, :, :, :] = x
                true_y[test_no_steps, :] = np.squeeze(y)                  
                adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y)
                adv_x = x + epsilon*adv_perturbations[test_no_steps, :, :, :] 
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)                         
                logits, sigma = test_on_batch(adv_x)  
                logits_[test_no_steps,:,:] =logits
                sigma_[test_no_steps, :, :, :]= sigma 
                mae_ = tf.math.reduce_mean(tf.abs(y - tf.expand_dims(logits, axis=2 ))) 
                rmse_ = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - tf.expand_dims(logits, axis=2 ))))                                         
                mae_test[test_no_steps] = mae_.numpy()
                rmse_test[test_no_steps] = rmse_.numpy()
                if step % 100 == 0:
                    print("Test MAE: %.6f" % mae_.numpy())     
                    print("Test RMSE: %.6f" % rmse_.numpy())                           
                test_no_steps+=1       
        adv_test_stop = timeit.default_timer()
        print('Total Test Time: ', adv_test_stop - adv_test_start) 
        test_mae = np.mean(mae_test) 
        test_rmse = np.mean(rmse_test)          
        print('Average Test MAE : ', test_mae )        
        print('Average Test RMSE : ', test_rmse )
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_, sigma_, true_y, mae_test, rmse_test ], pf)                                                   
        pf.close()
        
        snr_signal = np.zeros([int(test_df.shape[0]/(batch_size)) ,batch_size])
        for i in range(int(test_df.shape[0]/batch_size)):
            for j in range(batch_size): 
                if(i<test_no_steps):               
                   snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/(np.sum( np.square(true_x[i,j,:, :] - np.clip(true_x[i,j,:, :]+epsilon*adv_perturbations[i, j, :, :], 0.0, 1.0)  ))+1e-5 ))   
        print('SNR', np.mean(snr_signal))
        
        sigma_1 = np.reshape(tf.squeeze(tf.squeeze(sigma_)), int(test_df.shape[0]/(batch_size))* batch_size)
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(sigma_1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()    
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write('\n Total test time in sec : ' +str(adv_test_stop - adv_test_start))    
        textfile.write("\n Average Test MAE : "+ str( test_mae))  
        textfile.write("\n Average Test RMSE : "+ str( test_rmse))  
        textfile.write("\n Output Variance: "+ str(np.mean(sigma_)))                   
        textfile.write("\n---------------------------------")
        if Adversarial_noise:              
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
        textfile.write("\n---------------------------------")    
        textfile.close()
        
    if(BIM_adversarial):
        test_path = 'test_BIMadversarial_noise_{}_alpha{}_iter{}/'.format(epsilon, alpha, maxAdvStep)  
        os.makedirs(PATH + test_path)   
        lstm_model.load_weights(PATH + 'vdp_lstm_model')
        lstm_model.trainable = False                     
        
        rmse_test = np.zeros(int(test_df.shape[0] /batch_size))     
        mae_test = np.zeros(int(test_df.shape[0] /batch_size))         
        true_x = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        adv_perturbations = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size])
        logits_ = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, output_size])
        sigma_ = np.zeros([int(test_df.shape[0] / (batch_size)), batch_size, output_size, output_size])  
        bim_adv_test_start = timeit.default_timer()
        test_no_steps = 0         
        for step, (x, y) in enumerate(test_dataset):
            update_progress(step / int(test_df.shape[0] / (batch_size)) ) 
            if (x.shape[0] == batch_size): 
                true_x[test_no_steps, :, :, :] = x
                true_y[test_no_steps, :] = np.squeeze(y)  
                adv_x = x
                for advStep in range(maxAdvStep):                   
                    adv_perturbations1 = create_adversarial_pattern(x, y)
                    adv_x = adv_x + alpha *adv_perturbations1
                    adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                    adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)           
                adv_perturbations[test_no_steps, :, :, :] = adv_x                                      
                logits, sigma = test_on_batch(adv_x)  
                logits_[test_no_steps,:,:] =logits
                sigma_[test_no_steps, :, :, :]= sigma 
                mae_ = tf.math.reduce_mean(tf.abs(y - tf.expand_dims(logits, axis=2 ))) 
                rmse_ = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - tf.expand_dims(logits, axis=2 ))))                                         
                mae_test[test_no_steps] = mae_.numpy()
                rmse_test[test_no_steps] = rmse_.numpy()
                if step % 100 == 0:
                    print("Test MAE: %.6f" % mae_.numpy())     
                    print("Test RMSE: %.6f" % rmse_.numpy())                           
                test_no_steps+=1       
        bim_adv_test_stop = timeit.default_timer()
        print('Total Test Time: ', bim_adv_test_stop - bim_adv_test_start) 
        test_mae = np.mean(mae_test) 
        test_rmse = np.mean(rmse_test)          
        print('Average Test MAE : ', test_mae )        
        print('Average Test RMSE : ', test_rmse )
        
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_, sigma_, true_y, mae_test, rmse_test ], pf)                                                   
        pf.close()
        
        snr_signal = np.zeros([int(test_df.shape[0]/(batch_size)) ,batch_size])
        for i in range(int(test_df.shape[0]/batch_size)):
            for j in range(batch_size): 
                if(i<test_no_steps):              
                   snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/(np.sum( np.square(true_x[i,j,:, :] - adv_perturbations[i, j, :, :]  )) +1e-5))                
        print('SNR', np.mean(snr_signal))
        
        sigma_1 = np.reshape(tf.squeeze(tf.squeeze(sigma_)), int(test_df.shape[0]/(batch_size))* batch_size)
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(sigma_1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()    
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write('\n Total test time in sec : ' +str(bim_adv_test_stop - bim_adv_test_start))    
        textfile.write("\n Average Test MAE : "+ str( test_mae))  
        textfile.write("\n Average Test RMSE : "+ str( test_rmse))  
        textfile.write("\n Output Variance: "+ str(np.mean(sigma_)))                   
        textfile.write("\n---------------------------------")
        if BIM_adversarial:            
            textfile.write('\n BIM Adversarial Noise epsilon: '+ str(epsilon )) 
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))   
            textfile.write("\n Alpha: "+ str(alpha)) 
            textfile.write("\n Maximum number of iterations: "+ str(maxAdvStep))             
        textfile.write("\n---------------------------------")    
        textfile.close() 
if __name__ == '__main__':
    main_function()