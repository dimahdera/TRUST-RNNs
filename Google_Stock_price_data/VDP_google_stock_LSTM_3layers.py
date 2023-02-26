from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import seaborn as sns # used for plot interactive graph.
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.utils import shuffle

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
import wandb
import xlsxwriter
from scipy import stats
#os.environ["WANDB_API_KEY"] = ""
## Data can be downloaded from: http://archive.ics.uci.edu/ml/machine-learning-databases/00235/
## Just open the zip file and grab the file 'household_power_consumption.txt' put it in the directory
## that you would like to run the code.

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

def sigma_regularizer1(x):
    input_size = 1.
    f_s = tf.math.softplus(x) #tf.clip_by_value(t=x, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) )  # tf.math.log(1. + tf.math.exp(x))
   # f_s = tf.where(tf.math.is_nan(f_s), tf.constant(0.01, shape=f_s.shape), f_s)
    #f_s = tf.where(tf.math.is_inf(f_s), tf.constant(0.01, shape=f_s.shape), f_s)
    return -input_size*tf.reduce_mean(1. + tf.math.log(f_s) - f_s, axis=-1)

def sigma_regularizer2(x):
    f_s = tf.math.softplus(x )#tf.clip_by_value(t=x, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) )  # tf.math.log(1. + tf.math.exp(x))
  #  f_s = tf.where(tf.math.is_nan(f_s), tf.constant(0.01, shape=f_s.shape), f_s)
  #  f_s = tf.where(tf.math.is_inf(f_s), tf.constant(0.01, shape=f_s.shape), f_s)
    return -tf.reduce_mean(1. + tf.math.log(f_s) - f_s, axis=-1)
    

class densityPropLSTMCell_first(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units]), tf.TensorShape([units, units]),
                           tf.TensorShape([units, units])]
        super(densityPropLSTMCell_first, self).__init__(**kwargs)
    def build(self, input_shape):        
        input_size = input_shape[-1]        
        ini_sigma = -4.6
        #min_sigma = -2.2
        init_mu = 0.05
        seed_ = None
        tau1 = 1./ input_size
        tau2 = 1./self.units
        # Forget Gate
        self.U_f =self.add_weight(name='U_f', shape=(input_size, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_),  trainable=True, regularizer=tf.keras.regularizers.l2(tau1))
        self.uf_sigma = self.add_weight(name='uf_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_f =self.add_weight(name='W_f', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau2))
        self.wf_sigma = self.add_weight(name='wf_sigma', shape=(self.units,),initializer=tf.constant_initializer(ini_sigma), trainable=True,regularizer=sigma_regularizer2)
        # Input Gate
        self.U_i = self.add_weight(name='U_i', shape=(input_size, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True,regularizer=tf.keras.regularizers.l2(tau1))
        self.ui_sigma = self.add_weight(name='ui_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_i = self.add_weight(name='W_i', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_),trainable=True, regularizer=tf.keras.regularizers.l2(tau2))
        self.wi_sigma = self.add_weight(name='wi_sigma', shape=(self.units,),initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer2)
        # Output Gate
        self.U_o = self.add_weight(name='U_o', shape=(input_size, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau1))
        self.uo_sigma = self.add_weight(name='uo_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_o = self.add_weight(name='W_o', shape=(self.units, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau2))
        self.wo_sigma = self.add_weight(name='wo_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer2)
        # Gate Gate
        self.U_g =self.add_weight(name='U_g', shape=(input_size, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau1))
        self.ug_sigma = self.add_weight(name='ug_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_g = self.add_weight(name='W_g', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True,regularizer=tf.keras.regularizers.l2(tau2))
        self.wg_sigma = self.add_weight(name='wg_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True,regularizer=sigma_regularizer2)
        self.built = True 
    def call(self, inputs, states):
        # state should be in [(batch, units), (batch, units, units)], mean vector and covaraince matrix
        prev_state, prev_istate, Sigma_state, Sigma_istate = states

        ## Forget Gate
        f = tf.sigmoid(tf.matmul(inputs, self.U_f) + tf.matmul(prev_state, self.W_f))        
        Uf_Sigma = tf.linalg.diag(tf.math.softplus(self.uf_sigma ))#tf.clip_by_value(t=self.uf_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        
        Sigma_Uf = x_Sigma_w_x_T(inputs, Uf_Sigma)
        ################
        Wf_Sigma = tf.linalg.diag(tf.math.softplus(self.wf_sigma ))#tf.clip_by_value(t=self.wf_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
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
        Ui_Sigma = tf.linalg.diag(tf.math.softplus(self.ui_sigma ))#tf.clip_by_value(t=self.ui_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_Ui = x_Sigma_w_x_T(inputs, Ui_Sigma)
        ################
        Wi_Sigma = tf.linalg.diag(tf.math.softplus(self.wi_sigma ))#tf.clip_by_value(t=self.wi_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
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
        Uo_Sigma = tf.linalg.diag(tf.math.softplus(self.uo_sigma ))#tf.clip_by_value(t=self.uo_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_Uo = x_Sigma_w_x_T(inputs, Uo_Sigma)
        ################
        Wo_Sigma = tf.linalg.diag(tf.math.softplus(self.wo_sigma ))#tf.clip_by_value(t=self.wo_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
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
        Ug_Sigma = tf.linalg.diag(tf.math.softplus(self.ug_sigma ))#tf.clip_by_value(t=self.ug_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2))  ))
        Sigma_Ug = x_Sigma_w_x_T(inputs, Ug_Sigma)
        ################
        Wg_Sigma = tf.linalg.diag(tf.math.softplus(self.wg_sigma ))#tf.clip_by_value(t=self.wg_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
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
        output = (mu_out,Sigma_out)
        new_state = (mu_out, c, Sigma_out, Sigma_out_c)
        return output, new_state
        
        
class densityPropLSTMCell_second(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units]), tf.TensorShape([units, units]),
                           tf.TensorShape([units, units])]
        super(densityPropLSTMCell_second, self).__init__(**kwargs)
    def build(self, input_shape):
        mu_shape, sigma_shape = input_shape        
        input_size = mu_shape[-1] 
      #  print(  input_size)     
        ini_sigma = -4.6
        #min_sigma = -2.2
        init_mu = 0.05
        seed_ = None
        tau1 = 1./ input_size
        tau2 = 1./self.units
        # Forget Gate
        self.U_f =self.add_weight(name='U_f', shape=(input_size, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_),  trainable=True, regularizer=tf.keras.regularizers.l2(tau1))
        self.uf_sigma = self.add_weight(name='uf_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_f =self.add_weight(name='W_f', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau2))
        self.wf_sigma = self.add_weight(name='wf_sigma', shape=(self.units,),initializer=tf.constant_initializer(ini_sigma), trainable=True,regularizer=sigma_regularizer2)
        # Input Gate
        self.U_i = self.add_weight(name='U_i', shape=(input_size, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True,regularizer=tf.keras.regularizers.l2(tau1))
        self.ui_sigma = self.add_weight(name='ui_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_i = self.add_weight(name='W_i', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_),trainable=True, regularizer=tf.keras.regularizers.l2(tau2))
        self.wi_sigma = self.add_weight(name='wi_sigma', shape=(self.units,),initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer2)
        # Output Gate
        self.U_o = self.add_weight(name='U_o', shape=(input_size, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau1))
        self.uo_sigma = self.add_weight(name='uo_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_o = self.add_weight(name='W_o', shape=(self.units, self.units),initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau2))
        self.wo_sigma = self.add_weight(name='wo_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer2)
        # Gate Gate
        self.U_g =self.add_weight(name='U_g', shape=(input_size, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True, regularizer=tf.keras.regularizers.l2(tau1))
        self.ug_sigma = self.add_weight(name='ug_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True, regularizer=sigma_regularizer1)
        self.W_g = self.add_weight(name='W_g', shape=(self.units, self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=init_mu, seed=seed_), trainable=True,regularizer=tf.keras.regularizers.l2(tau2))
        self.wg_sigma = self.add_weight(name='wg_sigma', shape=(self.units,), initializer=tf.constant_initializer(ini_sigma), trainable=True,regularizer=sigma_regularizer2)
        self.built = True 
    def call(self, inputs, states):
        # state should be in [(batch, units), (batch, units, units)], mean vector and covaraince matrix
        prev_state, prev_istate, Sigma_state, Sigma_istate = states
        mu_inputs, sigma_inputs = inputs
        #mu_inputs = tf.squeeze(mu_inputs)
        ## Forget Gate
        f = tf.sigmoid(tf.matmul(mu_inputs, self.U_f) + tf.matmul(prev_state, self.W_f))
        Uf_Sigma = tf.linalg.diag(tf.math.softplus( self.uf_sigma))#tf.clip_by_value(t=self.uf_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2))  ))
        Sigma_Uf1 = w_t_Sigma_i_w(self.U_f, sigma_inputs)
        Sigma_Uf2 = x_Sigma_w_x_T(mu_inputs, Uf_Sigma)
        Sigma_Uf3 = tr_Sigma_w_Sigma_in(sigma_inputs, Uf_Sigma)       
        Sigma_Uf = Sigma_Uf1 + Sigma_Uf2 + Sigma_Uf3
        ################
        Wf_Sigma = tf.linalg.diag(tf.math.softplus(self.wf_sigma))# tf.clip_by_value(t= self.wf_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_f1 = w_t_Sigma_i_w(self.W_f, Sigma_state)
        Sigma_f2 = x_Sigma_w_x_T(prev_state, Wf_Sigma)
        Sigma_f3 = tr_Sigma_w_Sigma_in(Sigma_state, Wf_Sigma)
        Sigma_out_ff = Sigma_f1 + Sigma_f2 + Sigma_f3 + Sigma_Uf
        ################
        gradi_f = grad_sigmoid(tf.matmul(mu_inputs, self.U_f) + tf.matmul(prev_state, self.W_f))
        Sigma_out_f = activation_Sigma(gradi_f, Sigma_out_ff)
        ###################################
        ###################################
        ## Input Gate
        i = tf.sigmoid(tf.matmul(mu_inputs, self.U_i) + tf.matmul(prev_state, self.W_i))
        Ui_Sigma = tf.linalg.diag(tf.math.softplus(self.ui_sigma))# tf.clip_by_value(t=self.ui_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_Ui1 = w_t_Sigma_i_w(self.U_i, sigma_inputs)
        Sigma_Ui2 = x_Sigma_w_x_T(mu_inputs, Ui_Sigma)
        Sigma_Ui3 = tr_Sigma_w_Sigma_in(sigma_inputs, Ui_Sigma)       
        Sigma_Ui = Sigma_Ui1 + Sigma_Ui2 + Sigma_Ui3      
        ################
        Wi_Sigma = tf.linalg.diag(tf.math.softplus( self.wi_sigma))#tf.clip_by_value(t=self.wi_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_i1 = w_t_Sigma_i_w(self.W_i, Sigma_state)
        Sigma_i2 = x_Sigma_w_x_T(prev_state, Wi_Sigma)
        Sigma_i3 = tr_Sigma_w_Sigma_in(Sigma_state, Wi_Sigma)
        Sigma_out_ii = Sigma_i1 + Sigma_i2 + Sigma_i3 + Sigma_Ui
        ################
        gradi_i = grad_sigmoid(tf.matmul(mu_inputs, self.U_i) + tf.matmul(prev_state, self.W_i))
        Sigma_out_i = activation_Sigma(gradi_i, Sigma_out_ii)
        ###################################
        ###################################
        ## Output Gate
        o = tf.sigmoid(tf.matmul(mu_inputs, self.U_o) + tf.matmul(prev_state, self.W_o))
        Uo_Sigma = tf.linalg.diag(tf.math.softplus(self.uo_sigma))# tf.clip_by_value(t=self.uo_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_Uo1 = w_t_Sigma_i_w(self.U_o, sigma_inputs)
        Sigma_Uo2 = x_Sigma_w_x_T(mu_inputs, Uo_Sigma)
        Sigma_Uo3 = tr_Sigma_w_Sigma_in(sigma_inputs, Uo_Sigma)       
        Sigma_Uo = Sigma_Uo1 + Sigma_Uo2 + Sigma_Uo3       
        ################
        Wo_Sigma = tf.linalg.diag(tf.math.softplus(self.wo_sigma))# tf.clip_by_value(t=self.wo_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_o1 = w_t_Sigma_i_w(self.W_o, Sigma_state)
        Sigma_o2 = x_Sigma_w_x_T(prev_state, Wo_Sigma)
        Sigma_o3 = tr_Sigma_w_Sigma_in(Sigma_state, Wo_Sigma)
        Sigma_out_oo = Sigma_o1 + Sigma_o2 + Sigma_o3 + Sigma_Uo
        ################
        gradi_o = grad_sigmoid(tf.matmul(mu_inputs, self.U_o) + tf.matmul(prev_state, self.W_o))
        Sigma_out_o = activation_Sigma(gradi_o, Sigma_out_oo)
        ###################################
        ###################################
        ## Gate Gate
        g = tf.tanh(tf.matmul(mu_inputs, self.U_g) + tf.matmul(prev_state, self.W_g))
        Ug_Sigma = tf.linalg.diag(tf.math.softplus(self.ug_sigma))# tf.clip_by_value(t=self.ug_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_Ug1 = w_t_Sigma_i_w(self.U_g, sigma_inputs)
        Sigma_Ug2 = x_Sigma_w_x_T(mu_inputs, Ug_Sigma)
        Sigma_Ug3 = tr_Sigma_w_Sigma_in(sigma_inputs, Ug_Sigma)       
        Sigma_Ug = Sigma_Ug1 + Sigma_Ug2 + Sigma_Ug3      
        ################
        Wg_Sigma = tf.linalg.diag(tf.math.softplus( self.wg_sigma ))#tf.clip_by_value(t=self.wg_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) ))
        Sigma_g1 = w_t_Sigma_i_w(self.W_g, Sigma_state)
        Sigma_g2 = x_Sigma_w_x_T(prev_state, Wg_Sigma)
        Sigma_g3 = tr_Sigma_w_Sigma_in(Sigma_state, Wg_Sigma)
        Sigma_out_gg = Sigma_g1 + Sigma_g2 + Sigma_g3 + Sigma_Ug
        ################
        gradi_g = grad_tanh(tf.matmul(mu_inputs, self.U_g) + tf.matmul(prev_state, self.W_g))
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
        output = (mu_out,Sigma_out)
        new_state = (mu_out, c, Sigma_out, Sigma_out_c)
        return output, new_state        
                
class VDP_Dense(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units):
        super(VDP_Dense, self).__init__()
        self.units = units
    def build(self, input_shape):
        ini_sigma = -4.6
      #  min_sigma = -6.9
        tau = 1./input_shape[-1]
       # print(input_shape[-1])
        self.w_mu = self.add_weight(name='w_mu', shape=(input_shape[-1], self.units), initializer=tf.random_normal_initializer(mean=0.0, stddev=.05 , seed=None), regularizer=tf.keras.regularizers.l2(tau),#l1_l2(l1=tau, l2=tau),
                                    trainable=True, )
        self.w_sigma = self.add_weight(name='w_sigma',shape=(self.units,),
                                       initializer= tf.constant_initializer(ini_sigma), #tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,seed=None),
                                       regularizer=sigma_regularizer2, #  
                                       trainable=True, ) 
    def call(self, mu_in, Sigma_in):
        mu_out = tf.matmul(mu_in, self.w_mu)  
        W_sigma = tf.math.softplus(self.w_sigma  )#tf.clip_by_value(t=self.w_sigma, clip_value_min=tf.constant(-1.0e+3), clip_value_max=tf.constant(1.0e+2)) )       
        Sigma_1 = w_t_Sigma_i_w(self.w_mu, Sigma_in)  # shape=[batch_size]
        Sigma_2 = tf.expand_dims(tf.multiply(tf.reduce_sum(tf.multiply(mu_in, mu_in), axis=1, keepdims=True), W_sigma), axis=2)  # shape=[batch_size]
        Sigma_3 = tf.expand_dims(tf.expand_dims(tf.multiply(tf.linalg.trace(Sigma_in), W_sigma), axis=1), axis=2)  # shape=[batch_size]
        Sigma_out = (Sigma_1 + Sigma_2 + Sigma_3)  # shape=[batch_size]
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        return mu_out, Sigma_out

class VDPDropout(keras.layers.Layer):
    def __init__(self, drop_prop):
        super(VDPDropout, self).__init__()
        self.drop_prop = drop_prop

    def call(self, mu_in, Sigma_in, Training=True):
       # batch_size = mu_in.shape[0]  # shape=[batch_size,1, hidden_size]
        hidden_size = mu_in.shape[-1]        
        scale_sigma = 1.0/(1 - self.drop_prop)
        
        if Training:        
           mu_out = tf.nn.dropout(mu_in, rate=self.drop_prop)   
           non_zero = tf.not_equal(tf.squeeze(mu_out), tf.zeros_like(tf.squeeze(mu_out)))           
           non_zero_sigma1 = tf.tile(tf.expand_dims(non_zero , -1), [1, 1 ,hidden_size])
           non_zero_sigma2 = tf.tile(tf.expand_dims(non_zero , 1), [1, hidden_size ,1])
           non_zero_sigma = tf.math.logical_and(non_zero_sigma1, non_zero_sigma2)
           non_zero_sigma_mask = tf.boolean_mask(Sigma_in, non_zero_sigma)
           idx_sigma = tf.dtypes.cast(tf.where(non_zero_sigma), tf.int32)
           Sigma_out = (scale_sigma ** 2) * tf.scatter_nd(idx_sigma, non_zero_sigma_mask, tf.shape(non_zero_sigma))          
        else:
           mu_out = mu_in
           Sigma_out = Sigma_in         
        return mu_out, Sigma_out

class Density_prop_LSTM(tf.keras.Model):
  def __init__(self, units, name=None): 
    super(Density_prop_LSTM, self).__init__()
    self.units = units
   # self.drop_prop = drop_prop
    self.cell1 = densityPropLSTMCell_first(self.units)
    self.cell2 = densityPropLSTMCell_second(self.units)
    self.cell3 = densityPropLSTMCell_second(self.units)
    self.rnn1 = tf.keras.layers.RNN(self.cell1, return_state=True, return_sequences=True, stateful=False)
    self.rnn2 = tf.keras.layers.RNN(self.cell2, return_state=True, return_sequences=True, stateful=False)
    self.rnn3 = tf.keras.layers.RNN(self.cell3, return_state=True, return_sequences=False, stateful=False)    
    self.linear_1 = VDP_Dense(1)
   # self.dropout_1 = VDPDropout(self.drop_prop) 
  def call(self, inputs, training=True):   
   # h_mu = tf.convert_to_tensor(np.zeros((inputs.shape[0], self.units,)).astype(np.float32))
   # h_sigma = tf.convert_to_tensor(np.random.random_sample((inputs.shape[0], self.units, self.units)).astype(np.float32)) 
    
    xx1 = self.rnn1(inputs)#, initial_state=[h_mu, h_mu, h_sigma, h_sigma ])
    x, mu_state, c_state, sigma_state, sigma_cstate = xx1
    x_mu, x_sigma = x
   # print(x.shape)
    #print(sigma_state.shape)
 #   x, sigma_state = self.dropout_1(x, sigma_state, Training=training) 
    #sigma_state = tf.expand_dims( sigma_state, axis=1 )
    
    xx_in2 = (x_mu, x_sigma)
    xx2 = self.rnn2(xx_in2)#, initial_state=[h_mu, h_mu, h_sigma, h_sigma ])
    x, mu_state, c_state, sigma_state, sigma_cstate = xx2
    x_mu, x_sigma = x
  #  x, sigma_state = self.dropout_1(x, sigma_state, Training=training) 
   # sigma_state = tf.expand_dims( sigma_state, axis=1 )
    
    xx_in3 = (x_mu, x_sigma)
    xx3 = self.rnn3(xx_in3)#, initial_state=[h_mu, h_mu, h_sigma, h_sigma ])
    x, mu_state, c_state, sigma_state, sigma_cstate = xx3 
    x_mu, x_sigma = x
  #  x, sigma_state = self.dropout_1(x, sigma_state, Training=training) 
         
    outputs, Sigma = self.linear_1(x_mu, x_sigma)   
    Sigma = tf.where(tf.math.is_nan(Sigma), tf.zeros_like(Sigma), Sigma)
    Sigma = tf.where(tf.math.is_inf(Sigma), tf.zeros_like(Sigma), Sigma)
    return outputs, Sigma  
    
      
def mean_square_loss(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred) 
   return tf.math.reduce_mean(squared_difference)
   
def nll_gaussian(y_test, y_pred_mean, y_pred_sd):
    loss1 = tf.math.reduce_mean(tf.math.divide_no_nan(tf.square(y_test - y_pred_mean) , y_pred_sd)  )
    loss2 = tf.math.reduce_mean(tf.math.log(y_pred_sd ) )
    loss2 =  tf.where(tf.math.is_nan(loss2), tf.zeros_like(loss2), loss2)
    loss = tf.reduce_mean(tf.math.add(loss1, loss2 )) 
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    #loss = tf.clip_by_value(t=loss, clip_value_min=tf.constant(-1.0e+4), clip_value_max=tf.constant(1.0e+1))
    return loss
    
       
def main_function(time_step=60, input_dim=2, units=30, output_size=1, batch_size=50, epochs=200, lr=0.001, lr_end = 0.00001, kl_factor=0.0001,
                  Random_noise=False, gaussain_noise_std=0.5, Adversarial_noise=False, epsilon=0.5,  
                  BIM_adversarial=False, alpha= 1, maxAdvStep=100,  
                  Training=False, Testing=False, continue_training=False, saved_model_epochs=50): 
                  
    PATH = './google_stock/saved_models_with_hidden_unit_{}_lr_{}/VDP_lstm_epoch_{}/'.format(units, lr, epochs)    
    stock_data = pd.read_csv("./Dataset/Google_stock_data.csv",dtype={'Close': 'float64', 'Volume': 'int64','Open': 'float64','High': 'float64', 'Low': 'float64'})
    stock_data.columns = ['date', 'close/last', 'volume', 'open', 'high', 'low']        
    #create a new column "average" 
    stock_data['average'] = (stock_data['high'] + stock_data['low'])/2
    #pick the input features (average and volume)
    input_feature= stock_data.iloc[:,[2,6]].values
    input_data = input_feature
    
    #data normalization
    sc= MinMaxScaler(feature_range=(0,1))
    input_data[:,0:2] = sc.fit_transform(input_feature[:,:])    
    # data preparation
    lookback = 60
    
    test_size = int(.3 * len(stock_data))
    X = []
    y = []
    for i in range(len(stock_data) - lookback - 1):
        t = []
        for j in range(0, lookback):
            t.append(input_data[[(i + j)], :])
        X.append(t)
        y.append(input_data[i + lookback, 1])
    
    
    X, y= np.array(X), np.array(y)
    X = X.reshape(X.shape[0],lookback, 2)
    X, y = shuffle(X, y, random_state=0)
    X_train = X[test_size+lookback-8:,:,:]
    X_test = X[:test_size+lookback-1,:,:]
    y_train = y[test_size+lookback-8:]
    y_test = y[:test_size+lookback-1]
    
    train_size = X_train.shape[0] #1150
    test_size = X_test.shape[0] # 600
    
    train_X = X_train.astype('float32')
    test_X = X_test.astype('float32')   
    train_y = y_train.astype('float32')
    test_y = y_test.astype('float32')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].
    tr_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(batch_size)
    
    
    num_train_steps = epochs * int(train_X.shape[0] /batch_size)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,  decay_steps=num_train_steps,  end_learning_rate=lr_end, power=10.)    
    lstm_model = Density_prop_LSTM(units, name = 'vdp_lstm')  
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn , clipnorm= 1.)     
    
    @tf.function()# Make it fast.       
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
          logits, sigma = lstm_model(x, training=True)
          lstm_model.trainable = True  
          logits = tf.squeeze(logits )
          sigma = tf.squeeze(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1.0e-4), clip_value_max=tf.constant(1.0e+4)))
          loss_final = nll_gaussian(y, logits, sigma ) #tf.math.reduce_mean(tf.math.divide_no_nan(mse(y, tf.expand_dims(logits, axis=2 )), (sigma**2)) )+ tf.math.reduce_mean(tf.math.log(sigma**2 +1.0e-4))           
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
               
        return loss, logits, sigma, gradients  , regularization_loss, loss_final
        
            
    @tf.function()# Make it fast.     
    def valid_on_batch(x, y):          
        logits, sigma = lstm_model(x, training=False)
        lstm_model.trainable = False 
        logits = tf.squeeze(logits )
        sigma = tf.squeeze(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1.0e-4), clip_value_max=tf.constant(1.0e+4))  )          
        loss_final = nll_gaussian(y, logits, sigma ) #tf.math.reduce_mean(tf.math.divide_no_nan(mse(y, tf.expand_dims(logits, axis=2 )), (sigma**2)) )+ tf.math.reduce_mean(tf.math.log(sigma**2 +1.0e-4))               
        regularization_loss=tf.math.add_n(lstm_model.losses)
        loss = 0.5 * (loss_final  + kl_factor*regularization_loss )        
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)        
        return loss, logits, sigma, regularization_loss, loss_final 
        
    @tf.function
    def test_on_batch(x):  
        lstm_model.trainable = False                    
        mu_out, sigma = lstm_model(x, training=False)            
        return tf.squeeze(mu_out), tf.squeeze(sigma)
    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
       #     lstm_model.trainable = False 
            prediction, sigma = lstm_model(input_image, training=False) 
            loss_final = nll_gaussian(input_label, tf.squeeze(prediction),  tf.squeeze(tf.clip_by_value(t=tf.math.abs(sigma), clip_value_min=tf.constant(1.0e-3),
                                   clip_value_max=tf.constant(1.0e+12)))) 
            regularization_loss=tf.math.add_n(lstm_model.losses)
            loss = 0.5 * (loss_final  + kl_factor*regularization_loss )
                                                         
           # loss = 0.5 * loss_final 
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
      #  wandb.init(entity = "", project="VDP_LSTM_google_stock_timeseries_units_{}_epochs_{}_lr_{}_3lstm2_exp2_no_drop".format(units, epochs, lr)) 
        if continue_training:
            saved_model_path = './google_stock/saved_models_with_hidden_unit_{}_lr_{}/VDP_lstm_epoch_{}/'.format(units,lr,  saved_model_epochs)
            lstm_model.load_weights(saved_model_path + 'vdp_lstm_model')
        train_rmse = np.zeros(epochs)
        Scaled_train_rmse = np.zeros(epochs)
        valid_rmse = np.zeros(epochs)
        Scaled_valid_rmse = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs):
            print('Epoch: ', epoch + 1, '/', epochs)
            rmse1 = 0            
            rmse_valid1 = 0            
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0
            # -------------Training--------------------            
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step / int(train_X.shape[0] / (batch_size)))
                if (x.shape[0] == batch_size):                    
                    loss, logits, sigma, gradients, regularization_loss, loss_final = train_on_batch(x, y)  
                   # test_loss.update_state(loss)                
                    err1 += loss.numpy()#test_loss.result()                                     
                    rmse_err = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - logits)))  
                    rmse1 += rmse_err                     
                    tr_no_steps += 1
                    if step % 10 == 0:                        
                        print("Normalized Training RMSE: %.6f" % float(rmse1 / (tr_no_steps )))                        
                   # wandb.log({"Total Training Loss": loss.numpy(),
                   #            "Normalized Training RMSE per minibatch": rmse_err,                                
                   #            "gradient per minibatch": np.mean(gradients[0].numpy()),                                
                   #            "Average Variance value": tf.reduce_mean(sigma).numpy(),
                   #            "Regularization_loss": regularization_loss.numpy(),                               
                   #            "Log-Likelihood Loss": np.mean(loss_final.numpy()),                                                     
                   #            'epoch': epoch
                   # })           
            train_rmse[epoch] = rmse1 / tr_no_steps             
            train_err[epoch] = err1 / tr_no_steps
            print('Training RMSE  ', train_rmse[epoch])            
            print('Training Loss  ', train_err[epoch])                      
            # ---------------Validation----------------------           
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(test_X.shape[0] / (batch_size)))
                if (x.shape[0] == batch_size):                    
                    total_vloss, logits, sigma, regularization_loss, vloss = valid_on_batch(x, y)               
                    err_valid1 += total_vloss.numpy()# test_loss.result().numpy()                                        
                    valid_rmse_err = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - logits)))                
                    rmse_valid1 += valid_rmse_err     
                                      
                    if step % 10 == 0:                       
                        print(" validation RMSE so far: %.6f" % valid_rmse_err)
                    va_no_steps += 1
               #     wandb.log({"Total Validation Loss": total_vloss.numpy(),
               #                "Normalized Validation RMSE per minibatch": valid_rmse_err ,                                                            
               #                "Average Variance value (validation Set)": tf.reduce_mean(sigma).numpy(),
               #                "Regularization_loss (validation Set)": regularization_loss.numpy(),                               
               #                "Log-Likelihood Loss (validation Set)": np.mean(vloss.numpy() )                                                                          
               #                 })                                
                    lstm_model.save_weights(PATH + 'vdp_lstm_model') 
          #  wandb.log({"Training Loss": (err1 / tr_no_steps),
          #              "Training RMSE": (rmse1 / tr_no_steps),                                          
          #              "Validation Loss": (err_valid1 / va_no_steps),                       
          #              "Validation RMSE": (rmse_valid1 / va_no_steps),
          #              'epoch': epoch
          3             })
            valid_rmse[epoch] = rmse_valid1 / va_no_steps            
            valid_error[epoch] = err_valid1 / va_no_steps
            stop = timeit.default_timer()
            print('Total Training Time: ', stop - start)
            print('Training RMSE  ', train_rmse[epoch])
            print('Validation RMSE  ', valid_rmse[epoch])           
            print('------------------------------------')
            print('Training Loss  ', train_err[epoch])
            print('Validation Loss  ', valid_error[epoch])           
            # -----------------End Training--------------------------
            lstm_model.save_weights(PATH + 'vdp_lstm_model')
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_rmse, 'b', label='Training RMSE')
            plt.plot(valid_rmse,'r' , label='Validation RMSE')
           # plt.ylim(0, 1.1)
            plt.title("Density Propagation LSTM on Google Stock time series Data")
            plt.xlabel("Epochs")
            plt.ylabel("Root Mean square Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_LSTM_on_time_series_Data_rmse.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training Loss')
            plt.plot(valid_error,'r' , label='Validation Loss')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation LSTM on Google Stock time series Data")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_LSTM_on_time_series_Data_loss.png')
            plt.close(fig)
        
        f = open(PATH + 'training_validation_rmse_loss.pkl', 'wb')         
        pickle.dump([train_rmse, valid_rmse, train_err, valid_error], f)                                                   
        f.close()                  
             
        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs)) 
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))  
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))                              
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Training  RMSE : "+ str( train_rmse))
                textfile.write("\n Validation RMSE : "+ str(valid_rmse ))
                    
                textfile.write("\n Training  Loss : "+ str( train_err))
                textfile.write("\n Validation Loss : "+ str(valid_error ))
            else:
                textfile.write("\n Training  RMSE : "+ str(np.mean(train_rmse[epoch])))
                textfile.write("\n Validation RMSE : "+ str(np.mean(valid_rmse[epoch])))
                
                textfile.write("\n Training  Loss : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Validation Loss : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    if(Testing):
        test_path = 'test_results_latest/'
        if Random_noise:
            test_path = 'test_random_noise_{}/'.format(gaussain_noise_std)               
        os.makedirs(PATH + test_path) 
        lstm_model.load_weights(PATH + 'vdp_lstm_model')
        test_no_steps = 0        
        rmse_test = np.zeros(int(test_X.shape[0] /batch_size))        
        true_x = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size])        
        logits_ = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size])
        sigma_ = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size])
        test_start = timeit.default_timer()      
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(test_X.shape[0] / (batch_size)) ) 
            if (x.shape[0] == batch_size):            
                true_x[test_no_steps, :, :, :] = x.numpy()  
                true_y[test_no_steps, :] = y.numpy()  
                if Random_noise:
                    noise = tf.random.normal(shape = [batch_size, time_step, input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                    x = x +  noise         
                logits, sigma = test_on_batch(x)  
                logits_[test_no_steps,:] = logits.numpy()                            
                sigma_[test_no_steps, :]= sigma.numpy()             
                rmse_ = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - logits)))                   
                rmse_test[test_no_steps] = rmse_.numpy()                      
                if step % 10 == 0:
                    print("Test RMSE: %.6f" % rmse_.numpy()   )             
                test_no_steps+=1       
        test_stop = timeit.default_timer()
        print('Total Test Time: ', test_stop - test_start) 
        test_rmse = np.mean(rmse_test)       
        print('Average Test RMSE : ', test_rmse )          
        
        pf = open(PATH + test_path + 'Uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_ ,true_y,  sigma_ , rmse_test], pf)                                          
        pf.close()        
        
        if Random_noise:
            snr_signal = np.zeros([int(test_X.shape[0]/(batch_size)) ,batch_size])
            for i in range(int(test_X.shape[0]/(batch_size))):
                for j in range(batch_size):
                    noise = tf.random.normal(shape = [time_step, input_dim ], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ).numpy()   
                    snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/np.sum( np.square(noise) ))
            print('SNR', np.mean(snr_signal))
         
        #print('Output Variance', np.mean(np.abs(sigma_))  )
        
        sigma_1 = np.reshape(sigma_, int(test_X.shape[0]/(batch_size))* batch_size)
#        var = np.zeros([int(test_X.shape[0]/(batch_size))* batch_size])        
#        for i in range(int(test_X.shape[0]/(batch_size))*batch_size):
#            s = np.abs(sigma_1[i])
#            if (i != 0):
#               if(np.abs(s)> 10000 ):
#                    var[i] = 0.0#np.abs(sigma_1[i-1])
#               else:
#                    var[i] = s
#            else:
#               var[i] = s
#        data_mean, data_std = np.mean(np.abs(sigma_1)), np.std(np.abs(sigma_1))
#       # identify outliers
#        cut_off = data_std * 3
#        lower, upper = data_mean - cut_off, data_mean + cut_off
#        outliers = [x for x in np.abs(sigma_1) if x < lower or x > upper]
#        outliers_removed = [x for x in np.abs(sigma_1) if x > lower and x < upper]
#        print('outliers_removed' , np.mean(outliers_removed))

        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(sigma_1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()             
      #  print('Output Variance without outlier', np.mean(np.abs(var)) ) 
        print('Output Variance', np.mean(np.abs(sigma_)))
#        print('Trimmed mean of the Output Variance', stats.trim_mean(np.abs(sigma_), 0.011, axis=None))
#        print('Trimmed mean of the Output Variance', stats.trim_mean(np.abs(sigma_), 0.01, axis=None))       
        
#        sigma_1 = np.reshape(sigma_, int(test_X.shape[0]/(batch_size))* batch_size)
#        var = np.zeros([int(test_X.shape[0]/(batch_size))* batch_size])        
#        for i in range(int(test_X.shape[0]/(batch_size))*batch_size):
#            s = sigma_1[i]
#            if (i != 0):
#               if(np.abs(var[i-1] - s)> 10000 ):
#                    var[i] = sigma_1[i-1]
#               else:
#                    var[i] = s
#            else:
#               var[i] = s
                     
        #print('Output Variance without outlier', np.mean(var))  
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Learning rate : ' +str(lr)) 
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))  
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n KL term factor : ' +str(kl_factor))      
        textfile.write("\n---------------------------------") 
        textfile.write('\n Total test time in sec : ' +str(test_stop - test_start))        
        textfile.write("\n Test RMSE : "+ str( test_rmse))         
        textfile.write("\n Output Variance: "+ str(np.mean(sigma_)))                 
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std ))   
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))                
        textfile.write("\n---------------------------------")    
        textfile.close()
        
    if(Adversarial_noise):        
        test_path = 'test_adversarial_noise_{}/'.format(epsilon)  
        os.makedirs(PATH + test_path) 
        lstm_model.load_weights(PATH + 'vdp_lstm_model')
        lstm_model.trainable = False        
               
        test_no_steps = 0        
        rmse_test = np.zeros(int(test_X.shape[0] /batch_size))                              
        true_x = np.zeros([int(test_X.shape[0]/(batch_size)), batch_size, time_step, input_dim])
        adv_perturbations = np.zeros([int(test_X.shape[0]/(batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size])        
        logits_ = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size]) 
        sigma_ = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size ]) 
        adv_test_start = timeit.default_timer()           
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(test_X.shape[0] / (batch_size)) ) 
            if (x.shape[0] == batch_size):                                 
                true_x[test_no_steps, :, :, :] = x.numpy() 
                true_y[test_no_steps, :] = y.numpy()
                adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y)
               # print(adv_perturbations[test_no_steps, :, :, :])
                adv_x = x + epsilon*adv_perturbations[test_no_steps, :, :, :] 
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)               
            
                logits, sigma = test_on_batch(adv_x)              
                logits_[test_no_steps,:] = logits.numpy()  
                sigma_[test_no_steps, :]= sigma.numpy()                                 
                rmse_ = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - logits)))               
                rmse_test[test_no_steps] = rmse_.numpy()                
                
                if step % 10 == 0:
                    print("Test RMSE: %.6f" % rmse_.numpy() )             
                test_no_steps+=1       
        adv_test_stop = timeit.default_timer()
        print('Total Test Time: ', adv_test_stop - adv_test_start) 
        test_rmse = np.mean(rmse_test)      
        print('Average Test RMSE : ', test_rmse )        
        
        pf = open(PATH + test_path + 'Uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_ ,true_y, sigma_ , rmse_test ], pf)                                         
        pf.close()
        
        snr_signal = np.zeros([int(test_X.shape[0]/(batch_size)) ,batch_size])
        for i in range(int(test_X.shape[0]/batch_size)):
            for j in range(batch_size):               
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/np.sum( np.square(true_x[i,j,:, :] - np.clip(true_x[i,j,:, :]+epsilon*adv_perturbations[i, j, :, :], 0.0, 1.0)  ) ))              
                
        sigma_1 = np.reshape(sigma_, int(test_X.shape[0]/(batch_size))* batch_size)
#        var = np.zeros([int(test_X.shape[0]/(batch_size))* batch_size])        
#        for i in range(int(test_X.shape[0]/(batch_size))*batch_size):
#            s = np.abs(sigma_1[i])
#            if (i != 0):
#               if(np.abs(s)> 10000 ):
#                    var[i] = np.abs(sigma_1[i-1])
#               else:
#                    var[i] = s
#            else:
#               var[i] = s
#        data_mean, data_std = np.mean(np.abs(sigma_1)), np.std(np.abs(sigma_1))
#       # identify outliers
#        cut_off = data_std * 3
#        lower, upper = data_mean - cut_off, data_mean + cut_off
#        outliers = [x for x in np.abs(sigma_1) if x < lower or x > upper]
#        outliers_removed = [x for x in np.abs(sigma_1) if x > lower and x < upper]
#        print('outliers_removed' , np.mean(outliers_removed))

        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(sigma_1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()             
       # print('Output Variance without outlier', np.mean(np.abs(var)) ) 
        print('Output Variance', np.mean(np.abs(sigma_)))
#        print('Trimmed mean of the Output Variance', stats.trim_mean(np.abs(sigma_1), 0.011, axis=None))
#        print('Trimmed mean of the Output Variance', stats.trim_mean(np.abs(sigma_1), 0.01, axis=None))  
#        print('Trimmed both of the Output Variance', np.mean(stats.trimboth(np.abs(sigma_1), 0.01, axis=None)) )           
        print('SNR', np.mean(snr_signal))
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))         
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n KL term factor : ' +str(kl_factor))       
        textfile.write("\n---------------------------------")
        textfile.write('\n Total test time in sec : ' +str(adv_test_stop - adv_test_start)) 
        textfile.write("\n Test RMSE : "+ str( test_rmse))        
        textfile.write("\n Output Variance: "+ str(np.mean(np.abs(sigma_)))   )                    
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
               
        test_no_steps = 0        
        rmse_test = np.zeros(int(test_X.shape[0] /batch_size))                              
        true_x = np.zeros([int(test_X.shape[0]/(batch_size)), batch_size, time_step, input_dim])
        adv_perturbations = np.zeros([int(test_X.shape[0]/(batch_size)), batch_size, time_step, input_dim])
        true_y = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size])        
        logits_ = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size]) 
        sigma_ = np.zeros([int(test_X.shape[0] / (batch_size)), batch_size ]) 
        bim_adv_test_start = timeit.default_timer()           
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(test_X.shape[0] / (batch_size)) ) 
            if (x.shape[0] == batch_size):                                 
                true_x[test_no_steps, :, :, :] = x.numpy() 
                true_y[test_no_steps, :] = y.numpy()                
                adv_x = x
                for advStep in range(maxAdvStep):                   
                    adv_perturbations1 = create_adversarial_pattern(x, y)
                    adv_x = adv_x + alpha *adv_perturbations1
                    adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                    adv_x = tf.clip_by_value(adv_x, 0.0, 1.0) 
                               
                adv_perturbations[test_no_steps, :, :, :] = adv_x
                logits, sigma = test_on_batch(adv_x)              
                logits_[test_no_steps,:] = logits.numpy()  
                sigma_[test_no_steps, :]= sigma.numpy()                                 
                rmse_ = tf.math.sqrt(tf.math.reduce_mean(tf.square(y - logits)))               
                rmse_test[test_no_steps] = rmse_.numpy()                  
                if step % 10 == 0:
                    print("Test RMSE: %.6f" % rmse_.numpy() )             
                test_no_steps+=1       
        bim_adv_test_stop = timeit.default_timer()
        print('Total Test Time: ', bim_adv_test_stop - bim_adv_test_start) 
        test_rmse = np.mean(rmse_test)      
        print('Average Test RMSE : ', test_rmse )        
        
        pf = open(PATH + test_path + 'Uncertainty_info.pkl', 'wb')         
        pickle.dump([logits_ ,true_y, sigma_ , rmse_test ], pf)                                         
        pf.close()
        
        snr_signal = np.zeros([int(test_X.shape[0]/(batch_size)) ,batch_size])
        for i in range(int(test_X.shape[0]/batch_size)):
            for j in range(batch_size):               
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/np.sum( np.square(true_x[i,j,:, :] - adv_perturbations[i, j, :, :]  ) ))              
                
        sigma_1 = np.reshape(sigma_, int(test_X.shape[0]/(batch_size))* batch_size)
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(sigma_1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()          
        print('Output Variance', np.mean(np.abs(sigma_)))          
        print('SNR', np.mean(snr_signal))
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Output Size : ' +str(output_size))
        textfile.write('\n No of epochs : ' +str(epochs))         
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n KL term factor : ' +str(kl_factor))       
        textfile.write("\n---------------------------------")
        textfile.write('\n Total test time in sec : ' +str(bim_adv_test_stop - bim_adv_test_start)) 
        textfile.write("\n Test RMSE : "+ str( test_rmse))        
        textfile.write("\n Output Variance: "+ str(np.mean(np.abs(sigma_)))   )                    
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