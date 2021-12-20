import tensorflow as tf
from tensorflow import keras
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For multiple devices (GPUs: 4, 5, 6, 7)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 4, 5, 6"
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
import wandb
os.environ["WANDB_API_KEY"] = ""
import xlsxwriter
import pandas as pd
plt.ioff()
from sklearn.model_selection import KFold
import scipy.io 
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
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
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def x_Sigma_w_x_T(x, W_Sigma):
  batch_sz = x.shape[0]
  xx_t = tf.reduce_sum(tf.multiply(x, x),axis=1, keepdims=True)               
  xx_t_e = tf.expand_dims(xx_t,axis=2)                                      
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
  grad1 = tf.expand_dims(gradi,axis=2)
  grad2 = tf.expand_dims(gradi,axis=1)
  return tf.multiply(Sigma_in, tf.matmul(grad1, grad2))


def Hadamard_sigma(sigma1, sigma2, mu1, mu2):
  sigma_1 = tf.multiply(sigma1, sigma2)
  sigma_2 = tf.matmul(tf.matmul(tf.linalg.diag(mu1) ,   sigma2),   tf.linalg.diag(mu1))
  sigma_3 = tf.matmul(tf.matmul(tf.linalg.diag(mu2) ,   sigma1),   tf.linalg.diag(mu2))
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
  mu11 = tf.expand_dims(mu1,axis=2)
  mu22 = tf.expand_dims(mu2,axis=1)
  return tf.matmul(mu11, mu22)
           
def sigma_regularizer1(x):
    input_size = 1.0   
    f_s = tf.math.softplus(x) #tf.math.log(1. + tf.math.exp(x)) 
    return  -input_size * tf.reduce_mean(1. + tf.math.log(f_s) - f_s )
#
def sigma_regularizer2(x):
    units = 1.0        
    f_s = tf.math.softplus(x)#  tf.math.log(1. + tf.math.exp(x))
    return  -units*tf.reduce_mean(1. + tf.math.log(f_s) - f_s )
    
class densityPropGRUCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units, units])]
        super(densityPropGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):        
        input_size = input_shape[-1]
        ini_sigma = -6.9       
        init_mu = 0.05      
        seed_ = None
        tau1 = 1.#/input_size
        tau2 = 1.#/self.units        
        
        self.U_z = self.add_weight(shape=(input_size, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None), regularizer=tf.keras.regularizers.l2(tau1),name='U_z', trainable=True)
        self.uz_sigma = self.add_weight(shape=(self.units,),  initializer=tf.constant_initializer(ini_sigma),regularizer=sigma_regularizer1, name='uz_sigma', trainable=True)       

        self.W_z = self.add_weight(shape=(self.units, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None), regularizer=tf.keras.regularizers.l2(tau2), name='W_z', trainable=True)
        self.wz_sigma = self.add_weight(shape=(self.units,),  initializer=tf.constant_initializer(ini_sigma),regularizer=sigma_regularizer2, name='wz_sigma', trainable=True)

        self.U_r = self.add_weight(shape=(input_size, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None), regularizer=tf.keras.regularizers.l2(tau1), name='U_r',trainable=True)
        self.ur_sigma = self.add_weight(shape=(self.units,),  initializer=tf.constant_initializer(ini_sigma),regularizer=sigma_regularizer1, name='ur_sigma', trainable=True)

        self.W_r = self.add_weight(shape=(self.units, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None),regularizer=tf.keras.regularizers.l2(tau2),  name='W_r', trainable=True)
        self.wr_sigma = self.add_weight(shape=(self.units,),  initializer=tf.constant_initializer(ini_sigma),regularizer=sigma_regularizer2, name='wr_sigma', trainable=True)

        self.U_h = self.add_weight(shape=(input_size, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None),regularizer=tf.keras.regularizers.l2(tau1),  name='U_h',trainable=True)
        self.uh_sigma = self.add_weight(shape=(self.units,),  initializer=tf.constant_initializer(ini_sigma),regularizer=sigma_regularizer1, name='uh_sigma', trainable=True)

        self.W_h = self.add_weight(shape=(self.units, self.units), initializer=tf.random_normal_initializer( mean=0.0, stddev=init_mu, seed=None),regularizer=tf.keras.regularizers.l2(tau2),  name='W_h', trainable=True)
        self.wh_sigma = self.add_weight(shape=(self.units,),  initializer=tf.constant_initializer(ini_sigma), regularizer=sigma_regularizer2, name='wh_sigma', trainable=True)
        
        self.built = True
    def call(self, inputs, states):        
        # state should be in [(batch, units), (batch, units, units)], mean vecor and covaraince matrix
        prev_state, Sigma_state = states        
        ## Update Gate
        z = tf.sigmoid(tf.matmul(inputs, self.U_z) + tf.matmul(prev_state, self.W_z))
        Uz_Sigma = tf.linalg.diag(tf.math.softplus(self.uz_sigma)  )
        #Uz_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.uz_sigma) )    )                                     
        Sigma_Uz = x_Sigma_w_x_T(inputs, Uz_Sigma)         
        ################
        Wz_Sigma = tf.linalg.diag(tf.math.softplus(self.wz_sigma)  )
        #Wz_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.wz_sigma) )   )         
        Sigma_z1 = w_t_Sigma_i_w (self.W_z, Sigma_state)
        Sigma_z2 = x_Sigma_w_x_T(prev_state, Wz_Sigma)                                  
        Sigma_z3 = tr_Sigma_w_Sigma_in (Sigma_state, Wz_Sigma)
        Sigma_out_zz = Sigma_z1 + Sigma_z2 + Sigma_z3 + Sigma_Uz
        ################
        gradi_z = grad_sigmoid(tf.matmul(inputs, self.U_z) + tf.matmul(prev_state, self.W_z))
        Sigma_out_z = activation_Sigma(gradi_z, Sigma_out_zz)
        ###################################
        ###################################
        ## Reset Gate
        r = tf.sigmoid (tf.matmul(inputs, self.U_r) + tf.matmul(prev_state, self.W_r))
        Ur_Sigma = tf.linalg.diag(tf.math.softplus(self.ur_sigma) )
       # Ur_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.ur_sigma) )    )                               
        Sigma_Ur = x_Sigma_w_x_T(inputs, Ur_Sigma)
        ################
        Wr_Sigma = tf.linalg.diag(tf.math.softplus(self.wr_sigma)   )
       # Wr_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.wr_sigma) )  )        
        Sigma_r1 = w_t_Sigma_i_w(self.W_r, Sigma_state)
        Sigma_r2 = x_Sigma_w_x_T(prev_state, Wr_Sigma)                                  
        Sigma_r3 = tr_Sigma_w_Sigma_in (Sigma_state, Wr_Sigma)
        Sigma_out_rr = Sigma_r1 + Sigma_r2 + Sigma_r3 + Sigma_Ur
        ################        
        gradi_r = grad_sigmoid(tf.matmul(inputs, self.U_r) + tf.matmul(prev_state, self.W_r))
        Sigma_out_r = activation_Sigma(gradi_r, Sigma_out_rr)
        ###################################
        ###################################
        ## Intermediate Activation
        h = tf.tanh(tf.matmul(inputs, self.U_h) + tf.matmul(tf.multiply(prev_state, r), self.W_h))
        Uh_Sigma = tf.linalg.diag(tf.math.softplus(self.uh_sigma) )
      #  Uh_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.uh_sigma) )     )                               
        Sigma_Uh = x_Sigma_w_x_T(inputs, Uh_Sigma)
        ################
        sigma_g = Hadamard_sigma(Sigma_state, Sigma_out_r, prev_state, r)
        ################
        Wh_Sigma = tf.linalg.diag(tf.math.softplus(self.wh_sigma)    )                                
        #Wh_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.wh_sigma) ))        
        Sigma_h1 = w_t_Sigma_i_w (self.W_h, sigma_g)
        Sigma_h2 = x_Sigma_w_x_T(tf.multiply(prev_state, r), Wh_Sigma)                                   
        Sigma_h3 = tr_Sigma_w_Sigma_in (sigma_g, Wh_Sigma)
        Sigma_out_hh = Sigma_h1 + Sigma_h2 + Sigma_h3 + Sigma_Uh
        ################
        gradi_h = grad_tanh(tf.matmul(inputs, self.U_h) + tf.matmul(tf.multiply(prev_state, r), self.W_h))
        Sigma_out_h = activation_Sigma(gradi_h, Sigma_out_hh)
        ###################################
        ###################################
        ## Current State              
        mu_out = tf.multiply(z, prev_state) + tf.multiply(1-z, h)
        sigma_a = Hadamard_sigma(Sigma_out_z, Sigma_state,z, prev_state)
        sigma_b = Hadamard_sigma(Sigma_out_z, Sigma_out_h, 1-z, h)
        mu_s_muhT = mu_muT(prev_state, h)
        mu_h_mu_sT = mu_muT(h, prev_state)
        sigma_ab = tf.multiply(Sigma_out_z, mu_s_muhT)
        sigma_abT = tf.multiply(Sigma_out_z, mu_h_mu_sT)
        Sigma_out = sigma_a + sigma_b - sigma_ab - sigma_abT
        
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))         
        output = mu_out
        new_state = (mu_out, Sigma_out)       
        return output, new_state  
                

class LinearNotFirst(keras.layers.Layer):    
    def __init__(self, units):
        super(LinearNotFirst, self).__init__()
        self.units = units
                
    def build(self, input_shape):
        ini_sigma = -4.6       
        tau = 1. #/input_shape[-1]          
        self.w_mu = self.add_weight(name = 'w_mu', shape=(input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None), regularizer=tf.keras.regularizers.L1L2(tau,tau),#tau/self.units), #tf.keras.regularizers.l2(0.5*0.001),
            trainable=True,
        )
        self.w_sigma = self.add_weight(name = 'w_sigma',
            shape=(self.units,),
            initializer= tf.constant_initializer(ini_sigma),#tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None) , 
            regularizer=sigma_regularizer2, #   tf.constant_initializer(ini_sigma)
            trainable=True,
        )   
    def call(self, mu_in, Sigma_in):        
        mu_out = tf.matmul(mu_in, self.w_mu) #+ self.b_mu       
        W_Sigma = tf.linalg.diag(tf.math.softplus(self.w_sigma))    #tf.math.log(1. + tf.math.exp(self.w_sigma)))       
        Sigma_1 = w_t_Sigma_i_w (self.w_mu, Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)                                   
        Sigma_3 = tr_Sigma_w_Sigma_in (Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3 #+ tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.b_sigma)))  
        
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)  
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out

       

class mysoftmax(keras.layers.Layer):
    """Mysoftmax"""
    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.softmax(mu_in)
        pp1 = tf.expand_dims(mu_out, axis=2)
        pp2 = tf.expand_dims(mu_out, axis=1)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        Sigma_out = tf.matmul(grad, tf.matmul(Sigma_in, tf.transpose(grad, perm=[0, 2, 1])))
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)        
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out     
    
#def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):
#    NS = tf.linalg.diag(tf.constant(1e-4, shape=[batch_size, num_labels]))#shape=[batch_size, num_labels, num_labels]
#    I = tf.eye(num_labels, batch_shape=[batch_size])
#    y_pred_sd_ns = y_pred_sd + NS
#    y_pred_sd_inv = tf.linalg.solve(y_pred_sd_ns, I)
#    mu_ = y_pred_mean - y_test #shape=[batch_size, num_labels]
#    mu_sigma = tf.matmul(tf.expand_dims(mu_, axis=1) ,  y_pred_sd_inv)  #shape=[batch_size, 1, num_labels]
#    ms1 = tf.math.reduce_mean(tf.squeeze(tf.matmul(mu_sigma , tf.expand_dims(mu_, axis=2))) )
#    ms2 = tf.math.reduce_mean(tf.squeeze(tf.linalg.slogdet(y_pred_sd_ns)[1]))
#    ms = tf.math.reduce_mean(ms1 + ms2)
#    return ms         

def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):    
    y_pred_sd_ns = y_pred_sd 
    s, u, v = tf.linalg.svd(y_pred_sd_ns, full_matrices=True, compute_uv=True)	
    s_ =  s + 1.0e-4  #tf.clip_by_value(t=s, clip_value_min=tf.constant(1e-5),       clip_value_max=tf.constant(1e+5)) #
    s_inv = tf.linalg.diag(tf.math.divide_no_nan(1., s_) )    
    y_pred_sd_inv = tf.matmul(tf.matmul(v, s_inv), tf.transpose(u, [0, 2,1])) 
    mu_ = y_test - y_pred_mean 
    mu_sigma = tf.matmul( tf.expand_dims(mu_, axis=1)  ,  y_pred_sd_inv)     
    loss1 =  tf.squeeze(tf.matmul(mu_sigma ,  tf.expand_dims(mu_, axis=2) ))   
    loss2 =  tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(s_), axis =-1) )        
    loss = tf.math.reduce_mean(tf.math.add(loss1,loss2))
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss) 
    return loss 
    
            
class Density_prop_GRU(tf.keras.Model):
  def __init__(self, units, num_classes, name=None):
    super(Density_prop_GRU, self).__init__()
    self.units = units
    self.num_classes = num_classes
    self.cell = densityPropGRUCell(self.units)    
    self.rnn = tf.keras.layers.RNN(self.cell, return_state=True)
    self.linear_1 = LinearNotFirst(self.num_classes)
    self.mysoftma = mysoftmax() 

  def call(self, inputs, training=True):
  #  h_mu = tf.convert_to_tensor(np.zeros((inputs.shape[0], self.units,)).astype(np.float32))
  #  h_sigma = tf.convert_to_tensor(np.random.random_sample((inputs.shape[0], self.units, self.units)).astype(np.float32))        
    xx = self.rnn(inputs)#, initial_state=[h_mu, h_mu, h_sigma, h_sigma ])    
    x, mu_state, sigma_state = xx    
    m, s = self.linear_1(x, sigma_state) 
    outputs, Sigma = self.mysoftma(m, s)    
    Sigma = tf.where(tf.math.is_nan(Sigma), tf.zeros_like(Sigma), Sigma)
    Sigma = tf.where(tf.math.is_inf(Sigma), tf.zeros_like(Sigma), Sigma)       
    return outputs, Sigma    
    
def main_function(time_step=144, input_dim=963, units=400, class_num=7 , batch_size=10, epochs=600, kl_factor=0.01,
     lr = 0.0001, lr_end=0.000001 , Random_noise=True, gaussain_noise_std=0.2, Adversarial_noise=False, Targeted=False, epsilon=0.007, adversary_target_cls=3,
     BIM_adversarial=False, alpha=1, maxAdvStep=100,
     Training=False, Testing=True, continue_training=False, saved_model_epochs=500):

    PATH = './VDP_gru2/vdp_saved_models_units_{}_lr_{}/vdp_gru_epoch_{}/'.format(units,lr, epochs) 
    data = scipy.io.loadmat('./dataset/PEMS.mat')
    x_train = data['X']  # shape is [N,T,V]
    y_train = data['Y']  # shape is [N,1]
    x_test = data['Xte']
    y_test = data['Yte'] 
    
    #x_train = x_train[:,:,0:300].astype('float32')     
   # x_test = x_test[:,:,0:300].astype('float32')
    
    x_train = x_train.astype('float32')     
    x_test = x_test.astype('float32') 
    y_train = y_train -1
    y_test = y_test -1
    x_train = (x_train-np.amin(x_train))/(np.amax(x_train)-np.amin(x_train))    
    x_test = (x_test-np.amin(x_test))/(np.amax(x_test) - np.amin(x_test))  
    
    one_hot_y_train = tf.one_hot(np.squeeze(y_train).astype(np.float32), depth=class_num)
    one_hot_y_test = tf.one_hot(np.squeeze(y_test).astype(np.float32), depth=class_num)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)   
    
#    inputs = np.concatenate((x_train, x_test ), axis=0)
#    targets = np.concatenate((y_train, y_test), axis=0) 
#    x_train = inputs[0:400, :,:]
#    y_train = targets[0:400]
#    x_test = inputs[400:440, :,:]
#    y_test = targets[400:440]
#    #  print( inputs[0:9400, : ,:].shape)
#    #  print(  targets[0:9400].shape)   
    
#    pf = open('./dataset/FaceDetect_data.pkl', 'rb')                    
#    x_train, y_train, x_test, y_test = pickle.load(pf)                                                   
#    pf.close()      


    # Cutom Trianing Loop with Graph      
    gru_model = Density_prop_GRU(units, class_num, name = 'vdp_gru')           
    num_train_steps = epochs * int(x_train.shape[0] /batch_size)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,  decay_steps=num_train_steps,  end_learning_rate=lr_end, power=10.)      
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn , clipnorm= 1.0)        
        
    @tf.function()# Make it fast.       
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
          logits, sigma = gru_model(x, training=True)           
          loss_final = nll_gaussian(y, logits,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+6),
                                   clip_value_max=tf.constant(1e+6)), class_num , batch_size)
          regularization_loss=tf.math.add_n(gru_model.losses)             
          loss = 0.5 * (loss_final + kl_factor*regularization_loss )           
        gradients = tape.gradient(loss, gru_model.trainable_weights)            
#        for g,v in zip(gradients, gru_model.trainable_weights):
#            tf.print(v.name, tf.reduce_max(g))
#            gradients = [(tf.clip_by_value(grad, -10., 10.))
#                                  for grad in gradients]  
        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape),grad)) for grad in gradients] 
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape),grad)) for grad in gradients]    
        optimizer.apply_gradients(zip(gradients, gru_model.trainable_weights))       
        return loss, logits, sigma, gradients, regularization_loss, loss_final 
          
    @tf.function()# Make it fast.    
    def valid_on_batch(x, y):                     
        mu_out, sigma = gru_model(x, training=False) 
      #  gru_model.trainable = False              
        vloss = nll_gaussian(y, mu_out,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+6),
                                           clip_value_max=tf.constant(1e+6)), class_num , batch_size)                                           
        regularization_loss=tf.math.add_n(gru_model.losses)
        total_vloss = 0.5 *(vloss + kl_factor*regularization_loss)    
        return total_vloss, mu_out, sigma, regularization_loss, vloss
            
    @tf.function       
    def test_on_batch(x):  
      #  gru_model.trainable = False                    
        mu_out, sigma = gru_model(x, training=False)            
        return mu_out, sigma
        
            
    @tf.function   
    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
          tape.watch(input_image)
        #  gru_model.trainable = False 
          prediction, sigma = gru_model(input_image) 
          loss_final = nll_gaussian(input_label, prediction,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+6),
                                 clip_value_max=tf.constant(1e+6)), class_num , batch_size)                         
          loss = 0.5 * loss_final 
        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        gradient = tf.where(tf.math.is_nan(gradient),  tf.constant(1.0e-5, shape=gradient.shape), gradient)
        gradient = tf.where(tf.math.is_inf(gradient),  tf.constant(1.0e-5, shape=gradient.shape), gradient)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad              
    
              
    if Training:
        wandb.init(entity = "dimah", project="VDP_GRU_PEMS_units_{}_epochs_{}_lr_{}".format(units, epochs, lr)) 
        if continue_training:
            saved_model_path = './VDP_gru/vdp_saved_models_units_{}_lr_{}/vdp_gru_epoch_{}/'.format(units,lr, saved_model_epochs)             
            gru_model.load_weights(saved_model_path + 'vdp_gru_model')                      
                
        train_acc = np.zeros(epochs)
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)
            acc1 = 0
            acc_valid1 = 0
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0
            #Training
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step / int(x_train.shape[0] / (batch_size)) )
                if (x.shape[0] == batch_size):  
                    loss, logits, sigma, gradients, regularization_loss, loss_final  = train_on_batch(x, y)
                    err1+= loss.numpy()
                    corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
                    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                    acc1+=accuracy.numpy()                
                    if step % 10 == 0:
                        print("Step:", step, "Loss:", float(loss.numpy()))
                        print("Training accuracy so far: %.3f" % accuracy.numpy())                    
                    tr_no_steps+=1
                    wandb.log({"Trianing Loss per minibatch": loss.numpy(),
                                "Training Accuracy per minibatch": accuracy.numpy(),  
                                "gradient per minibatch": np.mean(gradients[0]),   
                                "Average Variance value": tf.reduce_mean(sigma).numpy(),
                                "Regularization_loss": regularization_loss.numpy(),                               
                                 "Log-Likelihood Loss": np.mean(loss_final.numpy())                       
                        })     
            train_acc[epoch] = acc1/tr_no_steps
            train_err[epoch] = err1/tr_no_steps
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])

            # Validation
            err_valid1 = np.zeros(int(x_test.shape[0] / (batch_size)))
            for step, (x, y) in enumerate(test_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)) )
                if (x.shape[0] == batch_size):                   
                    total_vloss, logits, sigma, regularization_loss, vloss = valid_on_batch(x, y)                
                    err_valid1[va_no_steps] = total_vloss.numpy()
                    corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
                    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                    acc_valid1+=accuracy.numpy()
                    if step % 10 == 0:
                        print("Step:", step, "Loss:", float(total_vloss.numpy()))
                        print("Average validation accuracy so far: %.3f" % accuracy.numpy())                    
                    va_no_steps+=1                          
                    wandb.log({"Average Variance value (validation Set)": tf.reduce_mean(sigma).numpy(),
                               "Regularization_loss (validation Set)": regularization_loss.numpy(),                               
                               "Log-Likelihood Loss (validation Set)": np.mean(vloss.numpy() ),
                               "Validation Loss per minibatch": total_vloss.numpy() ,                              
                               "Validation Acuracy per minibatch": accuracy.numpy()                                                         
                                    })                                                                                
                    gru_model.save_weights(PATH + 'vdp_gru_model') 
            wandb.log({"Training Loss": (err1 / tr_no_steps),  
                        "Training Accuracy": (acc1 / tr_no_steps),                      
                        "Validation Loss": (err_valid1 / va_no_steps),
                        "Validation Accuracy": (acc_valid1 / va_no_steps),
                        'epoch': epoch                        
                       })

            valid_acc[epoch] = acc_valid1/va_no_steps 
            valid_error[epoch] = np.amin(err_valid1) 
            stop = timeit.default_timer()
            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch])
            print('Validation Acc  ', valid_acc[epoch])
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])           
           
        gru_model.save_weights(PATH + 'vdp_gru_model')
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("VDP GRU on PEMS Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_GRU_on_PEMS_Data_acc.png')
            plt.close(fig)

            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')
            #plt.ylim(0, 1.1)
            plt.title("VDP GRU on PEMS Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_GRU_on_PEMS_Data_error.png')
            plt.close(fig)
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)
        f.close()

        textfile = open(PATH + 'Related_hyperparameters.txt','w')
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n time step : ' +str(time_step))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n batch size : ' +str(batch_size))
        textfile.write('\n KL term factor : ' +str(kl_factor))  
        textfile.write("\n---------------------------------")
        if Training:
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
                textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))

                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc[epoch])))

                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.close()

    if(Testing):    
        test_path = 'test_results/'        
        if Random_noise:
            test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)  
            os.makedirs(PATH + test_path)
        else:            
            os.makedirs(PATH + test_path) 
        gru_model.load_weights(PATH + 'vdp_gru_model')            
      
        valid_size = x_test.shape[0]        
        test_no_steps = 0
        err_test = np.zeros(int(valid_size /batch_size))
        acc_test = np.zeros(int(valid_size /batch_size))        
        true_x = np.zeros([int(valid_size /batch_size), batch_size, time_step, input_dim])
        true_y = np.zeros([int(valid_size /batch_size), batch_size, class_num])
        mu_out_ = np.zeros([int(valid_size /batch_size), batch_size, class_num])           
        sigma_ = np.zeros([int(valid_size / (batch_size)), batch_size, class_num, class_num])
        test_start = timeit.default_timer()
        for step, (x, y) in enumerate(test_dataset):
          update_progress(step / int(valid_size / (batch_size)) )
          if (x.shape[0] == batch_size):  
              true_x[test_no_steps, :, :, :] = x.numpy()
              true_y[test_no_steps, :, :] = y.numpy()
              if Random_noise:
                  noise = tf.random.normal(shape = [batch_size, time_step, input_dim], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype )
                  x = x +  noise
              #logits, sigma = test_on_batch(x) 
              tloss, logits, sigma, regularization_loss, vloss = valid_on_batch(x, y)
              mu_out_[test_no_steps,:,:] =logits.numpy() 
              sigma_[test_no_steps, :, :, :]= sigma.numpy()              
              err_test[test_no_steps] = tloss.numpy() 
    
              corr = tf.equal(tf.math.argmax(logits, axis=1),tf.math.argmax(y,axis=1))
              accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
              acc_test[test_no_steps]=accuracy.numpy()
    
              if step % 50 == 0:
                  print("Step:", step, "Loss:", float(tloss.numpy()))
                  print("Test accuracy: %.3f" % accuracy.numpy())
               # loss_total_layer.append(loss_layers)
              test_no_steps+=1               
        test_stop = timeit.default_timer()
        print('Total Test Time: ', test_stop - test_start)
        test_acc = np.mean(acc_test) 
        test_error = np.amin(err_test) 
        print('Test accuracy : ', test_acc)
        print('Test error : ', test_error)
        
        if Random_noise:
            snr_signal = np.zeros([int(valid_size/(batch_size)) ,batch_size])
            for i in range(int(valid_size/(batch_size))):
                for j in range(batch_size):
                    noise = tf.random.normal(shape = [time_step, input_dim ], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ).numpy()             
                    snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/np.sum( np.square(noise) ))
            print('SNR', np.mean(snr_signal))        
        
        pf = open(PATH + test_path + 'prediction_info.pkl', 'wb')         
        pickle.dump([mu_out_,true_y, sigma_,  acc_test, err_test ], pf)                                                   
        pf.close()  
        
        
        writer = pd.ExcelWriter(PATH + test_path + 'Covarience_matrices.xlsx', engine='xlsxwriter')
        for i in range(int(valid_size/batch_size)):
            for j in range(batch_size):  
                df = pd.DataFrame(sigma_[i,j,:,:])    
                # Write your DataFrame to a file   
                df.to_excel(writer, "Sheet", startrow=i*(class_num+4),  startcol=j*(class_num+6))
                # Save the result
                df1 = pd.DataFrame(mu_out_[i,j,:])
                df1.to_excel(writer, 'Sheet', startrow=i*(class_num+4),  startcol=(7 + j*(class_num+6)))
        writer.save() 
        
        var = np.zeros([int(valid_size /batch_size) ,batch_size])        
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]                
         
        print('Average Output Variance', np.mean(var))   
            
        var1 = np.reshape(var, int(valid_size/(batch_size))* batch_size)                
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(var1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")           
        writer.save()
        
              
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))  
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n KL term factor : ' +str(kl_factor))       
        textfile.write("\n---------------------------------")   
        textfile.write('\n Total test time in sec : ' +str(test_stop - test_start))   
        textfile.write("\n Test Accuracy : "+ str( test_acc))
        textfile.write("\n Test error : "+ str(test_error))     
        textfile.write("\n Average Output Variance: "+ str(np.mean(var1)))                   
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussain_noise_std )) 
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))                
        textfile.write("\n---------------------------------")    
        textfile.close() 
         
    if(Adversarial_noise):
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(epsilon)
            os.makedirs(PATH + test_path)
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)  
            os.makedirs(PATH + test_path)
        gru_model.load_weights(PATH + 'vdp_gru_model')    
        valid_size = x_test.shape[0]     
        
        test_no_steps = 0     
        err_test = np.zeros(int(valid_size /batch_size))
        acc_test = np.zeros(int(valid_size/batch_size))        
        true_x = np.zeros([int(valid_size /batch_size), batch_size, time_step, input_dim])
        adv_perturbations = np.zeros([int(valid_size /batch_size), batch_size, time_step, input_dim])
        true_y = np.zeros([int(valid_size /batch_size), batch_size, class_num])
        mu_out_ = np.zeros([int(valid_size /batch_size), batch_size, class_num])     
        sigma_ = np.zeros([int(valid_size / (batch_size)), batch_size, class_num, class_num])
        adv_test_start = timeit.default_timer()
        for step, (x, y) in enumerate(test_dataset):
            update_progress(step / int(valid_size / (batch_size)) ) 
            if (x.shape[0] == batch_size):  
                true_x[test_no_steps, :, :, :] = x.numpy()
                true_y[test_no_steps, :, :] = y.numpy()            
                if Targeted:
                    y_true_batch = np.zeros_like(y)
                    y_true_batch[:, adversary_target_cls] = 1.0            
                    adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y_true_batch)
                else:
                    adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y)
                adv_x = x + epsilon*adv_perturbations[test_no_steps, :, :, :] 
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)                
                
                tloss, mu_out, sigma, regularization_loss, vloss = valid_on_batch(adv_x, y)          
                mu_out_[test_no_steps,:,:] = mu_out.numpy()   
                sigma_[test_no_steps, :, :, :]= sigma.numpy()                      
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_test[test_no_steps]=accuracy.numpy()              
                err_test[test_no_steps] = tloss.numpy() 
                  
                if step % 50 == 0:
                    print("Step:", step, "Loss:", float(tloss.numpy()))
                    print("test accuracy: %.3f" % accuracy.numpy())             
                test_no_steps+=1               
        adv_test_stop = timeit.default_timer()
        print('Total Test Time: ', adv_test_stop - adv_test_start)    
        test_acc = np.mean(acc_test)         
        test_error = np.amin(err_test) 
        print('Test accuracy : ', test_acc)
        print('Test error : ', test_error)                     
        
        pf = open(PATH + test_path + 'prediction_info.pkl', 'wb')                    
        pickle.dump([mu_out_, true_y,sigma_, acc_test, err_test ], pf)                                                   
        pf.close()
        
        snr_signal = np.zeros([int(valid_size/(batch_size)) ,batch_size])
        for i in range(int(valid_size/batch_size)):
            for j in range(batch_size):               
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/np.sum( np.square(true_x[i,j,:, :] - np.clip(true_x[i,j,:, :]+epsilon*adv_perturbations[i, j, :, :], 0.0, 1.0)  ) ))         
        print('SNR', np.mean(snr_signal))        
        
        writer = pd.ExcelWriter(PATH + test_path + 'Covarience_matrices.xlsx', engine='xlsxwriter')
        for i in range(int(valid_size/batch_size)):
            for j in range(batch_size):  
                df = pd.DataFrame(sigma_[i,j,:,:])    
                # Write your DataFrame to a file   
                df.to_excel(writer, "Sheet", startrow=i*(class_num+4),  startcol=j*(class_num+8))
                # Save the result
                df1 = pd.DataFrame(mu_out_[i,j,:])
                df1.to_excel(writer, 'Sheet', startrow=i*(class_num+4),  startcol=(8 + j*(class_num+8)))
        writer.save() 
        
        pred_var = np.zeros(int(valid_size ))   
        true_var = np.zeros(int(valid_size )) 
        correct_classification = np.zeros(int(valid_size)) 
        misclassification_pred = np.zeros(int(valid_size )) 
        misclassification_true = np.zeros(int(valid_size )) 
        predicted_out = np.zeros(int(valid_size )) 
        true_out = np.zeros(int(valid_size )) 
        k=0   
        k1=0
        k2=0  
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out[k] = np.argmax(mu_out_[i,j,:])
                true_out[k] = np.argmax(true_y[i,j,:])
                pred_var[k] = sigma_[i,j, int(predicted_out[k]), int(predicted_out[k])]    
                true_var[k] = sigma_[i,j, int(true_out[k]), int(true_out[k])]  
                if (predicted_out[k] == true_out[k]):
                    correct_classification[k1] = sigma_[i,j, int(predicted_out[k]), int(predicted_out[k])] 
                    k1=k1+1
                if (predicted_out[k] != true_out[k]):
                    misclassification_pred[k2] = sigma_[i,j, int(predicted_out[k]), int(predicted_out[k])] 
                    misclassification_true[k2] = sigma_[i,j, int(true_out[k]), int(true_out[k])]   
                    k2=k2+1                 
                k=k+1         
        print('Average Output Variance', np.mean(pred_var))   
            
        var1 = pred_var#np.reshape(var, int(x_test.shape[0]/(batch_size))* batch_size)  
        #print(var1)              
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(var1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")      
        
        df1 = pd.DataFrame(predicted_out)
        df1.to_excel(writer, 'Sheet',  startcol=4)
        
        df2 = pd.DataFrame(true_out)
        df2.to_excel(writer, 'Sheet',  startcol=7)
        
        df3 = pd.DataFrame(correct_classification)
        df3.to_excel(writer, 'Sheet',  startcol=10)  
        
        df4 = pd.DataFrame(misclassification_pred)
        df4.to_excel(writer, 'Sheet',  startcol=13)
        
        df5 = pd.DataFrame(misclassification_true)
        df5.to_excel(writer, 'Sheet',  startcol=16)      
        writer.save()      
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))         
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))   
        textfile.write('\n KL term factor : ' +str(kl_factor))     
        textfile.write("\n---------------------------------")   
        textfile.write('\n Total test time in sec with adversarial attack : ' +str(adv_test_stop - adv_test_start))        
        textfile.write("\n Test Accuracy : "+ str( test_acc))
        textfile.write("\n Test error : "+ str(test_error))    
        textfile.write("\n Average Output Variance: "+ str(np.mean(var1)))                 
        textfile.write("\n---------------------------------")
        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
        textfile.write("\n---------------------------------")    
        textfile.close()          
        
    if(BIM_adversarial):
        if Targeted:
            test_path = 'test_results_BIM_targeted_adversarial_noise_{}_alpha_{}/'.format(epsilon, alpha)
            os.makedirs(PATH + test_path)
        else:
            test_path = 'test_results_BIM_non_targeted_adversarial_noise_{}_alpha_{}/'.format(epsilon, alpha)  
            os.makedirs(PATH + test_path)            
        gru_model.load_weights(PATH + 'vdp_gru_model')         
        valid_size = x_test.shape[0]      
                 
        test_no_steps = 0     
        err_test = np.zeros(int(valid_size /batch_size))
        acc_test = np.zeros(int(valid_size /batch_size))        
        true_x = np.zeros([int(valid_size /batch_size), batch_size, time_step, input_dim])
        adv_perturbations = np.zeros([int(valid_size/batch_size), batch_size, time_step, input_dim])
        true_y = np.zeros([int(valid_size /batch_size), batch_size, class_num])
        mu_out_ = np.zeros([int(valid_size/batch_size), batch_size, class_num])     
        sigma_ = np.zeros([int(valid_size / (batch_size)), batch_size, class_num, class_num])
        bim_adv_test_start = timeit.default_timer()
        for step, (x, y) in enumerate(test_dataset):
            update_progress(step / int(valid_size / (batch_size)) ) 
            if (x.shape[0] == batch_size):  
                true_x[test_no_steps, :, :, :] = x.numpy()
                true_y[test_no_steps, :, :] = y.numpy() 
                adv_x = x
                for advStep in range(maxAdvStep):           
                    if Targeted:
                        y_true_batch = np.zeros_like(y)
                        y_true_batch[:, adversary_target_cls] = 1.0            
                        adv_perturbations1 = create_adversarial_pattern(x, y_true_batch)
                    else:
                        adv_perturbations1 = create_adversarial_pattern(x, y)
                    adv_x = adv_x + alpha *adv_perturbations1
                    adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                    adv_x = tf.clip_by_value(adv_x, 0.0, 1.0) 
                                   
                adv_perturbations[test_no_steps, :, :, :] = adv_x
                tloss, mu_out, sigma, regularization_loss, vloss = valid_on_batch(adv_x, y)          
                mu_out_[test_no_steps,:,:] = mu_out.numpy()   
                sigma_[test_no_steps, :, :, :]= sigma.numpy()                      
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_test[test_no_steps]=accuracy.numpy()              
                err_test[test_no_steps] = tloss.numpy() 
                  
                if step % 50 == 0:
                    print("Step:", step, "Loss:", float(tloss.numpy()))
                    print("test accuracy: %.3f" % accuracy.numpy())             
                test_no_steps+=1               
        bim_adv_test_stop = timeit.default_timer()
        print('Total Test Time: ', bim_adv_test_stop - bim_adv_test_start)    
        test_acc = np.mean(acc_test)         
        test_error = np.amin(err_test) 
        print('Test accuracy : ', test_acc)
        print('Test error : ', test_error)                     
        
        pf = open(PATH + test_path + 'prediction_info.pkl', 'wb')                    
        pickle.dump([mu_out_, true_y,sigma_, acc_test, err_test ], pf)                                                   
        pf.close()
        
        snr_signal = np.zeros([int(valid_size/(batch_size)) ,batch_size])
        for i in range(int(valid_size/batch_size)):
            for j in range(batch_size):               
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :]))/np.sum( np.square(true_x[i,j,:, :] - adv_perturbations[i, j, :, :]  ) ))          
        print('SNR', np.mean(snr_signal))       
        
        writer = pd.ExcelWriter(PATH + test_path + 'Covarience_matrices.xlsx', engine='xlsxwriter')
        for i in range(int(valid_size/batch_size)):
            for j in range(batch_size):  
                df = pd.DataFrame(sigma_[i,j,:,:])    
                # Write your DataFrame to a file   
                df.to_excel(writer, "Sheet", startrow=i*(class_num+4),  startcol=j*(class_num+6))
                # Save the result
                df1 = pd.DataFrame(mu_out_[i,j,:])
                df1.to_excel(writer, 'Sheet', startrow=i*(class_num+4),  startcol=(7 + j*(class_num+6)))
        writer.save() 
        
        pred_var = np.zeros(int(valid_size ))   
        true_var = np.zeros(int(valid_size )) 
        correct_classification = np.zeros(int(valid_size )) 
        misclassification_pred = np.zeros(int(valid_size )) 
        misclassification_true = np.zeros(int(valid_size )) 
        predicted_out = np.zeros(int(valid_size )) 
        true_out = np.zeros(int(valid_size )) 
        k=0   
        k1=0
        k2=0  
        for i in range(int(valid_size /batch_size)):
            for j in range(batch_size):               
                predicted_out[k] = np.argmax(mu_out_[i,j,:])
                true_out[k] = np.argmax(true_y[i,j,:])
                pred_var[k] = sigma_[i,j, int(predicted_out[k]), int(predicted_out[k])]    
                true_var[k] = sigma_[i,j, int(true_out[k]), int(true_out[k])]  
                if (predicted_out[k] == true_out[k]):
                    correct_classification[k1] = sigma_[i,j, int(predicted_out[k]), int(predicted_out[k])] 
                    k1=k1+1
                if (predicted_out[k] != true_out[k]):
                    misclassification_pred[k2] = sigma_[i,j, int(predicted_out[k]), int(predicted_out[k])] 
                    misclassification_true[k2] = sigma_[i,j, int(true_out[k]), int(true_out[k])]   
                    k2=k2+1 
                
                k=k+1            
         
        print('Average Output Variance', np.mean(pred_var))            
        var1 = pred_var#np.reshape(var, int(x_test.shape[0]/(batch_size))* batch_size)  
        #print(var1)              
        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
        df = pd.DataFrame(np.abs(var1) )   
        # Write your DataFrame to a file   
        df.to_excel(writer, "Sheet")      
        
        df1 = pd.DataFrame(predicted_out)
        df1.to_excel(writer, 'Sheet',  startcol=2)
        
        df2 = pd.DataFrame(true_out)
        df2.to_excel(writer, 'Sheet',  startcol=5)
        
        df3 = pd.DataFrame(correct_classification)
        df3.to_excel(writer, 'Sheet',  startcol=7)  
        
        df4 = pd.DataFrame(misclassification_pred)
        df4.to_excel(writer, 'Sheet',  startcol=10)
        
        df5 = pd.DataFrame(misclassification_true)
        df5.to_excel(writer, 'Sheet',  startcol=13)      
        writer.save()
        
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No Hidden Nodes : ' +str(units))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))         
        textfile.write('\n time step : ' +str(time_step))   
        textfile.write('\n batch size : ' +str(batch_size))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))   
        textfile.write('\n KL term factor : ' +str(kl_factor))     
        textfile.write("\n---------------------------------")   
        textfile.write('\n Total test time in sec with BIM adversarial attack : ' +str(bim_adv_test_stop - bim_adv_test_start))        
        textfile.write("\n Test Accuracy : "+ str( test_acc))
        textfile.write("\n Test error : "+ str(test_error))    
        textfile.write("\n Average Output Variance: "+ str(np.mean(var1)))                 
        textfile.write("\n---------------------------------")
        if BIM_adversarial:
            if Targeted:
                textfile.write('\n BIM Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:      
                textfile.write('\n BIM Adversarial attack: Non-TARGETED')
            textfile.write('\n BIM Adversarial Noise epsilon: '+ str(epsilon ))
            textfile.write("\n Alpha: "+ str(alpha)) 
            textfile.write("\n Maximum number of iterations: "+ str(maxAdvStep)) 
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
        textfile.write("\n---------------------------------")    
        textfile.close()
                    

if __name__ == '__main__':
    main_function()
