# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:35:59 2020

@author: agarwal.270a
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import tensorflow.keras.layers as layers
#import modules.custom_layers as clayers
from tensorflow.keras import initializers as initizers
from lib.model_funcs import make_data_pipe, find_batch_size
import numpy as np
import copy
import matplotlib.pyplot as plt
#tf.keras.backend.set_floatx('float64')

class Model_CondVAE(tf.keras.Model):
    def __init__(self,optimizer,n_units,in_shape,cond_shape):
        '''
            Setting all the variables for our model.
        '''
        super(Model_CondVAE, self).__init__()
        #self.conv1 = Conv2D(32, 3, activation='relu')
        w_init=initizers.glorot_uniform()
        b_init=initizers.Zeros()
        #self.w1=tf.Variable(w_init,trainable=True)
        #self.w2=tf.Variable(w_init,trainable=True)
        #self.K=tf.Variable(K,trainable=False)
        self.n_units=n_units
        self.in_shape=in_shape
        self.cond_shape=cond_shape
        cntr=0
        #w_init_val1=dct(np.eye(in_shape),norm='ortho')
        #self.w1=tf.Variable(w_init_val1,trainable=True)
        #w_init_val2=idct(np.eye(in_shape),norm='ortho')
        #self.w2=tf.Variable(w_init_val2,trainable=True)
        self.w1=tf.Variable(w_init(shape=[self.in_shape+self.cond_shape,self.n_units[cntr],],
                                   dtype=tf.float64),trainable=True)
        self.b1=tf.Variable(b_init(shape=[self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        cntr+=1
        self.w_mu=tf.Variable(w_init(shape=[self.n_units[cntr-1],
                                            self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        self.b_mu=tf.Variable(b_init(shape=[self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        self.w_logsig=tf.Variable(w_init(shape=[self.n_units[cntr-1],
                                            self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        self.b_logsig=tf.Variable(b_init(shape=[self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        cntr+=1
        self.w2=tf.Variable(w_init(shape=[self.n_units[cntr-1]+self.cond_shape,
                                            self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        self.b2=tf.Variable(b_init(shape=[self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        cntr+=1
        self.w3=tf.Variable(w_init(shape=[self.n_units[cntr-1],
                                            self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        self.b3=tf.Variable(b_init(shape=[self.n_units[cntr]],
                                   dtype=tf.float64),trainable=True)
        

        #self.loss_mse = tf.keras.losses.MeanSquaredError()
        #self.loss_l1 = tf.keras.losses.MeanAbsoluteError()
        self.relu=tf.nn.relu
        self.sigmoid=tf.sigmoid
        self.optimizer = optimizer

        #self.train_metric_z = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_z')
        self.train_loss1 = tf.keras.metrics.Mean(name='train_loss1')
        self.train_loss2 = tf.keras.metrics.Mean(name='train_loss2')

        #self.train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
        self.test_loss2 = tf.keras.metrics.Mean(name='test_loss2')

        #self.test_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        #self.lamda=lamda
        self.var_list1=[self.w1,self.b1,self.w_mu,self.b_mu,
                        self.w_logsig,self.b_logsig,self.w2,self.b2,
                        self.w3,self.b3]
    
    def recon_loss(self,x_true,x_pred):
        # E[log P(X|z)]
        recon = tf.reduce_sum(tf.square(x_true - x_pred),axis=1)
        return tf.reduce_sum(recon,axis=0)
        
    def KL_loss(self,mu,logsigma):
        # D_KL(Q(z|X) || P(z|X))
        kl = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1. - 
                                 logsigma, axis=1)
        return tf.reduce_sum(kl,axis=0)
    
    def sample_z(self,mu,logsig):
        eps=tf.random.normal(shape=tf.shape(mu),mean=0.,stddev=1.,
                             dtype=tf.dtypes.float64)
        z=mu + tf.exp(logsig / 2) * eps
        return z
    
    def encoder(self,x,cond):
        cond=tf.pow(tf.cast(cond,tf.float64),-1) #invert to get HR in BPS
        VAE_in=tf.concat([x,cond],axis=-1)
        x = self.relu(tf.matmul(VAE_in,self.w1)+self.b1)
        mu = tf.matmul(x,self.w_mu)+self.b_mu
        logsig = tf.matmul(x,self.w_logsig)+self.b_logsig
        return mu,logsig
    
    def decoder(self,z,cond):
        cond=tf.pow(tf.cast(cond,tf.float64),-1) #invert to get HR in BPS
        latent=tf.concat([z,cond],axis=-1)
        x = self.relu(tf.matmul(latent,self.w2)+self.b2)
        x_hat=tf.matmul(x,self.w3)+self.b3
        return x_hat
    
    def nn_model(self, x,cond):
        '''
            Defining the architecture of our model. This is where we run 
            through our whole dataset and return it, when training and 
            testing.
        '''
        mu,logsig=self.encoder(x,cond)
        z=self.sample_z(mu,logsig)
        print(z.get_shape())
        x_hat=self.decoder(z,cond)
        return mu,logsig,x_hat
    
    
    def check_model(self,z):
        return self.nn_model(z)
        

    @tf.function
    def train_step(self, x, cond):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input. We move forward first, then calculate gradients 
            with Gradient Tape to move backwards.
        '''
        with tf.GradientTape() as tape:
            mu,logsig,x_hat = self.nn_model(x,cond)
            loss1 = self.recon_loss(x, x_hat)
            loss2 = self.KL_loss(mu,logsig)
            loss=loss1+loss2
        gradients = tape.gradient(loss, self.var_list1)
        self.optimizer.apply_gradients(zip(gradients, self.var_list1))

        self.train_loss1(loss1)
        self.train_loss2(loss2)
        #self.train_metric(x, predictions)
        
    @tf.function
    def test_step(self, x,cond,in_prediction=False):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input.
        '''
        mu,logsig,x_hat = self.nn_model(x,cond)
        t_loss1 = self.recon_loss(x, x_hat)
        t_loss2 = self.KL_loss(mu,logsig)
        t_loss=t_loss1+t_loss2

        self.test_loss1(t_loss1)
        self.test_loss2(t_loss2)
        #self.test_metric(x, predictions)
        if  in_prediction:
            return mu,logsig,x_hat
    
    def fit(self, data, summaries, epochs):
        '''
            This fit function runs training and testing.
        '''
        train, val=data
        batch_size_train,N=find_batch_size(train[0].shape[0],thres=1000)
        batch_size_val,N_test=find_batch_size(val[0].shape[0],thres=900,
                                              mode='val')
        #TODO: Overridden stuff here
        batch_size_train*=8
        #batch_size_val=int(batch_size_val/2)
        
        print(batch_size_train,batch_size_val)
        train_ds=make_data_pipe(train,batch_size_train)
        val_ds=make_data_pipe(val,batch_size_val)
        
        train_summary_writer, test_summary_writer=summaries
        
        for epoch in range(epochs):
            # Reset the metrics for the next epoch
            self.train_loss1.reset_states()
            self.train_loss2.reset_states()
            self.test_loss1.reset_states()
            self.test_loss2.reset_states()
            
            #if epoch in arr_epochs:
             #   self.K.assign(arr_K[(arr_epochs==epoch)][0])
              #  print('Changed K to {}'.format(arr_K[(arr_epochs==epoch)][0]))
            for images in train_ds:
                self.train_step(images[0],images[1])
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss1.result(), step=epoch)

            for test_images in val_ds:
                self.test_step(test_images[0],test_images[1])
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss1.result(), step=epoch)

            template = ('Epoch {}, Loss1: {},Loss: {},Test Loss1: {},'
                        'Test Loss: {} \n')
            print(template.format(epoch+1,
                            self.train_loss1.result(),
                            self.train_loss1.result()+self.train_loss2.result(),
                            self.test_loss1.result(),
                            self.test_loss1.result()+self.test_loss2.result()))

            
    def predict(self,test_data):
        self.test_loss1.reset_states()
        self.test_loss2.reset_states()
        test_mu_list=[]
        test_logsig_list=[]
        test_pred_list=[]
        batch_size_test,N_test=find_batch_size(test_data[0].shape[0],thres=1024
                                               ,mode='val')
        for i in range(N_test):           
            # Reset the metrics for the next batch and test z values
            mu,logsig,x_hat=self.test_step(test_data[0][i:i+batch_size_test],
                                test_data[1][i:i+batch_size_test],
                                in_prediction=True)
            test_mu_list.append(mu)
            test_logsig_list.append(logsig)
            test_pred_list.append(x_hat)

# =============================================================================
#         for images, labels in test_data:
#             labels, predictions=self.test_step(images, labels,
#                                                in_prediction=True)
#             test_label_list.append(labels)
#             test_pred_list.append(predictions)
#         loss=self.test_loss.result()
#         #metric=self.test_metric.result()*100
#         self.test_loss.reset_states()
#         self.test_metric.reset_states()
#         
#         return [tf.concat(test_label_list,axis=0).numpy(),
#             tf.concat(test_pred_list,axis=0).numpy(),loss,metric]
# =============================================================================
        test_data.append(np.concatenate(test_mu_list,axis=0))
        test_data.append(np.concatenate(test_logsig_list,axis=0))
        test_data.append(np.concatenate(test_pred_list,axis=0))

        return test_data
        
    def make_plot(self,x,x_hat,y):
        avg_y=np.mean(y)
        freq=np.fft.fftfreq(x.shape[0])*25
        spect=np.abs(np.fft.fft(x))
        z_hat=np.abs(np.fft.fft(x_hat))
        #spect=dct(x,norm='ortho')
        #z_hat=dct(x_hat,norm='ortho')
        plt.figure()
        plt.subplot(211);plt.plot(np.array(2*[avg_y]),
                                  np.array([np.min(spect),np.max(spect)]),'k')
        plt.plot(freq,spect,'b');plt.plot(freq,np.abs(z_hat),'r--')
        plt.legend(['True avg freq.','input FFT','Predicted FFT'])
        plt.title('Signal Spectrum');plt.grid(True)
        
        plt.subplot(212);plt.plot(np.real(x),'b');plt.plot(np.real(x_hat),'r--')
        plt.legend(['True Signal','Reconstructed Signal'])
        plt.title('Time domain Signal');plt.grid(True)