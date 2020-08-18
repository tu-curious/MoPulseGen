# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:35:59 2020

@author: agarwal.270a
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers
#import modules.custom_layers as clayers
from tensorflow.keras import initializers as initizers
from lib.model_funcs import make_data_pipe, find_batch_size
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
#tf.keras.backend.set_floatx('float64')

class Generator(tf.keras.layers.Layer):
    def __init__(self,layer_list,optimizer):
        super(Generator, self).__init__()
        self.layer_list=layer_list
        if optimizer is not None:
            self.optimizer=optimizer
        else:
            self.optimizer=tf.keras.optimizers.Adam(1e-4)
        self.bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)
            
    def loss(self,fake_output):
        return self.bc(tf.ones_like(fake_output), fake_output)

    def call(self,x,training=None):
        for lay in self.layer_list:
            x=lay(x,training=training)
            #print(x.shape.as_list())
        return x
    
class Discriminator(tf.keras.layers.Layer):
    def __init__(self,layer_list,optimizer):
        super(Discriminator, self).__init__()
        self.layer_list=layer_list
        if optimizer is not None:
            self.optimizer=optimizer
        else:
            self.optimizer=tf.keras.optimizers.Adam(1e-4)
            
        self.bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss(self,real_output, fake_output):
        real_loss = self.bc(tf.ones_like(real_output), real_output)
        fake_loss = self.bc(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def call(self,x,training=None):
        for lay in self.layer_list:
            x=lay(x,training=training)
            #print(x.shape.as_list())
        return x
    
class Net_stitchGAN(tf.keras.layers.Layer):
    def __init__(self,in_shape,out_shape,optimizers=[None,None],drop=0.2):
        super(Net_stitchGAN, self).__init__()
        self.in_shape=in_shape
        self.out_shape=out_shape
        
        if type(optimizers)!=type([]):
            raise AssertionError(('optimizers must be a list of 2 optimizers'
                                 ', one each for generator and discrimator'
                                 'respectively.'))

        self.optimizers=optimizers

        self.GenL=[]
        self.GenL.append(layers.GRU(64,return_sequences=True,name='gru_gen_1'))
        self.GenL.append(layers.Conv1D(1,1,name='conv1d_gen_1'))
        self.gen=Generator(self.GenL,self.optimizers[0])
        
        self.DiscL=[]
        self.DiscL.append(layers.GRU(64,name='gru_disc_1'))
        self.DiscL.append(layers.Dropout(0.3,name='drop_disc_1'))
        self.DiscL.append(layers.Flatten(name='flat_disc_1'))
        self.DiscL.append(layers.Dense(1,name='fc_disc_1'))
        self.disc=Discriminator(self.DiscL,self.optimizers[1])
        
        return

    def call(self, x, training=None):
        '''
        Defining the architecture of our model. This is where we run 
        through our whole dataset and return it, when training and 
        testing.
        '''
        x=self.gen(x,training=training)
        x=self.disc(x,training=training)
        return x
    
class Model_stitchGAN(tf.keras.Model):
    def __init__(self,net,model_path,mode='stitch'):
        '''
            Setting all the variables for our model.
        '''
        super(Model_stitchGAN, self).__init__()
        self.net=net
        self.model_path=model_path
        self.mode=mode
        #self.optimizer=self.net.optimizer
        #self.get_data=modify_get_data(get_data_old)
        
        #'Stateful' Metrics
        self.train_loss1 = tf.keras.metrics.Mean(name='train_loss1')
        self.train_loss2 = tf.keras.metrics.Mean(name='train_loss2')
        #self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        #self.test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
        #self.test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        #'Stateless' Losses
        self.loss_bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        #self.l1_loss=lambda z: tf.reduce_mean(tf.abs(z))
        self.acc= lambda y,y_hat: tf.reduce_mean(tf.cast(tf.equal(
                    tf.argmax(y,axis=1),tf.argmax(y_hat,axis=1)),tf.float64))

        
        #Checkpoint objects
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
                                        optimizer=self.net.gen.optimizer,
                                        model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt,self.model_path 
                                                  ,max_to_keep=2)
        
        #For fit function initialization
        self.fit_init=False
        return
    

    
    
# =============================================================================
#     def recon_loss(self,x_true,x_pred):
#         # E[log P(X|z)]
#         recon = tf.reduce_sum(tf.square(x_true - x_pred),axis=1)
#         return tf.reduce_sum(recon,axis=0)
#         
#     def KL_loss(self,mu,logsigma):
#         # D_KL(Q(z|X) || P(z|X))
#         kl = 0.5 * tf.reduce_sum(tf.exp(logsigma) + tf.square(mu) - 1. - 
#                                  logsigma, axis=1)
#         return tf.reduce_sum(kl,axis=0)
#     
#     def sample_z(self,mu,logsig):
#         eps=tf.random.normal(shape=tf.shape(mu),mean=0.,stddev=1.,
#                              dtype=tf.dtypes.float64)
#         z=mu + tf.exp(logsig / 2) * eps
#         return z
#     
#     def encoder(self,x,cond):
#         cond=tf.pow(tf.cast(cond,tf.float64),-1) #invert to get HR in BPS
#         VAE_in=tf.concat([x,cond],axis=-1)
#         x = self.relu(tf.matmul(VAE_in,self.w1)+self.b1)
#         mu = tf.matmul(x,self.w_mu)+self.b_mu
#         logsig = tf.matmul(x,self.w_logsig)+self.b_logsig
#         return mu,logsig
#     
#     def decoder(self,z,cond):
#         cond=tf.pow(tf.cast(cond,tf.float64),-1) #invert to get HR in BPS
#         latent=tf.concat([z,cond],axis=-1)
#         x = self.relu(tf.matmul(latent,self.w2)+self.b2)
#         x_hat=tf.matmul(x,self.w3)+self.b3
#         return x_hat
#     
#     def nn_model(self, x,cond):
#         '''
#             Defining the architecture of our model. This is where we run 
#             through our whole dataset and return it, when training and 
#             testing.
#         '''
#         mu,logsig=self.encoder(x,cond)
#         z=self.sample_z(mu,logsig)
#         print(z.get_shape())
#         x_hat=self.decoder(z,cond)
#         return mu,logsig,x_hat
#     
#     
#     def check_model(self,z):
#         return self.nn_model(z)
# =============================================================================

    @tf.function
    def train_step_stitch(self,interm_sig,true_sig):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input. We move forward first, then calculate gradients 
            with Gradient Tape to move backwards.
        '''
        #noise = tf.random.normal([BATCH_SIZE, noise_dim])
        generator=self.net.gen
        
        with tf.GradientTape() as gen_tape:
            sig_hat = generator(interm_sig, training=True)
            
            stitch_loss = self.loss_mse(true_sig,sig_hat)
    
        gradients = gen_tape.gradient(stitch_loss, generator.trainable_variables)    
        generator.optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        self.train_loss1(stitch_loss)
        
    @tf.function
    def train_step(self,raw_synth_sig,true_sig):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input. We move forward first, then calculate gradients 
            with Gradient Tape to move backwards.
        '''
        #noise = tf.random.normal([BATCH_SIZE, noise_dim])
        generator,discriminator=self.net.gen,self.net.disc
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            synth_sig = generator(raw_synth_sig, training=True)
            
            real_output = discriminator(true_sig, training=True)
            fake_output = discriminator(synth_sig, training=True)
            
            gen_loss = generator.loss(fake_output)
            disc_loss = discriminator.loss(real_output, fake_output)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
        generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        self.train_loss1(gen_loss)
        self.train_loss2(disc_loss)
        return

    
    def test_step_stitch(self,interm_sig,true_sig=None, in_prediction=False):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input.
        '''
        generator=self.net.gen
        sig_hat = generator(interm_sig, training=False)
        if  in_prediction:
            return sig_hat
        stitch_loss = self.loss_mse(true_sig,sig_hat)
        self.test_loss(stitch_loss)
        #self.test_metric(x, predictions)
        return
    
    def test_step(self,raw_synth_sig,true_sig=None, in_prediction=False):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input.
        '''
        generator,discriminator=self.net.gen,self.net.disc
        synth_sig = generator(raw_synth_sig, training=False)
        if  in_prediction:
            return synth_sig
        fake_output = discriminator(synth_sig, training=False)
        real_output = discriminator(true_sig, training=False)
        gen_loss_fake = generator.loss(fake_output)
        gen_loss_real = generator.loss(real_output)
        test_loss=tf.abs(gen_loss_real-gen_loss_fake)
        self.test_loss(test_loss)
        #self.test_metric(x, predictions)
        return
    
    @tf.function
    def val_step_stitch(self, interm_sig,true_sig,in_prediction=False):
        return self.test_step_stitch(interm_sig,true_sig,
                                     in_prediction=in_prediction)

    @tf.function
    def val_step(self, raw_synth_sig,true_sig,in_prediction=False):
        return self.test_step(raw_synth_sig,true_sig,in_prediction=in_prediction)
    
    def fit(self, data, summaries, epochs):
        '''
            This fit function runs training and testing.
        '''
        if self.mode=='stitch':
            train_step,val_step=self.train_step_stitch,self.val_step_stitch
            template = ('Epoch {}, Train_Loss: {},Val Loss: {},'
                        'Time used: {} \n')
        else:
            train_step,val_step=self.train_step,self.val_step
            template = ('Epoch {}, Gen_Loss: {}, Disc_Loss: {},Val Loss: {},'
                        'Time used: {} \n')
            
        train, val=data
        batch_size_train,N=find_batch_size(train[0].shape[0],thres=1000)
        batch_size_val,N_test=find_batch_size(val[0].shape[0],thres=900,
                                              mode='val')
        #TODO: Overridden stuff here
        #batch_size_train*=8
        #batch_size_val=int(batch_size_val/2)
        
        print(batch_size_train,batch_size_val)
        train_ds=make_data_pipe(train,batch_size_train)
        val_ds=make_data_pipe(val,batch_size_val)
        
        train_summary_writer, test_summary_writer=summaries
        
        for epoch in range(epochs):
            start = time.time()
            # Reset the metrics for the next epoch
            self.train_loss1.reset_states()
            self.train_loss2.reset_states()
            self.test_loss.reset_states()
            #self.test_loss2.reset_states()
            
            #if epoch in arr_epochs:
             #   self.K.assign(arr_K[(arr_epochs==epoch)][0])
              #  print('Changed K to {}'.format(arr_K[(arr_epochs==epoch)][0]))
            for sigs in train_ds:
                train_step(sigs[0],sigs[1])
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss1.result(), step=epoch)

            for test_sigs in val_ds:
                val_step(test_sigs[0],test_sigs[1])
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)

            if self.mode=='stitch':
                print(template.format(epoch+1,
                                self.train_loss1.result(),
                                self.test_loss.result(),
                                time.time()-start))
            else:
                print(template.format(epoch+1,
                self.train_loss1.result(),
                self.train_loss2.result(),
                self.test_loss.result(),
                time.time()-start))

            
    def predict(self,test_data):
        if self.mode=='stitch':
            test_step=self.test_step_stitch
        else:
            test_step=self.test_step

        self.test_loss.reset_states()
        test_synth_sig_list=[]
        batch_size_test,N_test=find_batch_size(test_data[0].shape[0],thres=1024
                                               ,mode='val')
        for i in range(N_test):           
            # Reset the metrics for the next batch and test z values
            synth_sig=test_step(test_data[0][i:i+batch_size_test],
                                     in_prediction=True)
            test_synth_sig_list.append(synth_sig)

        #test_data.append(np.concatenate(test_synth_sig_list,axis=0))
        #return test_data
        return np.concatenate(test_synth_sig_list,axis=0)
    
    def call(self,x):
        return self.net(x)
        
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