# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:35:59 2020

@author: agarwal.270a
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import tensorflow.keras.layers as layers
#from lib.custom_layers.utils import z_Mag
#from tensorflow.keras import initializers as initizers
from lib.model_funcs import make_data_pipe, find_batch_size
import numpy as np
#import copy
import matplotlib.pyplot as plt
import inspect
#tf.keras.backend.set_floatx('float64')

def modify_get_data(get_data):
    arg_list=inspect.getfullargspec(get_data).args
    if 'n_sins' in arg_list:
        n_sins_flag=True
    else:
        n_sins_flag=False
    def new_get_data(N_samples):
        if n_sins_flag:
            n_sins=np.random.choice(np.arange(1,6))
            all_returns=get_data(N_samples,n_sins=n_sins) #call get data
        else:
            all_returns=get_data(N_samples) #call get data
        time_sig=all_returns[-1] #take only the last return value
        
        return [np.stack([time_sig,np.zeros_like(time_sig)],axis=-1)]
    
    return new_get_data


        
class Model_GAN(tf.keras.Model):
    def __init__(self,gen,disc,model_path,get_data_old,lamda=1.):
        '''
            Setting all the variables for our model.
        '''
        super(Model_GAN, self).__init__()
        self.gen=gen
        self.disc=disc
        self.model_path=model_path

        self.get_data=modify_get_data(get_data_old)
        self.lamda=lamda
        
        #'Stateful' Metrics
        self.train_loss_gen = tf.keras.metrics.Mean(name='train_loss_gen')
        self.train_loss_disc = tf.keras.metrics.Mean(name='train_loss_disc')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        #self.test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
        #self.test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
        #self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        #'Stateless' Losses
        self.loss_cc=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
        #self.l1_loss=lambda z: tf.reduce_mean(tf.abs(z))
        self.acc= lambda y,y_hat: tf.reduce_mean(tf.cast(tf.equal(
                    tf.argmax(y,axis=1),tf.argmax(y_hat,axis=1)),tf.float64))

        
        #Checkpoint objects
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
                                        optimizer=self.optimizer,
                                        model=self.net)
        self.manager = tf.train.CheckpointManager(self.ckpt,self.model_path 
                                                  ,max_to_keep=2)
        
        #For fit function initialization
        self.fit_init=False
    
    
    @tf.function
    def train_step(self, x):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input. We move forward first, then calculate gradients 
            with Gradient Tape to move backwards.
        '''
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        generator,discriminator=self.gen,self.disc
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
        
            real_output = discriminator(x, training=True)
            fake_output = discriminator(generated_images, training=True)
        
            gen_loss = self.gen.loss(fake_output)
            disc_loss = self.disc.loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, 
                                            generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, 
                                            discriminator.trainable_variables)
        
        self.gen.optimizer.apply_gradients(zip(gradients_of_generator, 
                                          generator.trainable_variables))
        self.disc.optimizer.apply_gradients(zip(gradients_of_discriminator, 
                                           discriminator.trainable_variables))

        self.train_loss_gen(gen_loss)
        self.train_loss_disc(disc_loss)
        #self.train_metric(x, predictions)
        
    def test_step(self, x,in_prediction=False,epoch=0):
        '''
            This is a TensorFlow function, run once for each epoch for the
            whole input.
        '''
        x_hat = self.gen(x,training=False)
        fig = plt.figure(figsize=(4,4))

        for i in range(min(predictions.shape[0],8)):
            plt.subplot(4,2, i+1)
            plt.plot(x_hat);plt.plot(x,'r--')
            plt.legend(['Filtered','True'])
            
        if  in_prediction:
            return x_hat
        else:
            plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
            return
        
    @tf.function
    def val_step(self, x,in_prediction=False,epoch=0):
        return self.test_step(x,in_prediction,epoch)
        
# =============================================================================
#     def get_data(self,N_samples,n_sins=2):
#         f_samples=np.random.uniform(low=self.net.f_min,high=self.net.f_max,
#                                     size=(N_samples,1,n_sins))
#         amp_samples=np.random.uniform(low=0.5,high=5,size=(N_samples,1,n_sins))
#         #freq = np.fft.fftfreq(freq_d,d=1/Fs).reshape(-1,1)
#         #freq=np.linspace(0,Fs,freq_d).reshape(-1,1)
#         #sample for check
#         t = (1/self.net.Fs)*np.arange(self.net.Fs).reshape(1,-1,1)
#         time_sig=amp_samples*np.sin(2*np.pi*f_samples*t)
#         time_sig=np.sum(time_sig,axis=-1).astype(np.float32)
#         #form freqs in correct amp ratios
#         #amp_samples/=np.max(amp_samples,axis=-1,keepdims=True)
#         #f_samples*=amp_samples
#         #return t,f_samples,amp_samples,time_sig
#         return [np.stack([time_sig,np.zeros_like(time_sig)],axis=-1)]
# =============================================================================
    
    def fit(self,summaries, epochs):
        '''
        This fit function runs training and testing.
        '''
        epochs2regen=10
        #train, val=data
        #batch_size_train,N=find_batch_size(train[0].shape[0],thres=1000)
        #batch_size_val,N_test=find_batch_size(val[0].shape[0],thres=900,
        #                                      mode='val')
        
        N_samples_train=1024*8
        N_samples_val=1024*2
        batch_size_train,_=64,int(N_samples_train/64)
        batch_size_val,_=256,int(N_samples_val/256)
        #TODO: Overridden stuff here
        #batch_size_train*=8
        #batch_size_val=int(batch_size_val/2)
        
        #print(batch_size_train,batch_size_val)
        
        train_summary_writer, test_summary_writer=summaries
        
        for epoch in range(epochs):
            # Reset the metrics for the next epoch
            self.train_loss1.reset_states()
            self.train_loss2.reset_states()
            self.train_loss.reset_states()
            self.test_loss1.reset_states()
            self.test_loss2.reset_states()
            self.test_loss.reset_states()
            if epoch%epochs2regen==0:
                print('Regenerating Data...')
                train=self.get_data(N_samples_train)
                val=self.get_data(N_samples_val)
                train_ds=make_data_pipe(train,batch_size_train,training=True)
                val_ds=make_data_pipe(val,batch_size_val,training=False)
            
            
            #========================

            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)
        
            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)
        
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
        
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
          # Generate after the final epoch
          display.clear_output(wait=True)
          generate_and_save_images(generator,
                                   epochs,
                                   seed)
          #========================
            #if epoch in arr_epochs:
             #   self.K.assign(arr_K[(arr_epochs==epoch)][0])
              #  print('Changed K to {}'.format(arr_K[(arr_epochs==epoch)][0]))
            for images in train_ds:
                self.train_step(images[0])
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss1.result(), step=epoch)

            for test_images in val_ds:
                self.test_step(test_images[0])
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss1.result(), step=epoch)

            template = ('Epoch {}, Loss1: {},Loss: {},Test Loss1: {},'
                        'Test Loss: {} \n')
            print(template.format(epoch+1,
                            self.train_loss1.result(),
                            self.train_loss.result(),
                            self.test_loss1.result(),
                            self.test_loss.result()))

            
    def predict(self,test_data):
        self.test_loss1.reset_states()
        self.test_loss2.reset_states()
        self.test_loss.reset_states()
        test_z_list=[]
        test_pred_list=[]
        batch_size_test,N_test=find_batch_size(test_data[0].shape[0],thres=1024
                                               ,mode='val')
        for i in range(N_test):           
            # Reset the metrics for the next batch and test z values
            z,x_hat=self.test_step(test_data[0][i:i+batch_size_test],
                                in_prediction=True)
            test_z_list.append(z)
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
        test_data.append(np.concatenate(test_z_list,axis=0))
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
        
#%%
if __name__=='__main__':
    