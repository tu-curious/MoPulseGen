# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:44:30 2020

@author: agarwal.270a
"""

import tensorflow as tf
#import numpy as np
#import lib.custom_layers as clayers
import tensorflow.keras.layers as layers


class Net_Gen(tf.keras.layers.Layer):
    def __init__(self,optimizer=tf.keras.optimizers.Adam(),drop=0.2):
        super(Net_Gen, self).__init__()
        self.optimizer=optimizer

        #Define Layers in order
        self.GenB=[]
        self.GenB.append(layers.GRU(64,return_sequences=True))
        self.GenB.append(layers.Conv1D(1,1,return_sequences=True))
        
        self.loss_bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)

        
    def loss(self,fake_output):
        return self.loss_bc(tf.ones_like(fake_output), fake_output)
        
    def call(self,x,training=None):
        '''
        Defining the architecture of our model. This is where we run 
        through our whole dataset and return it, when training and 
        testing.
        '''
        i=0
        x=self.GenB[i](x)
        i+=1
        x=self.GenB[i](x)
        return x
    

class Net_Disc(tf.keras.layers.Layer):
    def __init__(self,optimizer=tf.keras.optimizers.Adam(),drop=0.2):
        super(Net_Disc, self).__init__()
        self.optimizer=optimizer

        #Define Layers in order

        self.DiscB=[]
        self.DiscB.append(layers.GRU(64,return_state=True))
        self.DiscB.append(layers.Dropout(drop))
        self.DiscB.append(layers.Dense(1))
        
        self.loss_bc=tf.keras.losses.BinaryCrossentropy(from_logits=True)

        
    def loss(self,real_output, fake_output):
        real_loss = self.loss_bc(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_bc(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def call(self,x,training=False):
        '''
        Defining the architecture of our model. This is where we run 
        through our whole dataset and return it, when training and 
        testing.
        '''
        i=0
        _,x=self.DiscB[i](x)#Change for LSTM
        i+=1
        x=self.DiscB[i](x,training=training)
        i+=1
        x=self.DiscB[i](x)
        return x