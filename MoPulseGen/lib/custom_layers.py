# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 01:02:52 2020

@author: agarwal.270a
"""


import tensorflow as tf
from tensorflow.keras import initializers as initizers
from tensorflow.keras.layers import Conv2D
import numpy as np

def zRelu(real,imag):
    z=tf.complex(real,imag)
    theta=tf.math.angle(z)
    condit=tf.logical_and(tf.less_equal(np.float64(0),theta),
                          tf.less_equal(theta,np.pi/2))
    #ans=z * tf.cast(condit, tf.float64) #TODO: check if int or float needed
    #return tf.math.real(ans),tf.math.imag(ans)
    ans=tf.cast(condit, tf.float64)
    return ans*real,ans*imag

def cRelu(real,imag):
    return tf.nn.relu(real),tf.nn.relu(imag)

def lin(real,imag):
    return real,imag

act_dict={'linear':lin,'zRelu':zRelu,'cRelu':cRelu}

def conv2d(x,w,b,name,strides=[1,1],activation='linear',padding='VALID',
           use_bias=True):
    '''
    W.shape=[H,W,C_in,C_out]
    b.shape=[C_out]
    '''
    if activation in act_dict:
        activation=act_dict[activation]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        out=tf.nn.conv2d(x,w,strides=[1,strides[0],strides[1],1],\
                       padding=padding.upper(),name='kernel_output')
        if use_bias:
            out=tf.nn.bias_add(out,b,data_format='NHWC')

        return activation(out,name='output')

class Get_Top_K(tf.keras.layers.Layer):
    def __init__(self,K):
        super(Get_Top_K,self).__init__()
        self.K=K

    def call(self,inputs):
        #k=1
        #x = tf.Variable([[6., 2., 1.], [2., 4., 5.]])  # of type tf.float32
        # indices will be [[0, 1], [1, 2]], values will be [[6., 2.], [4., 5.]]
        values, indices = tf.math.top_k(tf.abs(inputs), self.K, sorted=False)
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, self.K])  # will be [[0, 0], [1, 1]]
        # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        #full_indices = tf.concat([tf.expand_dims(my_range_repeated, -1), tf.expand_dims(indices, -1)], axis=2)
        full_indices=tf.stack([my_range_repeated,indices],axis=-1)
        full_indices = tf.reshape(full_indices, [-1, 2])
        values=tf.reshape(values, [-1])
        # only significant modification -----------------------------------------------------------------
        active_x = tf.scatter_nd(full_indices, values, tf.shape(inputs))
        return active_x

class Complex_Dense(tf.keras.layers.Layer):

    def __init__(self, units, w_init=initizers.glorot_uniform(seed=None), 
                 b_init= initizers.Zeros(),activation='linear',use_bias=True
                 ,weights=None,**kwargs):
        super(Complex_Dense, self).__init__(**kwargs)
        self.units = units
        self.w_init=w_init
        self.b_init=b_init
        self.use_bias=use_bias
        self.activation=activation
        self.w=weights

    def build(self, input_shape):
        if self.w is None:
            self.w = self.add_weight(shape=(input_shape[-2], self.units,2),
                                     initializer=self.w_init,
                                     trainable=True)
        self.b = self.add_weight(shape=(self.units,2),
                                 initializer=self.b_init,
                                 trainable=True)

    def call(self, inputs):
        in_real=inputs[:,:,0]
        in_imag=inputs[:,:,1]
        w_real=self.w[:,:,0]
        w_imag=self.w[:,:,1]
        b_real=self.b[:,0]
        b_imag=self.b[:,1]
        
        dense_real=tf.matmul(in_real,w_real)-tf.matmul(in_imag,w_imag)
        dense_imag=tf.matmul(in_real,w_imag)+tf.matmul(in_imag,w_real)
        if self.use_bias:
            dense_real+=b_real
            dense_imag+=b_imag
        
        #activate
        f_act=act_dict[self.activation]
        dense_real,dense_imag=f_act(dense_real,dense_imag)
        return tf.stack([dense_real, dense_imag], axis=-1)


class Complex_Conv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 w_init=initizers.glorot_uniform(seed=None),activation='linear'
                 ,b_init= initizers.Zeros(),use_bias=True,weights=None,**kwargs):
        super(Complex_Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding=padding
        self.strides=strides
        self.w_init=w_init
        self.b_init=b_init
        self.use_bias=use_bias
        self.activation=activation
        self.w=weights
        if self.w is not None:
            self.real_layer=tf.keras.layers.Conv2D(filters,kernel_size,
                                                padding=padding,use_bias=False
                                                ,strides=strides,
                                                weights=self.w[:,:,:,:,0])
            self.imag_layer=tf.keras.layers.Conv2D(filters,kernel_size,
                                                padding=padding,use_bias=False
                                                ,strides=strides,
                                                weights=self.w[:,:,:,:,1])
        else:
            self.real_layer=tf.keras.layers.Conv2D(filters,kernel_size,
                                                padding=padding,use_bias=False
                                                ,strides=strides,
                                                kernel_initializer=w_init)
            self.imag_layer=tf.keras.layers.Conv2D(filters,kernel_size,
                                                padding=padding,use_bias=False
                                                ,strides=strides,
                                                kernel_initializer=w_init)
    

    def build(self, input_shape):
        #self.w = self.add_weight(shape=(input_shape[-2], self.units,2),
        #                         initializer=self.w_init,
        #                         trainable=True)
        #b_shape=tf.shape(self.real_layer.trainable_weights).numpy()[-1]
        self.b = self.add_weight(shape=[self.filters,2],
                                 initializer=self.b_init,
                                 trainable=True)

    def call(self, inputs):
        in_real=inputs[:,:,:,:,0]
        in_imag=inputs[:,:,:,:,1]
        
        b_real=self.b[:,0]
        b_imag=self.b[:,1]
        
        #pass through layers
        conv_real=self.real_layer(in_real)-self.imag_layer(in_imag)
        conv_imag=self.real_layer(in_imag)+self.imag_layer(in_real)
        if self.use_bias:
            conv_real+=b_real
            conv_imag+=b_imag
    
        self.w=tf.stack([self.real_layer.trainable_weights[0],
                          self.imag_layer.trainable_weights[0]], axis=-1)
        
        #activate
        f_act=act_dict[self.activation]
        conv_real,conv_imag=f_act(conv_real,conv_imag)
        return tf.stack([conv_real,conv_imag], axis=-1)