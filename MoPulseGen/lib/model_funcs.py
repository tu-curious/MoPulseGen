# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:53:22 2018

@author: agarwal.270a
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

eps=1e-7

def flip(x,y):
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    return x,y

def make_plot(z_hat,z,y,x_hat,x):
    avg_y=np.mean(y)
    freq=np.fft.fftfreq(x.shape[0])*25
    spect=np.abs(z)
    plt.figure()
    plt.subplot(211);plt.plot(np.array(2*[avg_y]),
                              np.array([np.min(spect),np.max(spect)]),'k')
    plt.plot(freq,spect,'b');plt.plot(freq,np.abs(z_hat),'r--')
    plt.legend(['True avg freq.','input FFT','Predicted Sparse FFT'])
    plt.title('Signal Spectrum');plt.grid(True)
    
    plt.subplot(212);plt.plot(np.real(x),'b');plt.plot(np.real(x_hat),'r--')
    plt.legend(['True Signal','Reconstructed Signal'])
    plt.title('Time domain Signal');plt.grid(True)

def make_data_pipe_old(dataset_X_shape,dataset_Y_shape,reuse=True):
    '''
    tf data pipeline
    '''
    # dynamic dataset using placeholders
    with tf.variable_scope('Data',reuse=reuse):
        batch_size = tf.placeholder_with_default(tf.constant(64,dtype=tf.int64),shape=None,name='batch_size') # batch_size placeholder
        data_X= tf.placeholder(tf.float32, shape=[None]+list(dataset_X_shape[1:]),name='data_X')
        data_Y = tf.placeholder(tf.float32, shape=[None]+[dataset_Y_shape[1]],name='data_Y')
        #dataset = tf.data.Dataset.from_tensor_slices((data_X,data_Y))\
        #.shuffle(buffer_size=6*batch_size).batch(batch_size).repeat()
        dataset = tf.data.Dataset.from_tensor_slices((data_X,data_Y))
        dataset=dataset.shuffle(buffer_size=6*batch_size).repeat()
        #dataset=dataset.repeat()

        #Add any real_time augmentations
        #dataset = dataset.map(map_func=flip,num_parallel_calls=4)
        dataset=dataset.batch(batch_size).prefetch(2)
        
        # create a generic iterator of the correct shape and type
        iter_reini = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)
        # create the reinitialization operations, i.e. assign dataset to the iterator
        init_op = iter_reini.make_initializer(dataset,name='data_init_op')
        features, labels_Y = iter_reini.get_next()   

        return features, labels_Y
    
def make_data_pipe(data,batch_size=64,shuffle=True):
    #dataset = tf.data.Dataset.from_tensor_slices((data[0],data[1],data[2]))
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
    if shuffle:
        dataset=dataset.shuffle(buffer_size=6*batch_size)
    dataset=dataset.batch(batch_size).prefetch(2)
    return dataset


def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def smallest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i: # mean if NOT divisible
            i+=1
        else:
            return i
    return n
    #raise AssertionError("Something seems wrong. Couldn't find a prime factor.")
    #return

def find_batch_size(N,thres=1500,mode='train'):
    N_old=N*1
    if mode=='val':
        L=largest_prime_factor(N)
        #print(L)
        if L>thres:
            print('Largest factor is pretty high at {}. Be careful.'.format(L))
            return L,int(N_old/L)
    else:
        L=1
    N=N//L
    while N>=2:
        #print(N)
        l=smallest_prime_factor(N)
        #print(l)
        if L*l>thres:
            break
        else:
            L=L*l
            N=N//l
    return L,int(N_old/L)