# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:34:42 2020

@author: agarwal.270a
"""

import pickle

#open the file and get the fig object list
filepath='./figures/fig_REPv2.pickle'
with open(filepath, 'rb') as file:
    fig_list=pickle.load(file)

#show the figures. Maybe automatic in IDE like spyder
for fig in fig_list:
    fig.show()