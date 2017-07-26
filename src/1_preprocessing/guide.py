# Guide for data processing
import numpy as np
from tools import read_data
import scipy.io as sio

# Load data
acous_train = '../../data/utterance1/train/acoustics_train.mat'
artic_train = '../../data/utterance1/train/articulation_train.mat'

acous_raw, artic_raw, labels = read_data.load_data(acous_train, artic_train)
# acous_raw and artic_raw is list variable
# each element in the list is a numpy array
acous_dim = acous_raw[0].shape[0]
artic_dim = artic_raw[0].shape[0]

print('Acoustics data: {:d}'.format(len(acous_raw)))
print('Acoustics dimension: {:d}\n'.format(acous_dim))
print('Articulation data: {:d}'.format(len(artic_raw)))
print('Articulation dimension: {:d}\n'.format(artic_dim))

# Zero padding
# (batch size x max length x features)
acous_padded = read_data.zero_padding(acous_raw)
artic_padded = read_data.zero_padding(artic_raw)
print('padded acoustics:', acous_padded.shape)
print('padded articulation:', artic_padded.shape)

