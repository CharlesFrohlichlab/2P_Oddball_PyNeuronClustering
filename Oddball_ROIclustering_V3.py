# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from __future__ import print_function

import IPython
print('IPython:', IPython.__version__)

import numpy as np
print('numpy:', np.__version__)

import scipy
print('scipy:', scipy.__version__)

import matplotlib
print('matplotlib:', matplotlib.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)

import seaborn
print('seaborn', seaborn.__version__)

import scipy.io as sio
print('scipy', scipy.__version__)

import h5py
print('h5py', h5py.__version__)

import pandas as pd

import matplotlib.cm as cm

import pylab as plt

###

animalNum = 'All';
mat = scipy.io.loadmat('E:/2pdata/GroupLevel/evtRelated_devStandCtrl/1/allAnimals_cluster')
#mat = scipy.io.loadmat('E:/2pdata/GroupLevel/evtRelated_devStandCtrl/1/session/0148_VisualOddball_201802091_cluster')
#print(mat)

##

data = mat.get('toPythonClustAll') 
numSamples = len(data)
print( str(numSamples) + ' ROIs')
# print(data)