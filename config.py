#coding=utf-8
from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf


_GO = 0
_EOS = 1
AAs = ['_GO', '_EOS', 'A','R','N','Nmod','D','Cmod',
       'E','Q','Qmod','G','H','I','L','K','M','Mmod',
       'F','P','S','T','W','Y','V']
ID_AA = {i: aa for i, aa in enumerate(AAs)}
AA_ID = {aa : i for i, aa in enumerate(AAs)}

IDs = np.array([i for i, aa in enumerate(AAs)], dtype=np.int64)

###############################
#       Mass
###############################
mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078   # for H
mass_C_terminus = 17.0027  # for OH
mass_CO = 27.9949

mass_AAs = {'_GO': 0.0,
            '_EOS': 0.0,
            'A':71.03711,  # 0
            'R':156.10111,  # 1
            'N':114.04293,  # 2
            'Nmod':115.02695,
            'D':115.02694,  # 3
            #~ 'C':103.00919, # 4
            'Cmod':160.03065,  # C(+57.02)
            #~ 'Cmod':161.01919, # C(+58.01) # orbi
            'E':129.04259,  # 5
            'Q':128.05858,  # 6
            'Qmod':129.0426,
            'G':57.02146,  # 7
            'H':137.05891,  # 8
            'I':113.08406,  # 9
            'L':113.08406,  # 10
            'K':128.09496,  # 11
            'M':131.04049,  # 12
            'Mmod':147.0354,
            'F':147.06841,  # 13
            'P':97.05276,  # 14
            'S':87.03203,  # 15
            'T':101.04768,  # 16
            'W':186.07931,  # 17
            'Y':163.06333,  # 18
            'V':99.06841,  # 19
          }
MASSes = np.array([mass_AAs[aa] for aa in AAs], dtype=np.float32)

MAX_MZ = 3000.0
INPUT_DIFF_TOLERANCE = 0.01
RESOLUTION = 10
direction = 1

encoder_vocab_size = int(MAX_MZ * RESOLUTION)
embedding_size = 1024
hidden_size = 1024
decoder_vocab_size = 25