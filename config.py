#coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

# ID for  GO（<s>）, EOS(</s>)
# note that: in stanford's dataset, id of <unk> is 0
_GO = 1
_EOS = 2


# this vocab_size include UNK GO EOS
encoder_vocab_size = 17191
decoder_vocab_size = 7709

embedding_size = 512
hidden_size = 512

num_layers = 2
