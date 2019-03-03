#coding=utf-8
import numpy as np
import config
import pickle
import time

class DataLoader(object):
    def __init__(self, source_file, target_file, epochs, batch, num_gpu, mode="train"):
        self.source_file = source_file
        self.target_file = target_file
        self.epochs = epochs
        self.batch = batch
        self.mode = mode
        self.gpu = num_gpu

    def prepare_data(self):
        with open(self.source_file, 'rb') as f:
            x_set = pickle.load(f)
        with open(self.target_file, 'rb') as f:
            y_set = pickle.load(f)

        x_lens = [len(sentence) for sentence in x_set]
        y_lens = [len(sentence) for sentence in y_set]
        return x_set, y_set, x_lens, y_lens


    def data_iter(self):
        epochs, batch_size = self.epochs, self.batch
        x_set, y_set, x_lens, y_lens = self.prepare_data()
        pos, num_sentence = 0, len(x_lens)
        if self.mode == 'train':
            print("batch size on every GPU is %d" % self.batch)
            print("iter num per epoch =", num_sentence/(batch_size*self.gpu))

        tower = []
        for e in range(epochs):
            if self.mode == 'train':
                print("epochs = ", e)
            while batch_size + pos < num_sentence:
                batch_x_lens = x_lens[pos: batch_size+pos]
                batch_y_lens = y_lens[pos: batch_size+pos]
                max_len_x, max_len_y = max(batch_x_lens), max(batch_y_lens)

                x = np.ones((batch_size, max_len_x)).astype('int32') * config._EOS
                y = np.ones((batch_size, max_len_y)).astype('int32') * config._EOS

                for i in range(batch_size):
                    x[i, :batch_x_lens[i]] = x_set[pos+i]
                    y[i, :batch_y_lens[i]] = y_set[pos+i]

                tower.append([x, batch_x_lens, y, batch_y_lens])
                if len(tower) == self.gpu:
                    yield tower
                    tower = []

                pos += batch_size
            pos = 0
