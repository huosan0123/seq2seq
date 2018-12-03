#!coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, time
import math
import config
import pickle
import numpy as np
import prepare

def read_pickle(pickle_file):
    # the data in pickle_file is a dict.
    # {'pep_masses': pep_masses, 'mz_lists':mz_lists,
    #   'intensity_lists': intensity_lists}
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset['peptides'], dataset['mz_lists'], dataset['intensity_lists']
    # don't need this for now: dataset['pep_masses'],


class DataLoader():
    def __init__(self, pickle_file, batch, epoch, direction=1):
        self.batch = batch
        self.dire = direction
        self.upper = int(config.MAX_MZ * 10)
        self.epoch = epoch
        self.peptides, self.mz_lists, self.intensity_lists = read_pickle(pickle_file)
        self.aa_2id = config.AA_ID


    def prepare_batch(self, batch_ids):
        # 将seq的aa名字换成id， 赋给对应位置的batch: data for decoder
        batch_peps = [self.peptides[i] for i in batch_ids]
        batch_lens = np.array([len(seq) for seq in batch_peps], dtype=np.int32)
        max_len = np.max(batch_lens)
        batch_seqs = np.ones((self.batch, max_len), dtype=np.int32) * config._EOS

        # data for encoder
        batch_mzs = [self.mz_lists[i] for i in batch_ids]
        batch_intensities = [self.intensity_lists[i] for i in batch_ids]
        batch_spectra_len = np.array([len(peaks) for peaks in batch_mzs], dtype=np.int32)
        max_vec_len = np.max(batch_spectra_len)
        batch_spectra = np.zeros((self.batch, max_vec_len, self.upper), dtype=np.float32)

        for i in range(self.batch):
            for j, aa in enumerate(batch_peps[i]):
                batch_seqs[i][j] = config.AA_ID[aa]
            for j, mz in enumerate(batch_mzs[i]):
                index = int(mz * config.RESOLUTION)
                batch_spectra[i][j][index] = batch_intensities[i][j]

        return batch_spectra, batch_spectra_len, batch_seqs, batch_lens

    def data_iter(self):
        assert len(self.peptides) == len(self.mz_lists)
        assert len(self.mz_lists) == len(self.intensity_lists)
        peaks_num = [len(mzs) for mzs in self.mz_lists]
        ids = np.argsort(peaks_num)[::-1]

        i, batch_i, sample_i = 0, 0, 0
        batch_ids = []
        print(len(ids) / self.batch, len(ids))
        while i < self.epoch:
            while len(batch_ids) < self.batch:
                batch_ids.append(ids[sample_i])
                sample_i += 1
                # 当一个epoch结束，epoch 数目 +1， 重置sample_i
                if sample_i == len(ids):
                    i += 1
                    print("epoch %d is done"% i)
                    sample_i = 0
            if i == self.epoch:
                break

            # 调用函数给处理每个seq，赋值给batch
            s = time.time()

            batch_peps = [self.peptides[i] for i in batch_ids]
            batch_id_peps = [[config.AA_ID[aa] for aa in pep] for pep in batch_peps]
            batch_lens = np.array([len(seq) for seq in batch_peps], dtype=np.int32)
            max_aa = np.max(batch_lens)
            batch_seqs = prepare.process_seq(batch_id_peps, self.batch, max_aa, config._EOS)
            
            batch_mzs = [self.mz_lists[i] for i in batch_ids]
            batch_intensities = [self.intensity_lists[i] for i in batch_ids]
            batch_spectra_len = np.array([len(peaks) for peaks in batch_mzs], dtype=np.int32)
            max_peak = np.max(batch_spectra_len)
            
            batch_spectra = prepare.process_peak(batch_mzs, batch_intensities, 
                                                self.batch, max_peak, config.RESOLUTION)
            yield batch_spectra, batch_spectra_len, batch_seqs, batch_lens
            print(time.time() - s)
            # reset batch to empty
            batch_ids = []



if __name__ == "__main__":
    a = DataLoader("test.pickle", 16, 1)
    i = 0
    ss = time.time()
    for b, c, d, e in a.data_iter():
        # print(b)
        i += 1
        print(b.shape, d.shape)
    print("total = ", time.time() - ss)
