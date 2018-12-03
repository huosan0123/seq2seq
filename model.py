#coding=utf-8
from __future__ import print_function
from __future__ import division

import numpy
import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import config
import math, sys
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

# To be done:
# how to feed in data
class Seq2Seq(object):
    def __init__(self, mode, batch, drop_prob, gamma):
        self.mode = mode
        self.batch = batch
        self.beam_search = False
        self.beam_with = 5
        self.drop_prob = drop_prob
        self.gamma = gamma

        self.init_placeholder()
        self.build_encoder()
        self.build_decoder()
        self.summary = tf.summary.merge_all()


    def init_placeholder(self):
        # To make it compatable when train and decode, input sequence should without _GO and _EOS.
        # encoder seq [batch, max_time_step], lengths [batch,]
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, config.encoder_vocab_size], name="encoder_inputs")
        self.encoder_length = tf.placeholder(tf.float32, [None,], name="encoder_length")
        self.lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")

        if self.mode == "train":
            # decoder input only needed in training phase
            # decoder seq [batch, max_time_step], lengths [batch, ]
            self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
            self.decoder_length = tf.placeholder(tf.int32, [None, ], name="decoder_length")

            # To add _EOS, so modified inputs and lengths
            decoder_GO = tf.ones(shape=(self.batch, 1), dtype=tf.int32) * config._GO
            decoder_EOS = tf.ones(shape=(self.batch, 1), dtype=tf.int32) * config._EOS

            # input_train [batch, max_time_step+1], target_train [batch, max_time_step+1]
            self.decoder_inputs_train = tf.concat([decoder_GO, self.decoder_inputs], axis=1)
            self.decoder_length_train = self.decoder_length + 1
            self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_EOS], axis=1)


    def build_encoder(self):
        print("build encoder")
        with tf.variable_scope("encoder"):
            initializer = tf.orthogonal_initializer()
            encoder_inputs_embedded = tf.layers.dense(self.encoder_inputs, 512,
                                                      activation=tf.nn.elu,
                                                      kernel_initializer=initializer)

            encoder_cell = self.build_encoder_cell(config.hidden_size)
            (self.encoder_outputs, self.encoder_fs) = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                              sequence_length=self.encoder_length,
                                                              dtype=tf.float32, time_major=False)

    def build_decoder(self):
        print("build decoder")
        with tf.variable_scope("decoder"):
            decoder_cell, decoder_init_state = self.build_decoder_cell(config.hidden_size)
            
            initializer = tf.orthogonal_initializer()
            self.decoder_embedding = tf.get_variable(name="decoder_embedding",
                                                     shape=[config.decoder_vocab_size, config.embedding_size],
                                                     initializer=initializer, dtype=tf.float32)
            tf.summary.histogram("decoder_embed", self.decoder_embedding)

            output_layer = Dense(config.decoder_vocab_size)
            input_layer = Dense(config.hidden_size, dtype=tf.float32)

            if self.mode == "train":
                # decoder_inputs_embedded: [n, max_time_step+1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.decoder_embedding, self.decoder_inputs_train)
                #self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)
                
                train_helper = TrainingHelper(inputs=self.decoder_inputs_embedded,
                                        sequence_length=self.decoder_length_train,
                                        time_major=False, name="traing_helper")
                train_decoder = BasicDecoder(cell=decoder_cell, helper=train_helper,
                                             initial_state=decoder_init_state, output_layer=output_layer)

                self.max_decoder_length = tf.reduce_max(self.decoder_length_train)

                self.decoder_outputs_train, self.decoder_last_state_train, self.decoder_outputs_length_train = dynamic_decode(
                    decoder=train_decoder, output_time_major=False, impute_finished=True, maximum_iterations=self.max_decoder_length)

                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,)

                # mask for true and padded time steps. shape=[batch, max_length+1]
                self.masks = tf.sequence_mask(lengths=self.decoder_length_train,
                                         maxlen=self.max_decoder_length, dtype=tf.float32, name="masks")

                # decoder_logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # decoder_targets_train: [batch_size, max_time_steps + 1]
                self.loss1 = sequence_loss(logits=self.decoder_logits_train, targets=self.decoder_targets_train,
                                          weights=self.masks, average_across_timesteps=True,
                                          average_across_batch=True)
                tf.summary.scalar('loss', self.loss1)

                # build hashtable to get predicted seq and target seq mass
                self.keys, self.values = config.IDs, config.MASSes
                self.table = tf.contrib.lookup.HashTable(
                    tf.contrib.lookup.KeyValueTensorInitializer(self.keys, self.values, key_dtype=tf.int64), -1)

                # shape of this two should be [batch, max_len]
                pred_mass = self.table.lookup(tf.cast(self.decoder_pred_train, dtype=tf.int64))
                target_mass = self.table.lookup(tf.cast(self.decoder_targets_train, dtype=tf.int64))
                # compute l2 loss of mass at seq's every position
                # pred_cum_mass = tf.cast(tf.cumsum(pred_mass, axis=-1), dtype=tf.float32)
                # target_cum_mass = tf.cast(tf.cumsum(target_mass, axis=-1), dtype=tf.float32)
                diff_cum_l2 = tf.square( (pred_mass - target_mass) * self.masks )
                row_sum = tf.reduce_sum(diff_cum_l2, axis=-1) / tf.cast(self.decoder_length_train, tf.float32)
                self.loss2 = tf.reduce_mean(row_sum)

                self.loss = self.loss1 + self.gamma * self.loss2
                
                #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                opt = tf.train.AdamOptimizer(self.lr)
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
                self.train_op = opt.apply_gradients(zip(clipped_gradients, params))


            elif self.mode == "decode":
                start_tokens = tf.ones([self.batch,], dtype=tf.int32) * config._GO
                end_token = config._EOS
                def embed_input(inputs):
                    return tf.nn.embedding_lookup(self.decoder_embedding, inputs)
                if self.beam_search:
                    pred_decoder = beam_search_decoder.BeamSearchDecoder(cell=decoder_cell,
                                                                        embedding=embed_input,
                                                                        start_tokens=start_tokens,
                                                                        end_token=end_token,
                                                                        initial_state=decoder_init_state,
                                                                         beam_width=self.beam_with,
                                                                         output_layer=output_layer,)

                else:
                    decoding_helper = GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                            end_token=end_token,
                                                            embedding=embed_input)
                    pred_decoder = BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                initial_state=decoder_init_state,
                                                output_layer=output_layer)

                self.decoder_outputs_decode, self.decoder_last_state_decode, self.decoder_outputs_length_decode = dynamic_decode(
                                            decoder=pred_decoder, output_time_major=False, maximum_iterations=52)

                if self.beam_search:
                    self.pred_id = self.decoder_outputs_decode.predicted_ids
                else:
                    self.pred_id = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                self.shape = tf.shape(self.pred_id)
                if isinstance(self.decoder_last_state_decode, tf.Tensor):
                    self.ls_shape = tf.shape(self.decoder_last_state_decode)
                else:
                    print("not a tensor")
                #self.lg_shape = tf.shape(self.decoder_outputs_length_decode)


    def build_encoder_cell(self, num_units):
        layer1 = LSTMCell(num_units)
        layer2 = LSTMCell(num_units)
        # layer2 = tf.nn.rnn_cell.DropoutWrapper(layer2, output_keep_prob=self.drop_prob)
        layers = [layer1, layer2]
        return tf.nn.rnn_cell.MultiRNNCell(layers)
    

    def build_decoder_cell(self, num_units):
        encoder_outputs = self.encoder_outputs
        encoder_fs = self.encoder_fs
        encoder_length = self.encoder_length

        # use Bahdanua attention first
        self.attention_machenism = attention_wrapper.BahdanauAttention(num_units=num_units,
                                    memory=encoder_outputs,memory_sequence_length=encoder_length)

        
        # build decoder cell
        layer1 = LSTMCell(num_units)
        layer2 = LSTMCell(num_units)
        # layer2 = tf.nn.rnn_cell.DropoutWrapper(layer2, output_keep_prob=self.drop_prob)
        decoder_cells = [layer1, layer2]
        decoder_init_state = encoder_fs

        decoder_cells[-1] = attention_wrapper.AttentionWrapper(decoder_cells[-1], self.attention_machenism,
                                            attention_layer_size=num_units, 
                                            initial_cell_state=encoder_fs[-1],
                                            alignment_history=False,)
        
        # why the last layers' zero state different with
        init_state = [state for state in encoder_fs]
        init_state[-1] = decoder_cells[-1].zero_state(self.batch, dtype=tf.float32)
        decoder_init_state = tuple(init_state)

        return tf.nn.rnn_cell.MultiRNNCell(decoder_cells), decoder_init_state
