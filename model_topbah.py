#coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import config
from dataloader import *
import sys
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

# To be done:
# how to feed in data
class Seq2Seq(object):
    def __init__(self, mode, batch):
        self.mode = mode
        self.batch = batch
        self.beam_search = False
        self.beam_with = 5

        # self.build_encoder()
        # self.build_decoder()
        # self.summary = tf.summary.merge_all()

        # To make it compatable when train and decode, input sequence should without _GO and _EOS.
        # encoder seq [batch, max_time_step], lengths [batch,]

    def init_placeholder(self):
        with tf.variable_scope("inputs"):
            self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
            self.encoder_length = tf.placeholder(tf.int32, [None,], name="encoder_length")
            input_tensors = [self.encoder_inputs, self.encoder_length]
            if self.mode == "train":
                # decoder input only needed in training phase
                # decoder seq [batch, max_time_step], lengths [batch, ]
                self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
                self.decoder_length = tf.placeholder(tf.int32, [None, ], name="decoder_length")
                input_tensors += [self.decoder_inputs, self.decoder_length]

                # To add _EOS, so modified inputs and lengths
                decoder_GO = tf.ones(shape=(self.batch, 1), dtype=tf.int32) * config._GO
                decoder_EOS = tf.ones(shape=(self.batch, 1), dtype=tf.int32) * config._EOS

                # input_train [batch, max_time_step+1], target_train [batch, max_time_step+1]
                self.decoder_inputs_train = tf.concat([decoder_GO, self.decoder_inputs], axis=1)
                self.decoder_length_train = self.decoder_length + 1
                self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_EOS], axis=1)

        return input_tensors

    def build_encoder(self, keep_prob):
        print("build encoder")
        with tf.variable_scope("encoder"):
            initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            encoder_embeddings = tf.get_variable(name="encoder_embeddings",
                                                 shape=[config.encoder_vocab_size, config.embedding_size],
                                                 initializer=initializer, dtype=tf.float32)
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.encoder_inputs)
            # tf.summary.histogram('encoder_embedding', encoder_embeddings)
            encoder_fw_cells, encoder_bw_cells = self.build_encoder_cell(config.hidden_size//2, config.num_layers, keep_prob)
            (self.encoder_outputs, self.encoder_fs) = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cells,
                                                                            encoder_bw_cells,
                                                                            encoder_inputs_embedded,
                                                                            sequence_length=self.encoder_length,
                                                                            dtype=tf.float32, time_major=False)


    def build_decoder(self, keep_prob):
        print("build decoder")
        with tf.variable_scope("decoder"):
            decoder_cell, decoder_init_state = self.build_decoder_cell(config.hidden_size, config.num_layers,keep_prob)
            
            initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            self.decoder_embedding = tf.get_variable(name="decoder_embedding",
                                                     shape=[config.decoder_vocab_size, config.embedding_size],
                                                     initializer=initializer, dtype=tf.float32)
            # tf.summary.histogram("decoder_embed", self.decoder_embedding)

            output_layer = Dense(config.decoder_vocab_size)
            # input_layer = Dense(config.hidden_size, dtype=tf.float32)

            if self.mode == "train":
                # decoder_inputs_embedded: [n, max_time_step+1, embedding_size]
                decoder_inputs_embedded = tf.nn.embedding_lookup(self.decoder_embedding, self.decoder_inputs_train)
                #self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                train_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                        sequence_length=self.decoder_length_train,
                                        time_major=False, name="traing_helper")
                train_decoder = BasicDecoder(cell=decoder_cell, helper=train_helper,
                                             initial_state=decoder_init_state, output_layer=output_layer)

                self.max_decoder_length = tf.reduce_max(self.decoder_length_train)

                self.decoder_outputs_train, self.decoder_last_state_train, self.decoder_outputs_length_train = dynamic_decode(
                    decoder=train_decoder, output_time_major=False, impute_finished=True, maximum_iterations=self.max_decoder_length)

                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                self.pred = tf.argmax(self.decoder_logits_train, axis=-1)

                # mask for true and padded time steps. shape=[batch, max_length+1]
                self.masks = tf.sequence_mask(lengths=self.decoder_length_train,
                                         maxlen=self.max_decoder_length, dtype=tf.float32, name="masks")

                # decoder_logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # decoder_targets_train: [batch_size, max_time_steps + 1]
                self.loss = sequence_loss(logits=self.decoder_logits_train, targets=self.decoder_targets_train,
                                          weights=self.masks, average_across_timesteps=True,
                                          average_across_batch=True)

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


    def make_rnn_cell(self, num_units, keep_prob):
        initializer = tf.orthogonal_initializer()
        cell = LSTMCell(num_units, initializer=initializer)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell

    def build_encoder_cell(self, num_units, num_layers, keep_prob):
        encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(num_units, keep_prob) for _ in range(num_layers)])
        encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(num_units, keep_prob) for _ in range(num_layers)])
        return encoder_cell_fw, encoder_cell_bw

    def build_decoder_cell(self, num_units, num_layers, keep_prob):
        encoder_outputs = tf.concat(self.encoder_outputs, axis=-1)

        encoder_final_state = []
        encoder_fw_fs, encoder_bw_fs = self.encoder_fs
        for i in range(num_layers):
            final_state_c = tf.concat((encoder_fw_fs[i].c, encoder_bw_fs[i].c), axis=1)
            final_state_h = tf.concat((encoder_fw_fs[i].h, encoder_bw_fs[i].h), axis=1)
            encoder_final_state.append(LSTMStateTuple(c=final_state_c, h=final_state_h))
        encoder_fs = tuple(encoder_final_state)

        # build decoder cell
        decoder_cells = [self.make_rnn_cell(num_units, keep_prob) for _ in range(num_layers)]
        attention_cell = decoder_cells.pop()

        # use Bahdanua attention to all cell layers.
        self.attention_machenism = attention_wrapper.BahdanauAttention(num_units=num_units,
                                                                       memory=encoder_outputs, normalize=False,
                                                                       memory_sequence_length=self.encoder_length)

        attention_cell = attention_wrapper.AttentionWrapper(attention_cell, self.attention_machenism,
                                            attention_layer_size=None,
                                            initial_cell_state=None,
                                            output_attention=False,
                                            alignment_history=False,)
        decoder_cells.append(attention_cell)
        decoder_cells = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)
        batch = self.batch
        decoder_init_state = tuple(zs.clone(cell_state=es) if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState)
                                else es for zs, es in zip(decoder_cells.zero_state(batch, dtype=tf.float32), encoder_fs))
        
        # why the last layers' zero state different with
        # init_state = [state for state in encoder_fs]
        return decoder_cells, decoder_init_state


    def build_graph(self, keep_prob):
        self.build_encoder(keep_prob)
        self.build_decoder(keep_prob)
