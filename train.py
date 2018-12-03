#coding=utf-8
from __future__ import print_function
from __future__ import division

import sys, os
import datetime as dt
from dataloader import *
import tensorflow as tf
import config
from model import Seq2Seq

# parameters of train data
tf.app.flags.DEFINE_string("train_data", './data/human/human_train.pickle', "position of train data")
tf.app.flags.DEFINE_string("test_data", './data/human/human_test.pickle', "path of test data")

# parameters of network
# this part is setted up in config.py

# parameters of training details
tf.app.flags.DEFINE_integer("epoch", 1, "number of training epoch")
tf.app.flags.DEFINE_integer("batch", 16, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("decay", 8849, "learning rate decay at this number of iteration")
tf.app.flags.DEFINE_float("drop_out", 0.2, "drop out probability")
tf.app.flags.DEFINE_float("gamma", 0.001, "gamma to balance two loss")

tf.app.flags.DEFINE_integer("display", 20, "show the training detail at this number of iteration")
tf.app.flags.DEFINE_integer("save", 5000, "save model at this number of iteration")
tf.app.flags.DEFINE_integer("evaluate", 6880, "evaluate model at this intervals")

tf.app.flags.DEFINE_boolean("finetune", True, "whether need fine tune")
tf.app.flags.DEFINE_string("model_dir", "./models/", "path to save your model" )
tf.app.flags.DEFINE_string("model_name", "model.ckpt-35000", "name of your ckpt model file ")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def create_model(sess, model='train'):
    model = Seq2Seq(mode="train", batch=FLAGS.batch,
                    drop_prob=FLAGS.drop_out, gamma=FLAGS.gamma)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if FLAGS.finetune and FLAGS.model_name:
        print("loading model parameters from given model path and name")
        saver.restore(sess, FLAGS.model_dir + FLAGS.model_name)
    elif FLAGS.finetune and ckpt:
        print("loading model parameters from last saved model in %s" % FLAGS.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.mkdir(FLAGS.model_dir)
        print("create new model parameters")
        sess.run(tf.global_variables_initializer())
        
    model.table.init.run()
    return model, saver


def train():
    train_loader = DataLoader(FLAGS.train_data, FLAGS.batch, FLAGS.epoch)

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    with tf.Session(config=conf) as sess:
        
        model, saver = create_model(sess)

        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        global_step, local_step = 0, 0
        loss, lr = 0.0, FLAGS.learning_rate
        for data in train_loader.data_iter():
            spectra, spectra_len, decoder_seqs,  decoder_length = data
            feed_dict = {model.encoder_inputs: spectra, model.encoder_length: spectra_len,
                         model.decoder_inputs: decoder_seqs, model.decoder_length: decoder_length,
                         model.lr: lr}
            #logits, targets, l = sess.run([self.decoder_outputs_train.rnn_output, self.decoder_targets_train, self.decoder_logits_train], feed_dict=feed_dict)
    
            step_loss, _, summary, loss2 = sess.run([model.loss, model.train_op, model.summary, model.loss2], feed_dict=feed_dict)
            loss += step_loss
            global_step += 1
            local_step += 1
            if global_step % FLAGS.display == 0:
                print("{}, Global step={}, lr={}, loss={}, loss2={}".format(
                    dt.datetime.now().strftime("%m.%d-%H:%M:%S"), global_step, lr,  loss/local_step, loss2))
                sys.stdout.flush()
                loss = 0.0
                local_step = 0
                log_writer.add_summary(summary, global_step)

    
            if global_step % FLAGS.save == 0:
                print('saving model checkpoint to path')
                saver.save(sess, FLAGS.model_dir+FLAGS.model_name, global_step=global_step)

            #if global_step % FLAGS.evaluate == 0:
                #test_loader = DataLoader(FLAGS.source_test_data,
                #                         FLAGS.target_test_data,
                #                         1, FLAGS.batch)
                
    
            if global_step % FLAGS.decay == 0:
                lr *= 0.8
	    

if __name__ == "__main__":
    train()
