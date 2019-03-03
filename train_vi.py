#coding=utf-8
from __future__ import print_function
from __future__ import division

import sys, os
import datetime as dt
from gpuloader import *
import tensorflow as tf
import config
import pickle
from model_topbah import Seq2Seq
from nltk.translate.bleu_score import corpus_bleu


# parameters of train data
tf.app.flags.DEFINE_string("source_train_data", './vien/train.en.pkl', "position of source train data")
tf.app.flags.DEFINE_string("target_train_data", './vien/train.vi.pkl', "position of target train data")
tf.app.flags.DEFINE_string("source_val_data", './vien/tst2012.en.pkl', "path of source val")
tf.app.flags.DEFINE_string("target_val_data", './vien/tst2012.vi.pkl', "path of target val")
tf.app.flags.DEFINE_string("target_vocab", './vien/vocab_id.vi.pkl', "path of target val")

# parameters of training details
tf.app.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.app.flags.DEFINE_integer("batch", 128, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("decay", 2000, "learning rate decay at this number of iteration")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep prob for dropout layer")

tf.app.flags.DEFINE_integer("display", 20, "show the training detail at this number of iteration")
tf.app.flags.DEFINE_integer("save", 1000, "save model at this number of iteration")
tf.app.flags.DEFINE_integer("evaluate", 1000, "evaluate model at this intervals")

tf.app.flags.DEFINE_boolean("finetune", False, "whether need fine tune")
tf.app.flags.DEFINE_string("model_dir", "./models/", "path to save your model" )
tf.app.flags.DEFINE_string("model_name", "topbah.ckpt", "name of your ckpt model file ")


FLAGS = tf.app.flags.FLAGS
gpus = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
GPUS = [int(gpu) for gpu in gpus.split(',')]


def vocab():
    with open(FLAGS.target_vocab, 'rb') as f:
        target_vocab = pickle.load(f)
    id_2words = target_vocab["id2word"]
    return id_2words


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_var in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_var:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_var[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)
    return average_grads


def train():
    train_loader = DataLoader(FLAGS.source_train_data, FLAGS.target_train_data, FLAGS.epoch, FLAGS.batch, len(GPUS))
    val_loader = DataLoader(FLAGS.source_val_data, FLAGS.target_val_data, 1, FLAGS.batch, len(GPUS))
    id_2words = vocab()

    graph = tf.Graph()
    with graph.as_default(), tf.device("/cpu:10"):
        tower_grads, tower_loss  = [], []
        tower_inputs, tower_preds = [], []
        model = Seq2Seq(mode="train", batch=FLAGS.batch)
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        opt = tf.train.RMSPropOptimizer(learning_rate)
        #opt = tf.train.GradientDescentOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for g in range(len(GPUS)):
                with tf.device("/gpu:%d"%g):
                    with tf.name_scope("tower_%d"%g):
                        inputs = model.init_placeholder()
                        model.build_graph(keep_prob)
                        pred, loss = model.pred, model.loss
                        grad = opt.compute_gradients(loss)
                        tf.get_variable_scope().reuse_variables()

                        tower_inputs.append(inputs)
                        tower_grads.append(grad)
                        tower_loss.append(loss)
                        tower_preds.append(pred)

        mean_loss = tf.stack(axis=0, values=tower_loss)
        ave_loss = tf.reduce_mean(mean_loss, 0)
        mean_grads = average_gradients(tower_grads)
        clipped_grads, _ = tf.clip_by_global_norm(mean_grads, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(mean_grads)

        saver = tf.train.Saver(max_to_keep=3)
        init = tf.global_variables_initializer()

    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement=True
    conf.log_device_placement=False
    sess = tf.Session(graph=graph, config=conf)
    sess.run(init)

    print("start training")
    global_step, local_step = 0, 0
    train_loss, lr = 0.0, FLAGS.learning_rate
    for data in train_loader.data_iter():
        feed_dict = {learning_rate: lr, keep_prob: FLAGS.keep_prob}
        for i in range(len(GPUS)):
            feed_dict[tower_inputs[i][0]] = data[i][0]
            feed_dict[tower_inputs[i][1]] = data[i][1]
            feed_dict[tower_inputs[i][2]] = data[i][2]
            feed_dict[tower_inputs[i][3]] = data[i][3]

        step_loss, _ = sess.run([ave_loss, train_op], feed_dict=feed_dict)
        train_loss += step_loss
        global_step += 1
        local_step += 1
        if global_step % FLAGS.display == 0:
            print("{}, Global step={}, lr={}, loss={}".format(
                dt.datetime.now().strftime("%m.%d-%H:%M:%S"), global_step, lr,  train_loss/local_step))
            sys.stdout.flush()
            train_loss = 0.0
            local_step = 0
        #     log_writer.add_summary(summary, global_step)

        if global_step % FLAGS.save == 0:
            print('saving model checkpoint to path')
            saver.save(sess, FLAGS.model_dir+FLAGS.model_name, global_step=global_step)

        if global_step % FLAGS.evaluate == 0:
            bleu, iter_num, val_loss = 0.0, 0, 0.0
            for val_data in val_loader.data_iter():
                feed_dict = {keep_prob: 1.0}
                for i in range(len(GPUS)):
                    feed_dict[tower_inputs[i][0]] = val_data[i][0]
                    feed_dict[tower_inputs[i][1]] = val_data[i][1]
                    feed_dict[tower_inputs[i][2]] = val_data[i][2]
                    feed_dict[tower_inputs[i][3]] = val_data[i][3]

                step_loss = sess.run(ave_loss, feed_dict=feed_dict)
                val_loss += step_loss

                preds = sess.run(tower_preds, feed_dict=feed_dict)  # [num_gpu * [batch, max_len]]
                refs, hypes = [], []  # refs should be [list(list(str))]
                for i in range(len(GPUS)):
                    batch_true, batch_len = val_data[i][2], val_data[i][3]
                    batch_pred = preds[i]       # [batch, max_len]
                    for j in range(FLAGS.batch):
                        refs.append([[id_2words[w] for w in batch_true[j] if w != config._EOS]])
                        hypes.append([id_2words[w] for w in batch_pred[j][:batch_len[j]]])
                bleu += corpus_bleu(refs, hypes)

                iter_num += 1
            print("validation loss={}, bleu={}".format(val_loss / iter_num, bleu / iter_num))

        if global_step % FLAGS.decay == 0:
            if global_step > 8000:
                lr *= 0.8


if __name__ == "__main__":
    train()
