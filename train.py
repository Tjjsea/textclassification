#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from get_batch import getbatch
from models import bilstm,bilstm_attention
from tqdm import tqdm
import os
import json
import math

batch_size=64
sequence_length=128
num_classes=6

w2n=json.load(open('datas/w2n.json'))
vocab_size=len(w2n)
hiddensizes=[256]

tf.app.flags.DEFINE_integer("sequence_length",sequence_length,"sequence length")
tf.app.flags.DEFINE_integer("vocab_size",vocab_size,"vocab size")
tf.app.flags.DEFINE_integer("embedding_dim",100,"word level embedding size")
tf.app.flags.DEFINE_integer("char_biunits",256,"num of hidden layer for char embedding bilstm")
tf.app.flags.DEFINE_integer("num_classes",num_classes,"num of POStags")
tf.app.flags.DEFINE_integer("epochs",2,"num of epochs")
tf.app.flags.DEFINE_integer("batch_size",batch_size,"batch size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",10,"save model checkpoint every this iteration")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learing rate")
tf.app.flags.DEFINE_string("model_dir","model/","path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","bilstm-attention.ckpt","file name used for model checkpoints")
FLAGS=tf.app.flags.FLAGS

with tf.Session() as sess:
    model=bilstm(flags=FLAGS,hiddensizes=hiddensizes,w2v=None)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    
    current_step = 0
    #summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.epochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.epochs))

        batches=getbatch()
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        for batch in batches:
            loss,acc=model.train(sess, batch)
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("----- Step %d -- Loss %.2f -- Acc %.2f" % (current_step, loss, acc))
                #summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)