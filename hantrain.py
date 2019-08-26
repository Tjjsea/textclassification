#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from get_batch import getbatch,han_batch
from models import textCNN,bilstm,bilstm_attention,transformer,HAN
import os
import json
import math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #屏蔽warning信息

batch_size=16
sequence_length=128
num_classes=1
embedding_dim=100
Max_Sentence_Length=144
Max_Document_Length=96

w2n=json.load(open('data/w2n.json'))
vocab_size=len(w2n)
hiddensizes=[256,128]
whiddensizes=[256]
shiddensizes=[256]

tf.app.flags.DEFINE_integer("sequence_length",sequence_length,"sequence length")
tf.app.flags.DEFINE_integer("vocab_size",vocab_size,"vocab size")
tf.app.flags.DEFINE_integer("embedding_dim",embedding_dim,"word level embedding size")
tf.app.flags.DEFINE_integer("num_classes",num_classes,"num of POStags")
tf.app.flags.DEFINE_integer("epochs",2,"num of epochs")
tf.app.flags.DEFINE_integer("batch_size",batch_size,"batch size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",10,"save model checkpoint every this iteration")
tf.app.flags.DEFINE_integer("sentence_length",Max_Sentence_Length,"sentence length")
tf.app.flags.DEFINE_integer("document_length",Max_Document_Length,"document length")
tf.app.flags.DEFINE_integer("wt_hidunits",128,"word attention hidunits")
tf.app.flags.DEFINE_integer("st_hidunits",128,"sentence attention hidunits")
tf.app.flags.DEFINE_float("learning_rate",0.0001,"learing rate")
tf.app.flags.DEFINE_float("l2RegLambda",0.0,"l2RegLambda")
tf.app.flags.DEFINE_string("model_dir","model/HAN/","path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","HAN.ckpt","file name used for model checkpoints")
FLAGS=tf.app.flags.FLAGS

w2v=np.loadtxt('data/embedding.txt')

with tf.Session() as sess:
    #model=bilstm_attention(flags=FLAGS,hiddensizes=hiddensizes,w2v=None)
    #model=textCNN(flags=FLAGS,filter_sizes=filter_sizes,w2v=None)
    #model=transformer(flags=FLAGS,w2v=None)
    model=HAN(flags=FLAGS,w2v=None,whiddensizes=whiddensizes,shiddensizes=shiddensizes)
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
        for batch in han_batch(mode="train"):
            loss,acc=model.train(sess, batch)
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("----- Step %d -- Loss %.2f -- Acc %.2f" % (current_step, loss, acc))
                #summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)