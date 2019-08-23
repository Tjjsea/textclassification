#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from get_batch import getbatch
from models import textCNN,bilstm,bilstm_attention
import os
import json
import math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #屏蔽warning信息

batch_size=128
sequence_length=140
num_classes=1
embedding_dim=200

w2n=json.load(open('data/w2n.json'))
vocab_size=len(w2n)
hiddensizes=[512,256]
filter_sizes=[2,3,4,5]

tf.app.flags.DEFINE_integer("sequence_length",sequence_length,"sequence length")
tf.app.flags.DEFINE_integer("vocab_size",vocab_size,"vocab size")
tf.app.flags.DEFINE_integer("embedding_dim",embedding_dim,"word level embedding size")
tf.app.flags.DEFINE_integer("num_classes",num_classes,"num of POStags")
tf.app.flags.DEFINE_integer("featuremaps",2,"featuremaps")
tf.app.flags.DEFINE_integer("epochs",5,"num of epochs")
tf.app.flags.DEFINE_integer("batch_size",batch_size,"batch size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",40,"save model checkpoint every this iteration")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learing rate")
tf.app.flags.DEFINE_float("l2RegLambda",0,"l2RegLambda")
tf.app.flags.DEFINE_string("model_dir","model/textcnn/","path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","textcnn.ckpt","file name used for model checkpoints")
FLAGS=tf.app.flags.FLAGS

w2v=np.loadtxt('data/embedding.txt')

with tf.Session() as sess:
    model=bilstm_attention(flags=FLAGS,hiddensizes=hiddensizes,w2v=None)
    #model=textCNN(flags=FLAGS,filter_sizes=filter_sizes,w2v=None)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        for batch in getbatch(mode="test"):
            acc=model.eval(sess, batch)
            print("----- Step %d -- Acc %.2f" % (current_step, acc))
    else:
        print('error, no model exists')