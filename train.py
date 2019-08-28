#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from models import textCNN,bilstm,bilstm_attention,transformer,TextCNN
import os
import json
import math
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #屏蔽warning信息

batch_size=128
max_sequence_length=128
num_classes=1
embedding_dim=200

w2n=json.load(open('data/w2n.json'))
vocab_size=len(w2n)
hiddensizes=[256,128]
filter_sizes=[2,3,4,5]

class Batch():
    def __init__(self):
        self.input_x=[]
        self.input_y=[]
        self.sequence_length=[] #the actual lengths for each of the sequences of input_x in the batch
        self.position=[]
        self.transformer_position=[]

def get_batch(mode,num_classes=num_classes,batch_size=batch_size):
    if mode=='train':
        fin=open('data/train.txt')
        datas=fin.readlines()
    else:
        fin=open('data/dev.txt')
        datas=fin.readlines()
    
    for i in range(0,len(datas),batch_size):
        end=min(i+batch_size,len(datas))
        random.shuffle(datas) #这一步不应该放在验证中
        part=datas[i:end]
        batch=Batch()
        for line in part:
            line=line.strip()
            if line=='' or line==' ':
                continue
            line=line.split(' ')
            label=int(line[0])
            text=line[1:]
            inputx=[]
            for word in text:
                if word=='' or word==' ':
                    continue
                num=w2n.get(word,0)
                inputx.append(num)
                if len(inputx)>=max_sequence_length:
                    batch.sequence_length.append(max_sequence_length)
                    break
            if len(inputx)<max_sequence_length:
                batch.sequence_length.append(len(inputx))
                inputx.extend([0]*(max_sequence_length-len(inputx)))
            batch.input_x.append(inputx)
            if num_classes==1:
                inputy=[label]
            else:
                inputy=[0]*num_classes
                inputy[label]=1
            batch.input_y.append(inputy)
        yield batch

tf.app.flags.DEFINE_integer("sequence_length",max_sequence_length,"sequence length")
tf.app.flags.DEFINE_integer("vocab_size",vocab_size,"vocab size")
tf.app.flags.DEFINE_integer("embedding_dim",embedding_dim,"word level embedding size")
tf.app.flags.DEFINE_integer("num_classes",num_classes,"num of POStags")
tf.app.flags.DEFINE_integer("featuremaps",2,"featuremaps")
tf.app.flags.DEFINE_integer("epochs",20,"num of epochs")
tf.app.flags.DEFINE_integer("batch_size",batch_size,"batch size")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",50,"save model checkpoint every this iteration")
tf.app.flags.DEFINE_integer("numBlocks",2,"numBlocks")
tf.app.flags.DEFINE_integer("filters",128,"filters")
tf.app.flags.DEFINE_integer("numHeads",8,"num of heads of attention")
tf.app.flags.DEFINE_float("keepProp",0.9,"dropout in multihead attention")
tf.app.flags.DEFINE_float("epsilon",1e-8,"epsilon")
tf.app.flags.DEFINE_float("learning_rate",1e-3,"learing rate")
tf.app.flags.DEFINE_float("l2RegLambda",0.0,"l2RegLambda")
tf.app.flags.DEFINE_string("model_dir","model/textcnn/","path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","textcnn.ckpt","file name used for model checkpoints")
FLAGS=tf.app.flags.FLAGS

#w2v=np.loadtxt('data/embedding.txt')

with tf.Session() as sess:
    model=textCNN(flags=FLAGS,filter_sizes=filter_sizes,w2v=None)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    
    current_step = 0
    best_acc=[]
    #summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.epochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.epochs))
        for batch in get_batch(mode="train"):
            loss,acc=model.train(sess, batch,current_step)
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("----- Step %d -- Loss %.4f -- Acc %.4f" % (current_step, loss, acc))
                #summary_writer.add_summary(summary, current_step)
                if len(best_acc)<5:
                    best_acc.append(acc)
                    best_acc.sort()
                else:
                    if acc<best_acc[0]:
                        continue
                    else:
                        best_acc.pop(0)
                        best_acc.append(acc)
                        best_acc.sort()
                        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                        model.saver.save(sess, checkpoint_path, global_step=current_step)

                        