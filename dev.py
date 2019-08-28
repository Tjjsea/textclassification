#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from models import textCNN,bilstm,bilstm_attention,transformer,TextCNN
import os
import json
import math
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #屏蔽warning信息

batch_size=512
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
        #random.shuffle(datas) #这一步不应该放在验证中
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
            pos=[]
            count=0
            for word in text:
                if word=='' or word==' ':
                    continue
                num=w2n.get(word,0)
                inputx.append(num)
                temp=[0]*max_sequence_length
                temp[count]=1
                pos.append(temp)
                count+=1
                if len(inputx)>=max_sequence_length:
                    batch.sequence_length.append(max_sequence_length)
                    break
            if len(inputx)<max_sequence_length:
                batch.sequence_length.append(len(inputx))
                inputx.extend([0]*(max_sequence_length-len(inputx)))
                for i in range(count,max_sequence_length):
                    temp=[0]*max_sequence_length
                    temp[i]=1
                    pos.append(temp)
            batch.input_x.append(inputx)
            if num_classes==1:
                inputy=[label]
            else:
                inputy=[0]*num_classes
                inputy[label]=1
            batch.input_y.append(inputy)
            batch.transformer_position.append(pos)
        yield batch

    
def get_precision(pres,labels):
    '''
    pres:模型预测结果，1维
    labels:实际标签，1维
    '''
    assert len(pres)==len(labels)

    allpre=[[0,0],[0,0]]
    #labels=np.array(labels)
    #labels=np.argmax(labels,-1)
    #labels=[i[0] for i in labels]

    for i in range(len(pres)):
        if pres[i]==0:
            allpre[0][0]+=1
            if labels[i]==0:
                allpre[0][1]+=1
        if pres[i]==1:
            allpre[1][0]+=1
            if labels[i]==1:
                allpre[1][1]+=1
    
    l1=sum(labels)
    l0=len(labels)-l1
    pre0=float(allpre[0][1]/allpre[0][0])
    rec0=float(allpre[0][1]/l0)
    f10=float(2*pre0*rec0/(pre0+rec0))
    pre1=float(allpre[1][1]/allpre[1][0])
    rec1=float(allpre[1][1]/l1)
    f11=float(2*pre1*rec1/(pre1+rec1))
    accuracy=float((allpre[0][1]+allpre[1][1])/len(labels))

    print('中文 p@1:%.3f' % pre0)
    print('中文 R@2:%.3f' % rec0)
    print('中文 F1:%.3f' % f10)
    print('英文 p@1:%.3f' % pre1)
    print('英文 R@2:%.3f' % rec1)
    print('英文 F1:%.3f' % f11)
    print('accuracy :%.3f' % accuracy)

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
tf.app.flags.DEFINE_float("learning_rate",1e-4,"learing rate")
tf.app.flags.DEFINE_float("l2RegLambda",0.0,"l2RegLambda")
tf.app.flags.DEFINE_string("model_dir","model/transformer/","path to save model checkpoints")
tf.app.flags.DEFINE_string("model_name","transformer.ckpt","file name used for model checkpoints")
FLAGS=tf.app.flags.FLAGS

with tf.Session() as sess:
    #model=bilstm_attention(flags=FLAGS,hiddensizes=hiddensizes,w2v=None)
    #model=TextCNN(FLAGS,filter_sizes)
    #model=textCNN(flags=FLAGS,filter_sizes=filter_sizes,w2v=None)
    model=transformer(flags=FLAGS,w2v=None)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        #init = tf.global_variables_initializer()
        #sess.run(init)
        allpre=[]
        labels=[]
        for batch in get_batch(mode="test"):
            pres,acc=model.demo(sess, batch)
            #print("accuracy :%.3f" % acc)
            allpre.extend(pres)
            label=[i[0] for i in batch.input_y]
            labels.extend(label)
        get_precision(allpre,labels)
    else:
        print('error, no model exists')