#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
将语料转化成ELMo embedding
@author: taojunjie3
'''

import tensorflow as tf
import os
import sys
import json
import math
import random
import numpy as np
from elmo_utils import Config,Batch
import argparse
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher

o_path=os.getcwd()
sys.path.append("../")
from models import ELMo

ap=argparse.ArgumentParser()
ap.add_argument("--GPU",required=False,type=str,help="which GPU to use")
args=ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU if args.GPU else '3' #use GPU with ID=0
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM
tfconfig.gpu_options.allow_growth = True #allocate dynamically
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #屏蔽warning信息

config=Config()

if __name__=='__main__':
    datas=json.load(open('../data/test2.json',encoding='utf-8'))
    ndatas=[line.split()[1:] for line in datas[:10]]

    batcher=TokenBatcher(config.vocab_file)             #生成词表示的batch类

    inputData=tf.placeholder('int32',shape=(None,None))

    abilm=BidirectionalLanguageModel(config.option_file,config.weight_file,use_character_inputs=False,embedding_weight_file=config.tokenEmbeddingFile)
    inputEmbeddingsOp=abilm(inputData)

    elmoInput=weight_layers('input',inputEmbeddingsOp,l2_coef=0.0)

    sess=tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        inputids=batcher.batch_sentences(ndatas)    #生成batch数据
        inputvec=sess.run(elmoInput['weighted_op'],feed_dict={inputData:inputids})
        print(inputvec)        
    sess.close()