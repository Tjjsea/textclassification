#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import jieba
import numpy as np
import random
import time

batch_size=128
num_classes=1
embedding_dim=256
Max_Sequence_Length=128

Files={"train":"data/edatrain.json",
       "test":"data/test2.json"}

class Batch():
    def __init__(self):
        self.input_x=[]
        self.input_y=[]
        self.sequence_length=[] #the actual lengths for each of the sequences of input_x in the batch
        self.position=[]
        #self.transformer_position=[]
    
class Config():
    def __init__(self,mode):
        self.embedding_dim=embedding_dim
        self.num_classes=num_classes
        self.sequence_length=Max_Sequence_Length
        self.featuremaps=50
        self.filter_sizes=[2,3,4]
        self.hiddensizes=[256,128]
        self.l2RegLambda=0.5
        self.learning_rate=1e-4
        w2n=json.load(open('data/edaw2n.json',encoding='utf-8'))
        #w2n=json.load(open('model/textrnn/m4/edaw2n.json',encoding='utf-8'))
        self.vocab_size=len(w2n)
        self.model_name='sentiment.ckpt'
        self.model_dir='../model/ELMO/'
        self.option_file="model/elmo_small_options.json"
        self.weight_file="model/elmo_small_weights.hdf5"
        self.tokenEmbeddingFile='model/vocabs.txt'
        
        self.numBlocks=2
        self.filters=128
        self.numHeads=8
        self.keepProp=0.9
        self.epsilon=1e-8

def get_batch(mode,batch_size=batch_size):
    data=json.load(open(Files[mode]))
    random.shuffle(data)
    w2n=json.load(open('data/edaw2n.json',encoding='utf-8'))

    for i in range(0,len(data),batch_size):
        ed=min(i+batch_size,len(data))
        part=data[i:ed]
        batch=Batch()
        for sen in part:
            label=int(sen[0])
            words=sen.split(' ')[1:]
            if len(words)>=Max_Sequence_Length:
                words=words[:Max_Sequence_Length]
                batch.sequence_length.append(Max_Sequence_Length)
            nums=[]
            for word in words:
                if word in w2n:
                    nums.append(w2n[word])
                else:
                    nums.append(0)
            if len(nums)<Max_Sequence_Length:
                batch.sequence_length.append(len(nums))
                nums.extend([1]*(Max_Sequence_Length-len(nums)))
            batch.input_x.append(nums)
            if num_classes==1:
                inputy=[label]
            else:
                inputy=[0]*num_classes
                inputy[label]=1
            batch.input_y.append(inputy)               #y是one-hot形式的
            pos=[[0]*Max_Sequence_Length for i in range(Max_Sequence_Length)]
            for j in range(Max_Sequence_Length):
                pos[j][j]=1
            batch.position.append(pos)
        yield batch
    
if __name__=='__main__':
    pass