#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import jieba
import numpy as np
import random
import time
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, dump_token_embeddings, Batcher

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
    def __init__(self):
        self.embedding_dim=embedding_dim
        self.num_classes=num_classes
        self.sequence_length=Max_Sequence_Length
        self.featuremaps=50
        self.filter_sizes=[2,3,4]
        self.hiddensizes=[256,128]
        self.l2RegLambda=0.5
        self.learning_rate=1e-4
        self.model_name='sentiment.ckpt'
        self.model_dir='../model/ELMO/'
        self.vocab_file="model/vocabs.txt"                         #基于训练数据的词表，一行一个词
        self.option_file="model/elmo_small_options.json"
        self.weight_file="model/elmo_small_weights.hdf5"
        self.tokenEmbeddingFile='model/elmo_token_embedding.hdf5'  #词表中的词的向量表示
        
        self.numBlocks=2
        self.filters=128
        self.numHeads=8
        self.keepProp=0.9
        self.epsilon=1e-8
    
    def getElmoEmbedding(self):
        dump_token_embeddings(self.vocab_file,self.option_file,self.weight_file,self.tokenEmbeddingFile)

def getVocabs():
    datas=json.load(open('../data/edatrain.json',encoding='utf-8'))

    vocabs={'<S>','</S>','<UNK>'}
    for line in datas:
        line=line.split(' ')[1:]
        for word in line:
            vocabs.add(word)
    vocabs=list(vocabs)
    fout=open('model/vocabs.txt','w',encoding='utf-8')
    fout.write('\n'.join(vocabs))
    fout.close()

if __name__=='__main__':
    #getVocabs()
    pass