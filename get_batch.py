#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import jieba
import random

batch_size=128
max_sequence_length=140
num_classes=1

class Batch():
    def __init__(self):
        self.input_x=[]
        self.input_y=[]
        self.sequence_length=[] #the actual lengths for each of the sequences of input_x in the batch
        self.position=[]
    
def predata():
    vocabs={"UNK":0}
    count=1
    length=[]
    with open('datas/cutclean_label_corpus10000.txt',encoding='gb18030') as fin:
        texts=fin.readlines()
        for line in texts:
            line=line.strip()
            line=line[1:]
            line=line.split(' ')
            length.append(len(line))
            for word in line:
                if word==' ' or word=='':
                    continue
                if word not in vocabs:
                    vocabs[word]=count
                    count+=1
        random.shuffle(texts)
        train=texts[:8000]
        dev=texts[8000:]
        json.dump(vocabs,open('datas/w2n.json','w'))
        fout=open('datas/train.txt','w',encoding='utf-8')
        fout.writelines(train)
        fout=open('datas/dev.txt','w',encoding='utf-8')
        fout.writelines(dev)
        return length
    
def getbatch(mode,batch_size=batch_size,max_sequence_length=max_sequence_length):
    '''
    get batch from ch,en
    '''
    if mode=="train":
        fin=open('data/train.txt',encoding='utf-8')
        datas=fin.readlines()
    else:
        fin=open('data/dev.txt',encoding='utf-8')
        datas=fin.readlines()
        
    batches=[]
    w2n=json.load(open('data/w2n.json',encoding='utf-8'))
    fin=open('data/train.txt',encoding='utf-8')
    random.shuffle(datas)
    for i in range(0,len(datas),batch_size):
        ed=min(len(datas),i+batch_size)
        part=datas[i:ed]
        batch=Batch()
        for line in part:
            line=line.strip()
            line=line.split(' ')
            label=int(line[0])
            inputx=[]
            for word in line:
                if word == ' ' or word == '':
                    continue
                inputx.append(w2n.get(word,0))
                if len(inputx)>=max_sequence_length:
                    batch.sequence_length.append(max_sequence_length)
                    break
            if len(inputx)<max_sequence_length:
                batch.sequence_length.append(len(inputx))
                inputx.extend([0]*(max_sequence_length-len(inputx)))
            if num_classes==1:
                inputy=[0]*2
            else:
                inputy=[0]*num_classes
            inputy[label]=1
            batch.input_x.append(inputx)
            batch.input_y.append(inputy)
        yield batch
        #batches.append(batch)
    #return batches


if __name__=="__main__":
    for nextbatch in getbatch():
        x=nextbatch.input_x
        y=nextbatch.input_y
        sl=nextbatch.sequence_length
        print(x[0])
        print(y[1])
        print(sl)