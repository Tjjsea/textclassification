#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import jieba
import random
import numpy as np

batch_size=128
max_sequence_length=128
num_classes=1

class Batch():
    def __init__(self):
        self.input_x=[]
        self.input_y=[]
        self.sequence_length=[] #the actual lengths for each of the sequences of input_x in the batch
        self.position=[]
        self.transformer_position=[]
    
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
            pos=[]
            count=0
            for word in line:
                if word == ' ' or word == '':
                    continue
                inputx.append(w2n.get(word,0))
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
            if num_classes==1:
                inputy=[0]*2
            else:
                inputy=[0]*num_classes
            inputy[label]=1
            batch.input_x.append(inputx)
            batch.input_y.append(inputy)
            batch.transformer_position.append(pos)
        yield batch
        #batches.append(batch)
    #return batches

def han_batch(mode,batch_size=64):
    if mode=="train":
        data=json.load(open('ch_en/numtrain.json'))
    else:
        data=json.load(open('ch_en/numdev.json'))
    
    random.shuffle(data)
    for i in range(0,len(data),batch_size):
        ed=min(i+batch_size,len(data))
        part=data[i:ed]
        batch=Batch()
        for line in part:
            label=int(line[0])
            document=line[1:]
            batch.input_x.append(document)
            if num_classes==1:
                input_y=[0]*2
            else:
                input_y=[0]*num_classes
            input_y[label]=1
            batch.input_y.append(input_y)
        yield batch

if __name__=="__main__":
    for nextbatch in han_batch("train"):
        x=np.array(nextbatch.input_x,dtype=np.int32)
        y=np.array(nextbatch.input_y,dtype=np.int32)
        print(x.shape)
        print(y.shape)
