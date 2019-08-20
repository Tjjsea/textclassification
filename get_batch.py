#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import jieba
import random

batch_size=64
sequence_length=128
num_classes=2

class Batch():
    def __init__(self):
        self.input_x=[]
        self.input_y=[]
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
    
def getbatch():
    '''
    batch_size:64, sequence_length:128
    '''
    batches=[]
    w2n=json.load(open('datas/w2n.json'))
    fin=open('datas/train.txt',encoding='utf-8',errors='ignore')
    trains=fin.readlines()
    random.shuffle(trains)
    for i in range(0,len(trains),64):
        ed=min(len(trains),i+64)
        part=trains[i:ed]
        batch=Batch()
        for line in part:
            label=int(line[0]) #类别不超过10
            line=line.strip()[1:]
            line=line.split(' ')
            inputx=[]
            for word in line:
                if word == ' ' or word == '':
                    continue
                inputx.append(w2n.get(word,0))
                if len(inputx)>=sequence_length:
                    break
            if len(inputx)<sequence_length:
                inputx.extend([0]*(sequence_length-len(inputx)))
            
            inputy=[0]*num_classes
            inputy[label]=1
            batch.input_x.append(inputx)
            batch.input_y.append(inputy)
        batches.append(batch)
    return batches

                


if __name__=="__main__":
    batches=getbatch()
    batch=batches[0]
    x=batch.input_x
    y=batch.input_y
    for i in x:
        print(i)