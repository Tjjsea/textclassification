#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import numpy as np
import random
import gensim
from gensim.models import word2vec,Word2Vec

def writealphabet():
    alb={"UNK":0}
    A,a=65,97
    count=1
    for i in range(26):
        alb[chr(A+i)]=count
        count+=1
    for i in range(26):
        alb[chr(a+i)]=count
        count+=1
    json.dump(alb,open('dicts/alb.json','w'))

def get_vocabs(filepath):
    '''
    获取词表
    '''
    pass

def getdata(dirpath,outpath):
    '''
    将原始文本进行初步预处理，并拼接到一个文件中
    '''
    labels={"Art":0,"History":1,"Space":2,"Computer":3,"Enviornment":4,"Sports":5}
    predata=[]
    for cates in os.listdir(dirpath):
        catename=cates.split('-')[1]
        label='['+str(labels[catename])+']'
        catepath=os.path.join(dirpath,cates)
        for f in os.listdir(catepath):
            fpath=os.path.join(catepath,f)
            with open(fpath,encoding='gb18030',errors='ignore') as fin:                
                texts=fin.readlines()
                content=texts
                if texts[0][0]=='【':
                    flag=False
                    for idx,line in enumerate(texts):
                        if '正  文' in line:
                            content=texts[idx+1:]
                            flag=True
                            break
                    if not flag:
                        print('more pattern '+fpath)
                elif len(texts)>6:
                    if texts[3].strip()=='' or texts[4].strip()=='' or texts[5].strip()=='':
                        content=texts[4:]
                text=label+' '
                for line in content:
                    line=line.strip()
                    if not line:
                        continue
                    text+=line
                predata.append(text)
    random.shuffle(predata)
    with open(outpath,'w',encoding='utf-8') as fout:
        for line in predata:
            fout.write(line)
            fout.write('\n')

def trainw2v():
    '''
    基于现有asr语料训练词向量
    '''
    data=word2vec.LineSentence('data/material.txt')
    model=Word2Vec(data,sg=0,size=200,iter=8)
    #model.wv.save_word2vec_format("model/word2Vec"+".bin",binary=True)
    model.save('model/word2vec.txt')

def getembedding():
    model=Word2Vec.load('model/word2vec.txt')
    w2n=json.load(open('data/w2n.json'))
    sortw2n=sorted(w2n.items(),key=lambda x:x[1])
    unk=np.random.randn(200)
    vecs=[unk]
    for line in w2n:
        word=line[0]
        if word not in model:
            wvec=np.random.randn(200)
        else:
            wvec=model[word]
        vecs.append(wvec)
    vecs=np.array(vecs)
    np.savetxt('data/w2v.txt',vecs)



if __name__=="__main__":
    #trainw2v()
    getembedding()