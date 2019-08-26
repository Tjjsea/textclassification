#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import os
import json
import jieba
import random
import numpy as np

Max_Sequence_Length=200
Max_Sentence_Length=144
Max_Document_Length=96

def process1(data):
    '''
    将asr的短句合并，使合并后最大句子长度不超过Max_Sentence_Length
    '''
    predata=[]
    for line in data:
        line=line.strip()
        setes=line.split(' ')
        label=setes[0]
        setes=setes[1:]
        document=label+' '
        s=''
        for i in range(len(setes)):
            if len(s)+len(setes[i])<=Max_Sequence_Length:
                s+=setes[i]
            else:
                if i==len(setes)-1:
                    document=document+s+' '+setes[i]
                    s=''
                else:
                    document=document+s+' '
                    s=setes[i]
        if s:
            document=document+s
        predata.append(document)
    return predata

def process2():
    '''
    将asr的短句合并，使合并后最大句子长度不超过Max_Sequence_Length
    '''
    chdata=json.load(open('ch_en/chinese.json'))
    endata=json.load(open('ch_en/english.json'))

    prechdata=process1(chdata)
    preendata=process1(endata)
    json.dump(prechdata,open('ch_en/prechinese.json','w',encoding='utf-8'))
    json.dump(preendata,open('ch_en/preenglish.json','w',encoding='utf-8'))

def process3(data):
    '''
    为训练集、测试集等分词，并转换成数字
    '''
    w2n=json.load(open('data/w2n.json'))
    numdata=[]
    for line in data:
        line=line.strip()
        if line==' ' or line=='':
            continue
        line=line.split(' ')
        document=[]
        label=int(line[0])
        line=line[1:]
        for sts in line:
            sts=list(jieba.cut(sts))
            sentence=[]
            for word in sts:
                num=w2n.get(word,0)
                sentence.append(num)
                if len(sentence)>=Max_Sentence_Length:
                    break
            if len(sentence)<Max_Sentence_Length:
                sentence.extend([0]*(Max_Sentence_Length-len(sentence)))
            if len(sentence)!=Max_Sentence_Length:
                print(len(sentence))
            document.append(sentence)
            if len(document)>=Max_Document_Length:
                break
        if len(document)<Max_Document_Length:
            temp=[[0]*Max_Sentence_Length for i in range(Max_Document_Length-len(document))]
            document.extend(temp)
        print('length of document %d' % len(document))
        doc=np.array(document,dtype=np.int32)
        document.insert(0,label)
        
        numdata.append(document)
    return numdata    

def hansplit():
    '''
    划分训练集、验证集、测试集
    '''
    chdata=json.load(open('ch_en/prechinese.json'))
    endata=json.load(open('ch_en/preenglish.json'))
    cutendata=endata[:len(chdata)]
    alldata=chdata+cutendata
    random.shuffle(alldata)
    cut1=int(len(alldata)/5*3)
    cut2=int(len(alldata)/5*4)
    train=alldata[:cut1]
    dev=alldata[cut1:cut2]
    test=alldata[cut2:]

    numtrain=process3(train)
    numdev=process3(dev)
    numtest=process3(test)
    print("trans finished")

    json.dump(numtrain,open('ch_en/numtrain.json','w',encoding='utf-8'))
    json.dump(numdev,open('ch_en/numdev.json','w',encoding='utf-8'))
    json.dump(numtest,open('ch_en/numtest.json','w',encoding='utf-8'))

def splitdata():
    '''
    将合并后的句子划分训练集、验证集、测试集，并全部组合作为训练词向量的语料
    '''
    chseg=process1(json.load(open('data/chinese.json')))
    enseg=process1(json.load(open('data/english.json')))

    random.shuffle(enseg)
    cuten=enseg[:len(chseg)]
    ald=chseg+cuten
    random.shuffle(ald)
    cut1=int(len(ald)/5*3)
    cut2=int(len(ald)/5*4)
    train=ald[:cut1]
    dev=ald[cut1:cut2]
    test=ald[cut2:]
    fout=open('data/train.txt','w',encoding='utf-8')
    for line in train:
        fout.write(line)
        fout.write('\n')
    fout=open('data/test.txt','w',encoding='utf-8')
    for line in test:
        fout.write(line)
        fout.write('\n')
    fout=open('data/dev.txt','w')
    for line in dev:
        fout.write(line)
        fout.write('\n')
    
    ald=chseg+enseg
    fout=open('data/material.txt','w',encoding='utf-8')
    for line in ald:
        line=line[2:]
        fout.write(line)
        fout.write('\n')
    
def getw2n():
    fin=open('data/material.txt',encoding='utf-8')
    datas=fin.readlines()
    w2n={"UNK":0}
    count=1
    for line in datas:
        line=line.strip()
        words=line.split(' ')
        for word in words:
            if not word:
                continue
            if word not in w2n:
                w2n[word]=count
                count+=1
    json.dump(w2n,open('data/w2n.json','w',encoding='utf-8'))

def statis():
    chdata=json.load(open('ch_en/prechinese.json'))
    endata=json.load(open('ch_en/preenglish.json'))
    alldata=chdata+endata

    document_length=[]
    sentence_length=[]
    for line in alldata:
        line=line.strip()
        line=line.split(' ')
        label=line[0]
        line=line[1:]
        dl=0
        for sts in line:
            sts=sts.strip()
            if sts=='' or sts==' ':
                continue
            dl+=1
            sts=list(jieba.cut(sts))
            sl=len(sts)
            sentence_length.append(sl)
        document_length.append(dl)
    json.dump(document_length,open('data/document_length.json','w'))
    json.dump(sentence_length,open('data/sentence_length.json','w'))

def statis2():
    document_length=json.load(open('data/document_length.json'))
    sentence_length=json.load(open('data/sentence_length.json'))

    dbucket=[0]*(max(document_length)+1)
    sbucket=[0]*(max(sentence_length)+1)

    for i in document_length:
        dbucket[i]+=1
    for i in sentence_length:
        sbucket[i]+=1
    #d=range(max(document_length)+1)
    #s=range(max(sentence_length)+1)
    #print(list(zip(d,dbucket)))
    #print(list(zip(s,sbucket)))
    print(sum(document_length)/len(document_length))
    print(sum(sentence_length)/len(sentence_length))

def statis3():
    train=json.load(open('ch_en/train.json'))
    print(len(train))



if __name__=='__main__':
    hansplit()
    pass