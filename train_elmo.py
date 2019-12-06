#!/usr/bin/env python
# -*- coding: UTF-8 -*-

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

#o_path=os.getcwd()
#sys.path.append("../")  #models不在当前文件夹下，将其引入
from models import ELMo

ap=argparse.ArgumentParser()
ap.add_argument("--mode",required=False,type=str,help="mode")
ap.add_argument("--GPU",required=False,type=str,help="which GPU to use")
ap.add_argument("--epochs",required=False,type=int,help="number of epochs")
ap.add_argument("--lr",required=False,type=float,help="learning rate")
args=ap.parse_args()

GPU=args.GPU if args.GPU else '3'
epochs=args.epochs if args.epochs else 5
config=Config()
if args.lr:
    config.learning_rate=args.lr

os.environ["CUDA_VISIBLE_DEVICES"] = GPU #use GPU with ID=0
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM
tfconfig.gpu_options.allow_growth = True #allocate dynamically
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #屏蔽warning信息

batch_size=128
steps_per_checkpoint=10
#config.getElmoEmbedding()

def get_batch():
    data=json.load(open('../data/edatrain.json',encoding='utf-8'))
    random.shuffle(data)

    for i in range(0,len(data),batch_size):
        ed=min(i+batch_size,len(data))
        part=data[i:ed]
        batch=Batch()
        for sen in part:
            label=int(sen[0])
            words=sen.split()[1:]
            if len(words)>=config.sequence_length:
                words=words[:config.sequence_length]
                batch.sequence_length.append(config.sequence_length)
            if len(words)<config.sequence_length:
                batch.sequence_length.append(len(words))
                words.extend(['UNK']*(config.sequence_length-len(words)))
            batch.input_x.append(words)
            #batch.input_x=elmo(batch.input_x)[0]
            if config.num_classes==1:
                inputy=[label]
            else:
                inputy=[0]*config.num_classes
                inputy[label]=1
            batch.input_y.append(inputy)               #y是one-hot形式的
        batch.input_x=elmo(batch.input_x)[0]
        yield batch

def elmo(reviews):    
    inputDataIndex=batcher.batch_sentences(reviews)
    elmoInputVec=sess.run([elmoInput['weighted_op']],feed_dict={inputData:inputDataIndex})
    return elmoInputVec

if __name__=='__main__':
    sess=tf.Session()
    with sess.as_default():
        model=ELMo(flags=config)

        abilm=BidirectionalLanguageModel(config.option_file,config.weight_file,use_character_inputs=False,embedding_weight_file=config.tokenEmbeddingFile)
        inputData=tf.placeholder('int32',shape=(None,None))
        inputEmbeddingsOp=abilm(inputData)
        elmoInput=weight_layers('input',inputEmbeddingsOp,l2_coef=0.0)
        batcher=TokenBatcher(config.vocab_file)
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(config.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())
    
        current_step = 0
        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            for batch in get_batch():
                loss,acc=model.train(sess, batch)
                current_step += 1
                if current_step % steps_per_checkpoint == 0:
                    print("----- Step %d -- Loss %.4f -- Acc %.4f" % (current_step, loss, acc))
                    checkpoint_path = os.path.join(config.model_dir, config.model_name)
                    model.saver.save(sess, checkpoint_path, global_step=current_step)
    sess.close()