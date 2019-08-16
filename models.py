#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import numpy as np
import tensorflow as tf

class textCNN():
    def __init__(self,flags,filter_sizes,w2v):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32,name="keep_prob")
        
        with tf.name_scope("word-embedding"):
            if not w2v:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                self.embedding=tf.get_variable("embedding",initializer=w2v.vectors.astype(np.float32))
            self.embedded=tf.nn.lookup(self.embedding,self.input_x)
        
        conv_outs=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-%d" % i):
                h=self.conv2d(self.word_embedded,[filter_size,flags.embedding_dim,1,flags.featuremaps],"convolution-%d" % i,tf.nn.relu)
                output=tf.nn.max_pool(h,ksize=[1,flags.sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID') #[batch_size,1,1,featuremaps]
                conv_outs.append(output)
        conv_output=tf.nn.dropout(tf.concat(conv_outs,-1),self.keep_prob) #[batch_size,1,1,featuremaps*len(filter_sizes)]
        self.conv_out=tf.reshape(conv_output,[-1,flags.featuremaps*len(filter_sizes)])

        with tf.name_scope("output"):
            wout=tf.Variable(tf.random_normal([flags.featuremaps*len(filter_sizes),flags.num_classes],dtype=tf.float32))
            bout=tf.Variable(tf.constant(0.1),shape=[flags.num_classes])
            self.out=tf.nn.softmax(tf.nn.xw_plus_b(self.conv_out,wout,bout))
            self.pres=tf.argmax(self.out,1)
        
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.out))
        self.accuracy=tf.equal(self.pres,tf.argmax(self.input_y,1))
        self.train_op=tf.train.AdamOptimizer(flags.learning_rate).minimize(self.loss)

    def conv2d(self,input,shape,scope_name,activation_function=None):
        with tf.variable_scope(scope_name):
            W=tf.get_variable("filter",shape,initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable("bias",shape[-1],initializer=tf.zeros_initializer())
            out=tf.nn.conv2d(input,W,strids=[1,1,1,1],padding='VALID')+b
            if not activation_function:
                return activation_function(W)
            else:
                return out
    
    def train(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        _,loss=sess.run([self.train_op,self.loss],feed_dict=feed_dict)
        return loss
    
    def eval(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        accuracy=sess.run(self.accuracy,feed_dict=feed_dict)
        return accuracy
    
    def demo(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.keep_prob:0.5}
        pre=sess.run(self.pres)
        return pre

class bilstm():
    def __init__(self,flags,hiddensizes,w2v):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")

        with tf.name_scope("word-embedding"):
            if not w2v:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                self.embedding=tf.get_variable("embedding",initializer=w2v.vectors.astype(np.float32))
            self.embedded=tf.nn.lookup(self.embedding,self.input_x)

        with tf.name_scope("bilstm"):
            binput=self.embedded
            for i,hiddensize in enumerate(hiddensizes):
                with tf.variable_scope("bilstm-%d" % i):
                    fw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    bw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    outputs,_=tf.nn.bidirectional_dynamic_rnn(fw,bw,binput,dtype=tf.float32)
                    binput=tf.concat(outputs,-1)
            self.bi_out=binput[:,-1,:]
        
        self.bi_out=tf.reshape(self.bi_out,[-1,hiddensizes[-1]*2])
        with tf.name_scope("softmax"):
            W=tf.get_variable("W",shape=[hiddensizes[-1]*2,flags.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable("b",shape=[flags.num_classes],initializer=tf.zeros_initializer())
            self.out=tf.nn.softmax(tf.nn.xw_plus_b(self.bi_out,W,b))
        self.pre=tf.argmax(self.out,1)
        self.loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.out)
        self.train_op=tf.train.AdamOptimizer(flags.learning_rate).minimize(self.loss)
        self.accuracy=tf.equal(self.pre,tf.argmax(self.input_y))

    def train(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        _,loss=sess.run([self.train_op,self.loss],feed_dict=feed_dict)
        return loss
    
    def eval(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        accuracy=sess.run(self.accuracy,feed_dict=feed_dict)
        return accuracy
    
    def demo(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.keep_prob:0.5}
        pre=sess.run(self.pres)
        return pre
    
class bilstm_attention():
    def __init__(self,flags,w2v,hiddensizes):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")

        with tf.name_scope("word-embedding"):
            if not w2v:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                self.embedding=tf.get_variable("embedding",initializer=w2v.vectors.astype(np.float32))
            self.embedded=tf.nn.lookup(self.embedding,self.input_x)

        with tf.name_scope("bilstm"):
            binput=self.embedded
            for i,hiddensize in enumerate(hiddensizes):
                with tf.variable_scope("bilstm-%d" % i):
                    fw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    bw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    outputs,_=tf.nn.bidirectional_dynamic_rnn(fw,bw,binput,dtype=tf.float32)
                    binput=tf.concat(outputs,-1)
            self.bilstm_out=tf.split(binput,2,-1)
        
        with tf.name_scope("attention"):
            H=self.bilstm_out[0]+self.bilstm_out[1] #[batch_size,sequence_length,hiddensizes[-1]]
            M=tf.reshape(tf.tanh(H),[-1,hiddensizes[-1]])
            W=tf.get_variable("weigths",shape=[hiddensizes[-1],],initializer=tf.contrib.xavier_initializer())
            alpha=tf.reshape(tf.matmul(M,W),[-1,flags.sequence_length])
            alpha=tf.nn.softmax(alpha) #[batch_size*sequence_length,1]
            alpha=tf.reshape(alpha,[-1,1,flags.sequence_length])
            r=tf.matmul(alpha,H) #[batch_size,1,hiddensizes[-1]]
            self.hstar=tf.reshape(tf.tanh(r),[-1,hiddensizes[-1]])
        
        with tf.name_scope("classifying"):
            W=tf.get_variable("weights",shape=[hiddensizes[-1],flags.num_classes],initializer=tf.contrib.xavier_initializer())
            b=tf.get_variable("bias",shape=[flags.num_classes],initializer=tf.zeros_initializer())
            self.out=tf.nn.softmax(tf.matmul(self.hstar,W),b)
            self.pre=tf.argmax(self.out,-1)
            self.loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.out)
            self.accuracy=tf.equal(self.pre,tf.argmax(self.input_y))
        
    def train_op(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        _,loss=sess.run([self.train_op,self.loss],feed_dict=feed_dict)
        return loss
    
    def eval(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        accuracy=sess.run(self.accuracy,feed_dict=feed_dict)
        return accuracy
    
    def demo(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.keep_prob:0.5}
        pre=sess.run(self.pres)
        return pre
    
class charCNN():
    def __init__(self,flags):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")

        
