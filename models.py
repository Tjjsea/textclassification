#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import numpy as np
import tensorflow as tf

class textCNN():
    def __init__(self,flags,filter_sizes,w2v):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32,name="keep_prob")
        self.global_step=tf.placeholder(tf.int32,name="global_step")
        self.l2Loss=tf.constant(0.0)
        
        with tf.name_scope("word-embedding"):
            if w2v is None:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                self.embedding=tf.Variable(w2v,trainable=False,dtype=tf.float32)
            self.embedded=tf.nn.embedding_lookup(self.embedding,self.input_x)
        self.embedded=tf.reshape(self.embedded,[-1,flags.sequence_length,flags.embedding_dim,1])
        conv_outs=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-%d" % i):
                h=self.conv2d(self.embedded,[filter_size,flags.embedding_dim,1,flags.featuremaps],"convolution-%d" % i,tf.nn.relu)
                output=tf.nn.max_pool(h,ksize=[1,flags.sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID') #[batch_size,1,1,featuremaps]
                conv_outs.append(output)
        conv_output=tf.concat(conv_outs,-1) #[batch_size,1,1,featuremaps*len(filter_sizes)]
        conv_out=tf.reshape(conv_output,[-1,flags.featuremaps*len(filter_sizes)])
        self.conv_out=tf.nn.dropout(conv_out,self.keep_prob)

        with tf.name_scope("output"):
            wout=tf.get_variable("weight",shape=[flags.featuremaps*len(filter_sizes),flags.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            bout=tf.get_variable("bias",shape=[flags.num_classes],initializer=tf.zeros_initializer())
            self.l2Loss+=tf.nn.l2_loss(wout)
            self.logits=tf.nn.xw_plus_b(self.conv_out,wout,bout)
            if flags.num_classes==1:
                self.predictions=tf.cast(tf.greater_equal(self.logits,0.0),tf.int32,name="predictions")
                self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.input_y, dtype=tf.float32)))
                
            else:
                self.predictions=tf.argmax(self.logits,-1)
                self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predictions,tf.cast(self.input_y,dtype=tf.int32)),tf.float32))
            self.loss+=(flags.l2RegLambda*self.l2Loss)
        self.learning_rate=tf.train.exponential_decay(learning_rate=flags.learning_rate,global_step=self.global_step,decay_steps=100,decay_rate=0.9,staircase=True)
        self.train_op=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver(tf.global_variables())

    def conv2d(self,input,shape,scope_name,activation_function=None):
        with tf.variable_scope(scope_name):
            W=tf.get_variable("filter",shape,initializer=tf.truncated_normal_initializer())
            b=tf.get_variable("bias",shape[-1],initializer=tf.zeros_initializer())
            out=tf.nn.conv2d(input,W,strides=[1,1,1,1],padding='VALID')+b
            if not activation_function:
                return activation_function(W)
            else:
                return out
    
    def train(self,sess,batch,global_step):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5,
                   self.global_step:global_step}
        _,loss,acc=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
    def eval(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.keep_prob:0.5}
        pre=sess.run(self.predictions,feed_dict=feed_dict)
        return pre
    
    def demo(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        pre,acc=sess.run([self.predictions,self.accuracy],feed_dict=feed_dict)
        return pre,acc

class bilstm():
    def __init__(self,flags,hiddensizes,w2v):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")

        with tf.name_scope("word-embedding"):
            if not w2v:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                self.embedding=tf.Variable(w2v,trainable=False,dtype=tf.float32)
            self.embedded=tf.nn.embedding_lookup(self.embedding,self.input_x)

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
            b=tf.Variable(tf.constant(0.1,shape=[flags.num_classes]),name="b")
            self.scores=tf.nn.xw_plus_b(self.bi_out,W,b)
        self.pre=tf.argmax(self.scores,1)
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.scores))
        self.train_op=tf.train.AdamOptimizer(flags.learning_rate).minimize(self.loss)
        correct_predictions=tf.equal(self.pre,tf.argmax(self.input_y,-1))
        self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
        self.saver=tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        _,loss,acc,=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
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
        self.sequence_length=tf.placeholder(tf.int32,[None],name="sequence_length")
        self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")
        self.l2Loss=tf.constant(0.0)

        self.flags=flags
        self.hiddensizes=hiddensizes

        with tf.name_scope("word-embedding"):
            if w2v is None:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                self.embedding=tf.Variable(w2v,trainable=False,dtype=tf.float32)
            self.embedded=tf.nn.embedding_lookup(self.embedding,self.input_x)

        with tf.name_scope("bilstm"):
            binput=self.embedded
            for i,hiddensize in enumerate(hiddensizes):
                with tf.variable_scope("bilstm-%d" % i):
                    fw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    bw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    outputs,_=tf.nn.bidirectional_dynamic_rnn(fw,bw,binput,self.sequence_length,dtype=tf.float32)
                    binput=tf.concat(outputs,-1)
            self.bilstm_out=tf.split(binput,2,-1)
        
        '''
        with tf.name_scope("attention"):
            H=self.bilstm_out[0]+self.bilstm_out[1] #[batch_size,sequence_length,hiddensizes[-1]]
            M=tf.reshape(tf.tanh(H),[-1,hiddensizes[-1]])
            W=tf.get_variable("weigths",shape=[hiddensizes[-1],1],initializer=tf.contrib.layers.xavier_initializer())
            alpha=tf.reshape(tf.matmul(M,W),[-1,flags.sequence_length])
            alpha=tf.nn.softmax(alpha) #[batch_size*sequence_length,1]
            alpha=tf.reshape(alpha,[-1,1,flags.sequence_length])
            r=tf.matmul(alpha,H) #[batch_size,1,hiddensizes[-1]]
            self.hstar=tf.reshape(tf.tanh(r),[-1,hiddensizes[-1]])
        '''

        with tf.name_scope("attention"):
            H=self.bilstm_out[0]+self.bilstm_out[1] #[batch_size,sequence_length,hiddensizes[-1]]
            self.hstar=self.attention(H)
        
        with tf.name_scope("classifying"):
            W=tf.get_variable("weights",shape=[hiddensizes[-1],flags.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable("bias",shape=[flags.num_classes],initializer=tf.zeros_initializer())
            self.l2Loss+=tf.nn.l2_loss(W)
            self.logits=tf.matmul(self.hstar,W)+b

            if flags.num_classes==1:
                self.predictions=tf.cast(tf.greater_equal(self.logits,0.0),tf.int32,name="predictions")
                self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=tf.cast(self.input_y,dtype=tf.float32)))
            else:
                self.pre=tf.argmax(self.logits,-1)
                self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y))
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predictions,tf.cast(self.input_y,dtype=tf.int32)),tf.float32))
            self.loss+=(flags.l2RegLambda*self.l2Loss)
        self.train_op=tf.train.AdamOptimizer(flags.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver(tf.global_variables())
        
    def train(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.sequence_length:batch.sequence_length,
                   self.keep_prob:0.5}
        _,loss,acc=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
    def eval(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.sequence_length:batch.sequence_length,
                   self.keep_prob:0.5}
        accuracy=sess.run(self.accuracy,feed_dict=feed_dict)
        return accuracy
    
    def demo(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.sequence_length:batch.sequence_length,
                   self.keep_prob:0.5}
        pre,acc=sess.run([self.predictions,self.accuracy],feed_dict=feed_dict)
        return pre,acc
    
    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hiddensizes[-1]
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.flags.sequence_length])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.flags.sequence_length, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.keep_prob)
        
        return output
    
class charCNN():
    def __init__(self,flags):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32, name="keep_prob")

        #todo
    
class transformer():
    def __init__(self,flags,w2v):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, flags.sequence_length], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None,None], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.embeddedPosition = tf.placeholder(tf.float32, [None, flags.sequence_length, flags.sequence_length], name="embeddedPosition") #one-hot
        
        self.flags = flags
        
        # 定义l2损失
        l2Loss = tf.constant(0.0)
        
        # 词嵌入层, 位置向量的定义方式有两种：一是直接用固定的one-hot的形式传入，然后和词向量拼接，在当前的数据集上表现效果更好。另一种
        # 就是按照论文中的方法实现，这样的效果反而更差，可能是增大了模型的复杂度，在小数据集上表现不佳。
        
        with tf.name_scope("embedding"):
            if not w2v:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0))
            else:
                # 利用预训练的词向量初始化词嵌入矩阵
                self.embedding=tf.Variable(tf.cast(w2v, dtype=tf.float32, name="word2vec") ,name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded = tf.nn.embedding_lookup(self.embedding, self.inputX)
            self.embeddedwords = tf.concat([self.embedded, self.embeddedPosition], -1)

        with tf.name_scope("transformer"):
            self.transinput=self.embeddedwords
            for i in range(flags.numBlocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
            
                    # 维度[batch_size, sequence_length, embedding_size]
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.inputX, queries=self.transinput,
                                                            keys=self.transinput)
                    # 维度[batch_size, sequence_length, embedding_size]
                    self.transinput = self._feedForward(multiHeadAtt, 
                                                           [flags.filters,flags.embedding_dim + flags.sequence_length])
                
            outputs = tf.reshape(self.transinput, [-1, flags.sequence_length * (flags.embedding_dim + flags.sequence_length)])

        outputSize = outputs.get_shape()[-1].value
        
        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)
    
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, flags.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[flags.num_classes]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name="logits")
            
            if flags.num_classes == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
            elif flags.num_classes > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
        
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            
            if flags.num_classes == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.inputY, dtype=tf.float32))
            elif flags.num_classes > 1:
                #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                
            self.loss = tf.reduce_mean(losses) + flags.l2RegLambda * l2Loss
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predictions,tf.cast(self.inputY,tf.int32)),tf.float32))
        self.train_op=tf.train.AdamOptimizer(flags.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver(tf.global_variables())
            
    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = self.flags.epsilon

        inputsShape = inputs.get_shape() # [batch_size, sequence_length, embedding_dim]

        paramsShape = inputsShape[-1:]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        
        outputs = gamma * normalized + beta

        return outputs
            
    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值
        
        numHeads = self.flags.numHeads
        keepProp = self.flags.keepProp
        
        if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0) 
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0) 
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(rawKeys, [numHeads, 1]) 

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings, scaledSimilary) # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings, maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)
        
        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络
        
        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs
    
    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize
        
        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i-i%2) / embeddingSize) for i in range(embeddingSize)] 
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded
    
    def train(self,sess,batch):
        feed_dict={self.inputX:batch.input_x,
                   self.inputY:batch.input_y,
                   self.embeddedPosition:batch.transformer_position,
                   self.dropoutKeepProb:0.5}
        _,loss,acc=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
    def dev(self,sess,batch):
        feed_dict={self.inputX:batch.input_x,
                   self.embeddedPosition:batch.transformer_position,
                   self.dropoutKeepProb:0.5}
        pre=sess.run(self.predictions,feed_dict=feed_dict)
        return pre
    
    def demo(self,sess,batch):
        feed_dict={self.inputX:batch.input_x,
                   self.inputY:batch.input_y,
                   self.embeddedPosition:batch.transformer_position,
                   self.dropoutKeepProb:0.5}
        pre,acc=sess.run([self.predictions,self.accuracy],feed_dict=feed_dict)
        return pre,acc
    
class ELMo():
    def __init__(self,flags,hiddensizes):
        self.input_x=tf.placeholder(tf.float32,[None,flags.sequence_length,flags.embedding_dim],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None,flags.num_classes],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32,name="drop_out_keep_prob")

        with tf.name_scope("embedding"):
            embedding=tf.get_variable("embedding",shape=[flags.embedding_dim,flags.embedding_dim],initializer=tf.contrib.layers.xavier_initializer())
            xinput=tf.reshape(self.input_x,[-1,flags.embedding_dim])
            embedded=tf.matmul(xinput,embedding)
            self.embedded=tf.reshape(embedded,[-1,flags.sequence_length,flags.embedding_dim])
            self.embedded=tf.nn.dropout(self.embedded,self.keep_prob)

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
            W=tf.get_variable("weigths",shape=[hiddensizes[-1],1],initializer=tf.contrib.layers.xavier_initializer())
            alpha=tf.reshape(tf.matmul(M,W),[-1,flags.sequence_length])
            alpha=tf.nn.softmax(alpha) #[batch_size*sequence_length,1]
            alpha=tf.reshape(alpha,[-1,1,flags.sequence_length])
            r=tf.matmul(alpha,H) #[batch_size,1,hiddensizes[-1]]
            self.hstar=tf.reshape(tf.tanh(r),[-1,hiddensizes[-1]])
        
        with tf.name_scope("classifying"):
            W=tf.get_variable("weights",shape=[hiddensizes[-1],flags.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable("bias",shape=[flags.num_classes],initializer=tf.zeros_initializer())
            self.out=tf.nn.softmax(tf.matmul(self.hstar,W)+b)
            self.pre=tf.argmax(self.out,-1)
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.out))
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.pre,tf.argmax(self.input_y,-1)),tf.float32))
        self.train_op=tf.train.GradientDescentOptimizer(flags.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        _,loss,acc=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
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
    
class HAN():
    def __init__(self,flags,w2v,whiddensizes,shiddensizes):
        self.input_x=tf.placeholder(tf.int32,[None,None,None],name="input_x") #[batch_size,document_length,sentence_length]
        self.input_y=tf.placeholder(tf.int32,[None,None],name="input_y")
        self.keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.l2Loss=tf.constant(0.0)

        with tf.name_scope("word-embedding"):
            if w2v is None:
                self.embedding=tf.Variable(tf.random_uniform([flags.vocab_size,flags.embedding_dim],-1.0,1.0),name="embedding")
            else:
                self.embedding=tf.Variable(w2v,dtype=tf.float32,name="embedding")
            inputx=tf.reshape(self.input_x,[-1,flags.sentence_length])
            self.embedded=tf.nn.embedding_lookup(self.embedding,inputx)
            self.embedded=tf.reshape(self.embedded,[-1,flags.sentence_length,flags.embedding_dim])
        
        with tf.name_scope("word-encoder"):
            weinput=self.embedded
            for i,hid in enumerate(whiddensizes):
                with tf.variable_scope("wordencoder-%d" % i):
                    fw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hid),output_keep_prob=self.keep_prob)
                    bw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hid),output_keep_prob=self.keep_prob)
                    outputs,_=tf.nn.bidirectional_dynamic_rnn(fw,bw,weinput,dtype=tf.float32)
                    weinput=tf.concat(outputs,-1)
            self.word_encoder=weinput #[batch_size*document_length,sentence_length,whiddensizes[-1]*2]

        with tf.variable_scope("word-attention"):
            h=tf.reshape(self.word_encoder,[-1,whiddensizes[-1]*2])
            W=tf.get_variable(name="weights",shape=[whiddensizes[-1]*2,flags.wt_hidunits],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable(name="bias",shape=[flags.wt_hidunits],initializer=tf.zeros_initializer())
            uw=tf.get_variable(name="Uw",shape=[flags.wt_hidunits,1],initializer=tf.contrib.layers.xavier_initializer())
            #u=tf.tanh(tf.matmul(h,W)+b)
            #u=tf.matmul(u,uw) #[batch_size*document_length*sentence_length,1]
            u=tf.reshape(tf.matmul(tf.tanh(tf.matmul(h,W)+b),uw),[-1,flags.document_length,flags.sentence_length])
            alpha=tf.reshape(tf.nn.softmax(u),[-1,1])
            s=h*alpha #???
            s=tf.reshape(s,[-1,flags.document_length,flags.sentence_length,whiddensizes[-1]*2])
            #s=tf.reduce_sum(s,-2) #[batch_size,document_length,whiddensizes[-1]*2]
            self.sentence=tf.reduce_sum(s,-2)
        
        with tf.name_scope("sentence-encoder"):
            seinput=self.sentence
            for i,hid in enumerate(shiddensizes):
                with tf.variable_scope("sentenceencoder-%d" % i):
                    fw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hid),output_keep_prob=self.keep_prob)
                    bw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hid),output_keep_prob=self.keep_prob)
                    outputs,_=tf.nn.bidirectional_dynamic_rnn(fw,bw,seinput,dtype=tf.float32)
                    seinput=tf.concat(outputs,-1)
            self.sentence_encoder=seinput #[batch_size,document_length,shiddensizes[-1]*2]

        with tf.variable_scope("sentence_attention"):
            sh=tf.reshape(self.sentence_encoder,[-1,shiddensizes[-1]*2])
            W=tf.get_variable(name="weights",shape=[shiddensizes[-1]*2,flags.st_hidunits],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable(name="bias",shape=[flags.st_hidunits],initializer=tf.zeros_initializer())
            us=tf.get_variable(name="Us",shape=[flags.st_hidunits,1],initializer=tf.contrib.layers.xavier_initializer())
            #su=tf.tanh(tf.matmul(sh,W)+b)
            #su=tf.matmul(su,us) #[batch_size*document_length,1]
            su=tf.reshape(tf.matmul(tf.tanh(tf.matmul(sh,W)+b),us),[-1,flags.document_length])
            #alpha=tf.nn.softmax(su)
            alpha=tf.reshape(tf.nn.softmax(su),[-1,1])
            #v=tf.multiply(alpha,sh) #???[batch_size*document_lenth,shiddensizes[-1]*2]
            v=tf.reshape(tf.multiply(alpha,sh),[-1,flags.document_length,shiddensizes[-1]*2])
            #v=tf.reduce_sum(v,-2) #[batch_size,shiddensizes[-1]*2]
            self.document=tf.reduce_sum(v,-2)
        
        with tf.variable_scope("classifying"):
            W=tf.get_variable(name="weights",shape=[shiddensizes[-1]*2,flags.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b=tf.get_variable(name="bias",shape=[flags.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            self.l2Loss+=tf.nn.l2_loss(W)
            self.scores=tf.matmul(self.document,W)+b

            if flags.num_classes==1:
                self.pre=tf.cast(tf.greater_equal(self.scores,0.0),tf.int32,name="predictions")
                self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,labels=tf.cast(self.input_y,dtype=tf.float32)))
            else:
                self.pre=tf.argmax(self.scores,-1)
                self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y))
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.pre,tf.cast(self.input_y,dtype=tf.int32)),tf.float32))
            self.loss+=(flags.l2RegLambda*self.l2Loss)
        self.train_op=tf.train.AdamOptimizer(flags.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        _,loss,acc=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
    def eval(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        accuracy=sess.run(self.accuracy,feed_dict=feed_dict)
        return accuracy
    
    def demo(self,sess,batch):
        feed_dict={self.input_x:batch.input_x,
                   self.input_y:batch.input_y,
                   self.keep_prob:0.5}
        pre,acc=sess.run([self.pre,self.accuracy],feed_dict=feed_dict)
        return pre,acc

class TextCNN(object):
    """
    Text CNN 用于文本分类
    """
    def __init__(self, config, filterSizes):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequence_length], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None,config.num_classes], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        
        # 定义l2损失
        l2Loss = tf.constant(0.0)
        
        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.random_uniform([config.vocab_size,config.embedding_dim],-1.0,1.0))
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filterSize in enumerate(filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filterShape = [filterSize, config.embedding_dim, 1, config.featuremaps]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.featuremaps]), name="b")
                conv = tf.nn.conv2d(
                    self.embeddedWordsExpanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                
                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.sequence_length - filterSize + 1, 1, 1],  # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        numFiltersTotal = config.featuremaps * len(filterSizes)
        
        # 池化后的维度不变，按照最后的维度channel来concat
        self.hPool = tf.concat(pooledOutputs, 3)
        
        # 摊平成二维的数据输入到全连接层
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

        # dropout
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)
       
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB= tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="logits")
            if config.num_classes == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
            elif config.num_classes > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
            
            print(self.predictions)
        
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            
            if config.num_classes == 1:
                #losses=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=tf.reshape(tf.cast(tf.argmax(self.inputY,-1),dtype=tf.float32),[-1,1]))
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.inputY, dtype=tf.float32))
            elif config.num_classes > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                
            self.loss = tf.reduce_mean(losses) + config.l2RegLambda * l2Loss
            self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.predictions,tf.cast(self.inputY,dtype=tf.int32)),tf.float32))
        self.train_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
        self.saver=tf.train.Saver(tf.global_variables())

    def train(self,sess,batch):
        feed_dict={self.inputX:batch.input_x,
                   self.inputY:batch.input_y,
                   self.dropoutKeepProb:0.5}
        _,loss,acc=sess.run([self.train_op,self.loss,self.accuracy],feed_dict=feed_dict)
        return loss,acc
    
    def demo(self,sess,batch):
        feed_dict={self.inputX:batch.input_x,
                   self.inputY:batch.input_y,
                   self.dropoutKeepProb:0.5}
        pre,acc=sess.run([self.predictions,self.accuracy],feed_dict=feed_dict)
        return pre,acc
