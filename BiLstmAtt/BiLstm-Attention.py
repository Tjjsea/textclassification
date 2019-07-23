import tensorflow as tf

class BiLstmAttention():
    def __init__(self,config):
        self.inputx=tf.placeholder(tf.int32,[None,config.sequencelength],name='inputx')
        self.inputy=tf.placeholder(tf.int32,[None],name='inputy')
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')

        l2loss=tf.constant(0,0)

        with tf.name_scope('embedding'):
            embedding=tf.Variable(tf.truncated_normal(shape=[config.vocab_size,config.embedding_dim],name='encoder_embedding'))
            embedded=tf.nn.embedding_lookup(embedding,self.inputx)
        
        with tf.name_scope('BiLstm'):
            for idx,hiddensize in enumerate(config.hiddensizes):
                with tf.name_scope("BiLSTM-%d" %(idx)):
                    lstmfw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    lstmbw=tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hiddensize),output_keep_prob=self.keep_prob)
                    outputs_,current_state=tf.nn.bidirectional_dynamic_rnn(lstmfw,lstmbw,embedded,dtype=tf.float32,scope='BiLSTM-%d' %(idx))
                    embedded=tf.concat(outputs_,2)
                
        outputs=tf.split(embedded,2,-1)
        with tf.name_scope('Attention'):
            H=outputs[0]+outputs[1]
            hsize=config.hiddensizes[-1]
            W=tf.Variable(tf.random_normal([hsize],stddev=0.1))
            M=tf.tanh(H)
            newM=tf.matmul(tf.reshape(M,[-1,hsize]),tf.reshape(W,[-1,1]))
            newM=tf.reshape(newM,[-1,config.sequencelength])
            self.alpha=tf.nn.softmax(newM)
            r=tf.matmul(tf.transpose(H,[0,2,1]),tf.reshape(self.alpha,[-1,config.sequencelength,10]))
            sequeezeR=tf.reshape(r,[-1,hsize])
            sentenceRepren=tf.tanh(sequeezeR)
            output=tf.nn.dropout(sentenceRepren,self.keep_prob)
            outsize=config.hiddensizes[-1]
        
        with tf.name_scope('Output'):
            OW=tf.get_variable("outputW",shape=[outsize, config.numClasses],initializer=tf.contrib.layers.xavier_initializer())
            OB=tf.Variable(tf.constant(0.1,shape=[config.numClasses]),name="outputB")
            l2loss+=tf.nn.l2_loss(OW)
            l2loss+=tf.nn.l2_loss(OB)
            self.logits=tf.nn.xw_plus_b(output,OW,OB,name="logits")
            self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
        
        with tf.name_scope('Loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputy)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2loss
        
        with tf.name_scope('Accuracy'):
            correct_predictions = tf.equal(self.predictions, self.inputy)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        with tf.name_scope('Optimizer'):
            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # 计算梯度,得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(self.loss)
            # 将梯度应用到变量下，生成训练器
            trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
