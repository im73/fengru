import tensorflow as tf
import numpy as np

from train_model.dl.model_base import AdvancedModel

class Model_LSTM(AdvancedModel):
    def __init__(self, session, env, seq_len, label_len,
                 input_size, output_size, batch_size, **options):
        super(Model_LSTM, self).__init__(session, env, **options)
        self.seq_len = seq_len
        self.label_len = label_len
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        try:
            self.hidden_size = options['hidden_size']
        except KeyError:
            self.hidden_size = 32
        try:
            self.layer_count = options['layer_count']
        except KeyError:
            self.layer_count = 1
        with tf.device(self.device):
            self._init_input()
            self._init_nn()
            self._init_op()
            self._init_saver()
        
        
    def _init_input(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.input_size])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.label_len, self.output_size])
    
    def _init_nn(self):
        with tf.variable_scope('layer_in'):
            # self.layer_in = self.inputs
            #dense:全连接层 units:输出的维度大小，改变inputs的最后一维 inputs:输入该网络层的数据
            self.layer_in = tf.layers.dense(units=self.hidden_size, inputs=self.inputs)#, activation=tf.tanh)
        with tf.variable_scope('layer_rnn', initializer=tf.orthogonal_initializer()):
            #self.rnn = self.add_rnn(layer_count=2, hidden_size=self.hidden_size, initializer=tf.orthogonal_initializer())
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, forget_bias=1.0, activation=tf.tanh, 
                                             initializer=tf.orthogonal_initializer())
                    for _ in range(self.layer_count)]
            self.rnn = tf.nn.rnn_cell.MultiRNNCell(cells)
            #rnn_output:每一个迭代隐状态的输出,它包括了训练中所有隐层状态
            #rnn_states:由(c,h)组成的tuple，大小均为[batch,hidden_size]
            self.rnn_output, self.rnn_states = tf.nn.dynamic_rnn(cell=self.rnn, inputs=self.layer_in, dtype=tf.float32)
            self.rnn_output = self.rnn_output[:, -1]
            print(self.rnn_output) #shape=(?, 32)
        
        with tf.variable_scope('layer_out'):
            #self.outputs = tf.layers.dense(units=self.output_size, inputs=self.rnn_output)
            self.outputs = tf.layers.dense(units=self.output_size*self.label_len,inputs=self.rnn_output)
            print(self.outputs) #shape=(?, self.output_size)
            
    def _init_op(self):
        with tf.variable_scope('loss'):
            self.labels_comp = tf.reshape(self.labels, [-1, self.label_len*self.output_size])
            #self.labels_comp的shape=(?,self.label_len*self.output_size)
            print(self.labels, self.labels_comp) #
            self.loss = tf.losses.mean_squared_error(self.outputs,self.labels_comp) #
            
        with tf.variable_scope('train'):
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
        
        self.session.run(tf.global_variables_initializer())
    
    def _init_saver(self):
        if self.enable_saver:
            self.saver = tf.train.Saver()
            
    def train(self):
        train_steps = self.env.get_steps(self.batch_size,'train')
        for epoch in range(self.train_epochs):
            self.env.reset_count()
            for step in range(train_steps):
                batch_x, batch_y = self.env.get_batch_data(batch_size=self.batch_size, mode='train')
                _, loss = self.session.run(
                    [self.train_op,self.loss],
                    feed_dict = {self.inputs:batch_x, self.labels:batch_y})
                
            print("Epoch: {0}, Steps: {1} | Loss: {2:.7f}".format(epoch+1, train_steps, loss))
            if self.enable_saver and (epoch+1)%self.save_step==0:
                self.save(epoch)
    
    def predict(self, x):
        return self.session.run(
            self.outputs, 
            feed_dict={
                self.inputs: x, 
                #self.labels: np.ones([self.batch_size,self.label_len,self.output_size])
            })
