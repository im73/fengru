import tensorflow as tf
import numpy as np

from train_model.dl.model_base import AdvancedModel


class Model_BiLSTM(AdvancedModel):
    def __init__(self, session, env, seq_len, label_len,
                 input_size, output_size, batch_size,name, **options):
        super(Model_BiLSTM, self).__init__(session, env, **options)
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
            self.layer_in = tf.layers.dense(units=self.hidden_size, inputs=self.inputs)#, activation=tf.tanh)
        with tf.variable_scope('layer_rnn', initializer=tf.orthogonal_initializer()):
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, forget_bias=1.0, activation=tf.tanh, 
                                             initializer=tf.orthogonal_initializer())
                    for _ in range(self.layer_count)]
            self.rnn_fw = tf.nn.rnn_cell.MultiRNNCell(cells)
            
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, forget_bias=1.0, activation=tf.tanh, 
                                             initializer=tf.orthogonal_initializer())
                    for _ in range(self.layer_count)]
            self.rnn_bw = tf.nn.rnn_cell.MultiRNNCell(cells)
            
            self.birnn_output, self.birnn_states= tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_fw,
                                                                                  cell_bw=self.rnn_bw,
                                                                                  inputs=self.layer_in,
                                                                                  dtype=tf.float32)
            self.birnn_output = tf.concat(self.birnn_output,-1)
            self.birnn_output = self.birnn_output[:,-1]
            #self.birnn_output = tf.reshape(self.birnn_output, [-1, 2*self.seq_len*self.hidden_size])
            print(self.birnn_output)
        
        with tf.variable_scope('layer_out'):
            #self.outputs = tf.layers.dense(units=self.output_size,inputs=self.birnn_output)
            self.outputs = tf.layers.dense(units=self.output_size*self.label_len, inputs=self.birnn_output)
            
    def _init_op(self):
        with tf.variable_scope('loss'):
            self.labels_comp = tf.reshape(self.labels, [-1, self.label_len*self.output_size])
            print(self.labels, self.labels_comp)
            self.loss = tf.losses.mean_squared_error(self.outputs,self.labels_comp)
            
        with tf.variable_scope('train'):
            #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.optimizer = self.train_optimizer(self.learning_rate)
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
                Train_model.objects.get()
    
    def predict(self, x):
        return self.session.run(
            self.outputs, 
            feed_dict={
                self.inputs: x, 
                #self.labels: np.ones([self.batch_size,self.label_len,self.output_size])
            })