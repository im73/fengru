import tensorflow as tf
import numpy as np

from train_model.dl.model_base import AdvancedModel
from train_model.dl.tf_helpers import TimeSeriesHelper

class Model_Seq2Seq(AdvancedModel):
    def __init__(self, session, env, seq_len, label_len, input_size, output_size, batch_size, **options):
        super(Model_Seq2Seq, self).__init__(session, env, **options)
        
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
            self.attn_size = options['attn_size']
        except KeyError:
            self.attn_size = self.hidden_size
        try:
            self.layer_count = options['layer_count']
        except KeyError:
            self.layer_count = 1
        
        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
    
    def _init_input(self):
        # inputs [batch_size, seq_len, input_size]
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.seq_len, self.input_size])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.label_len, self.output_size])
    
    def _init_nn(self):
        with tf.variable_scope("layer_in"):
            self.layer_in = tf.layers.dense(inputs=self.inputs, units=self.hidden_size)
            
        with tf.variable_scope("encoder", initializer=tf.orthogonal_initializer()):
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, forget_bias=1.0, activation=tf.tanh,
                                            initializer=tf.orthogonal_initializer())
                    for _ in range(self.layer_count)]
            self.encoder_rnn = tf.nn.rnn_cell.MultiRNNCell(cells)
            self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(
                cell=self.encoder_rnn, 
                inputs=self.layer_in, dtype=tf.float32)
        
        with tf.variable_scope("decoder"):
            # time series helper
            self.helper = TimeSeriesHelper(inputs=self.labels, 
                                           sequence_length=[self.label_len]*self.batch_size)#tf.shape(self.inputs)[0])
            # attention mechanism
            self.attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.attn_size,
                memory=self.encoder_output)
            
            # decoder cell
            with tf.variable_scope("decoder_cell",initializer=tf.orthogonal_initializer()):
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, forget_bias=1.0, activation=tf.tanh,
                                                initializer=tf.orthogonal_initializer())
                        for _ in range(self.layer_count)]
                self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                
                self.decoder_rnn = tf.contrib.seq2seq.AttentionWrapper(
                    cell=self.decoder_cell,
                    attention_mechanism=self.attn_mechanism,
                    #attention_layer_size=self.hidden_size
                )
            
            # use final encoder state to init decoder state
            self.decoder_state_init = self.decoder_rnn.zero_state(batch_size=tf.shape(self.inputs)[0], 
                dtype=tf.float32).clone(cell_state=self.encoder_state)
            # basic decoder
            self.decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_rnn,
                helper=self.helper,
                initial_state=self.decoder_state_init,
                output_layer=tf.layers.Dense(units=self.output_size)
                #output_layer=tf.layers.Dense(units=self.label_len * self.output_size)
            )
            # dynamic decoder
            self.decoder_output, self.decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                self.decoder)
            
            self.outputs = self.decoder_output[0]
            print(self.outputs)
            #self.outputs = tf.layers.dense(units=self.output_size,inputs=self.output,activation=tf.tanh)
        
    def _init_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.outputs,self.labels)
            
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
                self.labels: np.ones([self.batch_size,self.label_len,self.output_size])})
    