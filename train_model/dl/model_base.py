import tensorflow as tf


class BaseModel(object):
    def __init__(self, session, env, **options):
        self.session = session
        self.env = env
        self.total_step = 0
        # save graph
        try:
            self.enable_saver = options["enable_saver"]
        except KeyError:
            self.enable_saver = False
        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None
        
        # summary writer
        try:
            self.enable_summary_writer = options['enable_summary_writer']
        except KeyError:
            self.enable_summary_writer = False
        try:
            self.summary_path = options["summary_path"]
        except KeyError:
            self.summary_path = None
        
        # mode and device
        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'
        try:
            self.device = options['device']
        except KeyError:
            self.device = '/gpu:0'
            
        # training parameters
        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001
        try:
            self.train_optimizer = options['optimizer']
        except KeyError:
            self.train_optimizer = tf.train.AdamOptimizer
            
    def restore(self):
        self.saver.restore(self.session, self.save_path)
    def _init_saver(self):
        if self.enable_saver:
            self.saver = tf.train.Saver()
    def _init_summary_writer(self):
        if self.enable_summary_writer:
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.session.graph)
            
    #@abstractmethod
    def _init_input(self, *args):
        pass
    #@abstractmethod
    def _init_nn(self, *args):
        pass
    #@abstractmethod
    def _init_op(self):
        pass
    #@abstractmethod
    def train(self):
        pass
    #@abstractmethod
    def predict(self, a):
        pass
    #@abstractmethod
    def run(self):
        pass

    @staticmethod
    def add_rnn(layer_count, hidden_size, 
                cell=tf.nn.rnn_cell.LSTMCell, activation=tf.tanh, forget_bias=1.0):
        cells = [cell(num_units=hidden_size, forget_bias=forget_bias, activation=activation) 
                 for _ in range(layer_count)]
        return tf.nn.rnn_cell.MultiRNNCell(cells)

    @staticmethod
    def add_cnn(x_input, filters, kernel_size, pooling_size):
        convoluted_tensor = tf.layers.conv2d(x_input, filters, kernel_size, padding='SAME', activation=tf.nn.relu)
        return tf.layers.max_pooling2d(convoluted_tensor, pooling_size, strides=[1, 1], padding='SAME')

    @staticmethod
    def add_fc(x, units, activation=None):
        return tf.layers.dense(inputs=x, units=units, activation=activation)
    

class AdvancedModel(BaseModel):
    def __init__(self, session, env, **options):
        super(AdvancedModel, self).__init__(session, env, **options)
        self.x, self.label, self.y, self.loss = None, None, None, None
        
        try:
            self.train_epochs = options["train_epochs"]
        except KeyError:
            self.train_epochs = 10
        try:
            self.save_step = options["save_step"]
        except KeyError:
            self.save_step = 10
            
    def run(self):
        if self.mode == 'train':
            self.train()
        else:
            self.restore()
            
    def save(self, step):

        self.saver.save(self.session, self.save_path)
        print("Step: {} | Saver reach checkpoint.".format(step + 1))
        
    def eval_and_plot(self):
        pass
