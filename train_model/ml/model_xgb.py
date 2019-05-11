import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor

from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

class Model_XGB(object):
    def __init__(self,env,**options):
        #self.data_all = np.load('../dataset/all_x_t.npy')
        #self.label_all = np.load('../dataset/all_y_t.npy')
        self.env = env
        self.x_shape = self.env.use_x.shape
        self.y_shape = self.env.use_y.shape
        self.data_all = self.env.use_x.reshape(-1,self.x_shape[1]*self.x_shape[2])
        self.label_all = self.env.use_y.reshape(-1,self.y_shape[1]*self.y_shape[2])
        print(self.data_all.shape,self.label_all.shape)
        print(type(self.data_all),type(self.label_all))

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
            #self.learning_rate = 0.001
            self.learning_rate = 0.01

        '''
        try:
            self.train_optimizer = options['optimizer']
        except KeyError:
            self.train_optimizer = tf.train.AdamOptimizer
        '''

        self._init_gbd()
        self._init_input()

    def _init_gbd(self):
        cv_params = {'estimator__n_estimators': [500, 800, 1000, 1600, 2400], 'estimator__max_depth': [3, 6, 8, 10]}  #
        other_params = {'learning_rate': self.learning_rate, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
        self.model = xgb.XGBRegressor(**other_params)
        self.model = MultiOutputRegressor(self.model)
        self.best_model = GridSearchCV(estimator=self.model, param_grid=cv_params, scoring='r2', cv=5, verbose=2)

    def _init_input(self):
        border = int(len(self.data_all) * 0.9) + 1
        self.train_data = self.data_all[:border]
        self.train_label = self.label_all[:border]

        #print(np.array(self.train_data).shape)
        #print(np.array(self.train_label).shape)

        self.test_data = self.data_all[border:]
        self.test_label = self.label_all[border:]

    def _init_summary_writer(self):
        pass

    def train(self):
        self.best_model.fit(self.train_data, self.train_label)

        #'''
        print(self.best_model.best_estimator_.get_params())
        self.b_model = self.best_model.best_estimator_
        #'''

        #self.b_model = self.best_model

        # joblib.dump(self.b_model,'./model/xgboost_t.pkl')
        if self.enable_saver:
            joblib.dump(self.b_model, self.save_path)
        print('xgboost_1 model saved')

    def restore(self):
        self.b_model = joblib.load(self.save_path)

    def predict(self,x):
        pred = self.b_model.predict(x)
        # mse = mean_squared_error(pred, test_label)
        # mae = mean_absolute_error(pred, test_label)
        # print(mse, mae)
        return pred

    def run(self):
        if self.mode == 'train':
            self.train()
        else:
            self.restore()



