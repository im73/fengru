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

data_all = np.load('../../dataset/all_x_t.npy')
label_all = np.load('../../dataset/all_y_t.npy')

cv_params = {'estimator__n_estimators': [500, 800, 1000, 1600, 2400], 'estimator__max_depth': [3,6,8,10]}#
other_params = {'learning_rate': 0.01, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
#other_params = {'learning_rate': 0.01, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
#                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
#other_params = {'estimator__learning_rate': 0.01, 'estimator__n_estimators': 500, 'estimator__max_depth': 5, 'estimator__min_child_weight': 1, 'estimator__seed': 0,
#                    'estimator__subsample': 0.8, 'estimator__colsample_bytree': 0.8, 'estimator__gamma': 0, 'estimator__reg_alpha': 0, 'estimator__reg_lambda': 1}

model = xgb.XGBRegressor(**other_params)
model = MultiOutputRegressor(model)
best_model = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=2)

border = int(len(data_all)*0.9)+1
train_data = data_all[:border]
train_label = label_all[:border]

print(np.array(train_data).shape)
print(np.array(train_label).shape)

test_data = data_all[border:]
test_label = label_all[border:]

best_model.fit(train_data, train_label)

#'''
print(best_model.best_estimator_.get_params())
b_model = best_model.best_estimator_
joblib.dump(b_model,'./model/xgboost_t.pkl')

#'''
#joblib.dump(best_model,'../model/xgboost_t.pkl')
print('xgboost_1 model saved')

pred = b_model.predict(test_data)
#pred = best_model.predict(test_data)
mse = mean_squared_error(pred,test_label)
mae = mean_absolute_error(pred,test_label)
print(mse,mae)
