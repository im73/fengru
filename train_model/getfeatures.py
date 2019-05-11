#data_env self.use_x self.use_y包含高有 高无 中有 中无 低有 低无 low_high mid_high
#lag_flag 为true则加上与时间有关的特征
#no_tcode 为True则没有time_data lag_flag为True才能发挥作用
#将data_env里的self.use_x self.use_y和get_train_data拼在一起??

#'''
from dl.data_env import DataEnv_Ele_M
import numpy as np
import pandas as pd
from ml.model_xgb import Model_XGB

path = "../dataset/df_110kV永宁变1.csv"
result_path = "./result/xgb_result.csv"

seq_len = 4
label_len = 2
target = "t_T"
data_env = DataEnv_Ele_M(path,seq_len,label_len,target)

model_path = "model/xgboost_t.pkl"
options = {"mode":"test","model": "xgb", "train_epochs": 10, "device": "/cpu:0", "enable_saver": True,
                   "save_path": model_path}

model_xgb = Model_XGB(data_env,**options)
model_xgb.run()

pred_x = model_xgb.test_data
#pred_x = model_xgb.test_data[-1].reshape(-1,1) #报错ValueError: feature_names mismatch
#pred_x = data_env.use_x[-1].reshape(-1,1)
pred = model_xgb.predict(pred_x)
#print(pred)
print(pred.shape)
pred = model_xgb.env.inverse_data(pred)
#print(type(pred))

pd.DataFrame(pred[-1*label_len]).to_csv(result_path,header=False,index=False)
#'''

