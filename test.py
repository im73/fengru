import os

from train_model.dl.model_run import read_arr

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name =dir_path+"/media/model/hahha/"
if  not os.path.exists(model_name):
    os.makedirs(model_name)


read_arr(path="train_model/df_110kV永宁变1.csv",model_path=model_name,run_model=True,run_pred=False,model_name="lstm")
