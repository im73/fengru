from train_model.dl.model_lstm import Model_LSTM
from train_model.dl.model_seq2seq import Model_Seq2Seq
from train_model.dl.model_bilstm import Model_BiLSTM
from train_model.dl.data_env import DataEnv_Ele_M
import sys

sys.path.append("..")
from train_model.ml.model_xgb import Model_XGB
import tensorflow as tf
import pandas as pd
import numpy as np

def get_xgb_model(data_env,**options):
    model = Model_XGB(data_env,**options)
    return model

def get_model(sess,data_env,seq_len,label_len,input_size,output_size,batch_size,**options):
    if options["model"] == "lstm":
        model = Model_LSTM(sess, data_env, seq_len, label_len, input_size, output_size, batch_size, **options)
    elif options["model"] == "bilstm":
        model = Model_BiLSTM(sess,data_env,seq_len,label_len,input_size,output_size,batch_size,**options)
    elif options["model"] == "seq2seq":
        model = Model_Seq2Seq(sess,data_env,seq_len,label_len,input_size,output_size,batch_size,**options)
    return model

def get_labels(model,outputsize,label_len):
    labels = model.env.test_y[-1*outputsize:].reshape(-1,label_len) #outputsize,96,1 --> outputsize,96
    ret = []
    for i in range(outputsize):
        ret.extend(model.env.inverse_data(labels[i]))
    # print(ret)
    return ret
    '''
    df = pd.DataFrame()
    for i in range(outputsize):
        df[i] = model.env.inverse_data(labels[i])
    df.to_csv("../result/_lstm_labels_48.csv",header=False,index=False)
    '''


def predict(model_name,model):
    if model_name == "xgb":
        pred_x = model.test_data
        ret = model.predict(pred_x)
        ret = model.env.inverse_data(ret)
        return ret[-1*model.env.label_len]
    elif model_name == "seq2seq":
        '''
        pred_x = model.env.test_x
        #print(type(pred_x),pred_x.shape)
        pred_x = pred_x[-1*model.batch_size:]
        #print(pred_x.shape)
        ret = model.predict(pred_x)
        ret = model.env.inverse_data(ret[-1])

        label_x = model.env.use_y[-1*model.batch_size:]
        label_y = model.env.inverse_data(label_x)
        return label_y[-1],ret
        '''
        pred_x = model.env.test_x[-1*model.batch_size:]
        ret = model.predict(pred_x)
        print(pred_x.shape,ret.shape)  # *,96,1
        ret = model.env.inverse_data(ret[-1])  # 1,96
        return ret

    else:
        pred_x = model.env.test_x[-1]  # 96,9
        ret = model.predict([pred_x])  # 1,96,9  1,96
        ret = model.env.inverse_data(ret[0])
        return ret

def read_arr(path,model_path,run_model,run_pred,model_name="lstm"):
    '''

    :param file_array: list 文件路径数组
    :param index: int 输入文件路径在数组中的下标
    :param model_path: str 存储model的路径
    :param run_model: Boolean 如果为True 则训练出来模型并存储在model_path下 否则从model_path中恢复模型
    :param model_name: str 用户选择的模型 默认为lstm
    :return: 将预测结果所在的路径添加到file_array末尾 并将file_array返回
    '''

    #path = file_array[index]
    tf.reset_default_graph()
    seq_len = 1 * 96
    label_len = 96
    target = "t_T"
    input_size = 9
    output_size = 1
    batch_size = 50
    multflag = True
    lagflag = False
    notcode = False

    if model_name == 'xgb':
        seq_len = 1
        lagflag = True

    # model_path = r"../model/%s/%s_%d"%(model_name,model_name,seq_len)
    # result_path = r"../result/%s_result_%d.csv" % (model_name, label_len)

    if run_model==True:
        options = {"mode":"train","model": model_name, "train_epochs": 10, "device": "/cpu:0", "enable_saver": True,
                   "save_path": model_path}
    else:
        options = {"mode":"test","model": model_name, "train_epochs": 10, "device": "/cpu:0", "enable_saver": True,
                   "save_path": model_path}

    data_env = DataEnv_Ele_M(path, seq_len, label_len, target,multflag,lagflag,notcode)

    input_size = data_env.use_x.shape[2]

    if model_name == "xgb":
        model = get_xgb_model(data_env,**options)
        model.run()

    else:
        sess = tf.Session()
        model = get_model(sess, data_env, seq_len, label_len, input_size, output_size, batch_size, **options)
        model.run()

    true_pred = []
    hist = []
    if run_pred:
        # ret = predict(model_name, model)
        # print(type(ret[0][1]))

        ret = predict(model_name,model)

        true_pred = np.hstack(ret).tolist()

        # ret = np.hstack((label_y,ret))

        # df_label = pd.DataFrame(data=label_y[-1])

        # df_result = pd.DataFrame(data=ret)
        # df_result.to_csv(result_path, header=False, index=False)

        t_true = get_labels(model, 1, label_len)
        true_pred.extend(t_true)

        t_hist = model.env.get_history(_len=96)
        hist = t_hist
        # df_label.to_csv("../result/seq2seq_label_96.csv", header=False, index=False)

    # file_array.append(result_path)
    # return file_array
    # return result_path
    return true_pred, hist


if __name__ == "__main__":
    path = "../../dataset/df_110kV永宁变1.csv"

    model_name = "seq2seq" #xgb lstm bilstm seq2seq
    model_path = r"../model/%s/%s_%d"%(model_name,model_name,96)
    #result_path = r"../result/%s_result_%d.csv" % (model_name, 96)
    runmodel = False
    runpred = True

    #file_array = [path]
    #print(file_array)
    #file_array = read_arr(file_array,0,model_path,runmodel,result_path,runpred,model_name)
    true_pred, hist = read_arr(path,model_path,runmodel,runpred,model_name)
    print(true_pred, hist)
    #print(file_array)

'''
seq_len = 1*96
label_len = 1*96
target = "t_T"
input_size = 9
output_size = 1
batch_size = 50
options = {"model":"seq2seq","train_epochs":10,"device":"/cpu:0","enable_saver":True,"save_path":model_path}
sess = tf.Session()
data_env = DataEnv_Ele_M(path,seq_len,label_len,target)
#run(seq_len,label_len,input_size,output_size,batch_size,**options)
'''
