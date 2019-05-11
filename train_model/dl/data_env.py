import sys
sys.path.append("..")
from train_model.transformer_clean import TransformerClean
from train_model.dl.tools import Tools_Time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataEnv_Ele_M(object):
    def __init__(self, path, seq_len, label_len, target, mult_flag=True, lag_flag=False, no_tcode=False):
        self.path = path # 原始数据所在的路径 该路径只对应一共文件
        self.seq_len = seq_len # 历史数据的时间序列的长度
        self.label_len = label_len # 未来数据的时间序列的长度
        self.target = target # 目标列的列名
        self.mult_flag = mult_flag # y是否对应多个数据
        self.lag_flag = lag_flag #
        self.no_tcode = no_tcode

        self.count_t = 0
        self.scaler = StandardScaler()

        self.read_data()
        self.get_features() # 调用Tools_time中的方法获得特征
        self.scaler_data() # 标准化,基于特征矩阵的列,将特征值转换至服从标准正态分布

        #self.prepare_time_code()#获取与时间有关的特征self.use_time_x

        self.prepare_data() # 准备数据集
        self.get_train_test()
        #self.save2npy()
        print(self.use_x.shape,self.use_y.shape)

    def save2npy(self):
        print(self.use_x[:5])
        print(self.df_x.shape,self.df_y.shape,self.x.shape,self.y.shape,self.use_x.shape,self.use_y.shape)
        np.save("../dataset/all_x_t.npy",self.use_x)
        np.save("../dataset/all_y_t.npy",self.use_y)

    def read_data(self):
        self.df_raw = pd.read_csv(self.path) # 原始数据

    def get_features(self):
        self.tools_time = Tools_Time()
        self.df_raw.date = pd.to_datetime(self.df_raw.date)
        self.df_data = self.tools_time.get_features(self.df_raw.copy())  # 归一化后的数据
        self.cols_data = self.df_data.columns[1:10]  # 高有 高无 中有 中无 低有 低无 油温 low_high mid_high

    def get_history(self, _len=96):
        hist = self.df_raw[self.target].tolist()
        hist = hist[-1*(_len+self.label_len):-1*self.label_len]
        # hist = self.inverse_data(hist)
        return hist

    def reset_count(self):
        self.count_t = 0
        self.shuffle_data()

    def scaler_data(self):
        data = self.df_data.drop(['date'],axis=1).values # 以array的形式获取归一化后的数据(除去date)
        scaled_data = self.scaler.fit_transform(data) # 通过StandardScaler对数据进行标准化
        cols = self.df_data.columns
        for i, col in enumerate(cols):
            if i != 0: # 将df_data中除date以外的列修改为标准化处理后的数据
                self.df_data[col] = scaled_data[:,i-1]

        self.scaler.fit(self.df_raw[self.target].values.reshape(-1,1)) # 用于计算训练数据target列的均值和方差

    def inverse_data(self,data):
        return self.scaler.inverse_transform(data) # 将标准化后的数据转换为原始数据

    def shuffle_data(self): # self.train_x self.train_y不放回的抽样 进行重新洗牌
        index = np.random.choice(len(self.train_x),len(self.train_x),replace=False) # 返回洗牌后的下标列表
        self.train_x = self.train_x[index]
        self.train_y = self.train_y[index]

    def prepare_time_code(self):
        '''
        df_all = self.df_data[:len(self.df_data)-self.label_len] #

        index = get_nan_index(df_all,col='t_Tlastyear_lag11')
        df_all = df_all[index+1:] #去掉有NaN的行

        df_ret = df_all.drop(['date','t1_high_useful', 't1_middle_useful', 't1_low_useful', 't_T',
                          'year', 'low_high', 'mid_high',],axis=1)
        df_label = df_all.t_T

        return df_ret,df_label
        '''
        #只用date列去获取时间有关的特征 最后把date列扔掉获得df_time_features
        #prepare_data中截取含有NaN的行 再与use_x use_y拼接在一起
        df_time = self.tools_time.get_time_feature(self.df_data.date.to_frame())
        #print(df_time[:5])
        df_time = self.tools_time.get_time_special(df_time)
        #print(df_time[:5])
        #df_time = self.tools_time.get_time_onehot(df_time)
        df_time_features = df_time.drop(['date'],axis=1)

        '''
        #排除NaN和reshape
        index = self.seq_len-1
        if self.lag_flag:
            index = 365*24*4
    
        df_time_x = df_time_features.values[index:].reshape(-1,1,len(df_time_features.columns)) #将数据切分成三维
        #df_time_x = df_time_features.values[index:].reshape(-1,len(df_time_features.columns))) #基础模型要求二维数据
        
        # 排除掉未来值的nan
        self.use_time_x = df_time_x[:len(df_time_x)-self.label_len]
        '''

        past_cols = df_time_features.columns
        #print(past_cols)
        #print(len(past_cols)) #95
        for i in range(self.seq_len-1,0-1,-1):
            for col in past_cols:
                df_time_features[col+'_past_'+str(i)] = df_time_features[col].shift(i)
        curr_cols = df_time_features.columns
        #print(curr_cols)
        #print(len(curr_cols)) #475
        x_cols = curr_cols[len(past_cols):(1+self.seq_len)*len(past_cols)]
        #print(x_cols,len(x_cols)) #380
        df_time_data = df_time_features[x_cols]
        #print(len(df_time_data)) #105252

        #排除NaN和reshape
        index = self.seq_len-1
        if self.lag_flag:
            index = 365*24*4

        df_time_x = df_time_data.values[index:].reshape(-1,self.seq_len,len(past_cols)) #将数据切分成三维
        #df_time_x = df_time_data.values[index:].reshape(-1,self.seq_len*len(past_cols)) #基础模型要求二维数据

        # 排除掉未来值的nan
        self.use_time_x = df_time_x[:len(df_time_x)-self.label_len]


    def prepare_data(self):
        # 获取过去和未来的值 即x和y
        past_cols = self.df_data.columns #
        cols = self.df_data.columns[1:10] # 高有 高无 中有 中无 低有 低无 油温 low_high mid_high
        for i in range(self.seq_len-1,0-1,-1): # 获得seq_len*9个历史数据
            for col in cols:
                self.df_data[col+'_past_'+str(i)] = self.df_data[col].shift(i)
        for i in range(1,self.label_len+1): # 获得label_len个未来数据
            self.df_data[self.target+'_next_'+str(i)] = self.df_data[self.target].shift(-1*i)

        curr_cols = self.df_data.columns # 增加历史数据和未来数据后的cols
        x_cols = curr_cols[len(past_cols):len(past_cols)+9*self.seq_len] # next add time onehot
        y_cols = curr_cols[len(curr_cols)-1*self.label_len:] # y对应的列
        self.df_x = self.df_data[x_cols]
        if self.mult_flag:
            self.df_y = self.df_data[y_cols]
        else:
            self.df_y = self.df_data[y_cols[-1]] # 只取label_len中最后一个未来数据

        # 排除掉过去值的nan
        index = self.seq_len-1
        if self.lag_flag:
            index = 365*24*4

        self.x = self.df_x.values[index:].reshape(-1,self.seq_len,9) #将数据切分成三维
        #self.x = self.df_x.values[index:].reshape(-1,self.seq_len*9) #基础模型要求二维数据
        if self.mult_flag:
            self.y = self.df_y.values[index:].reshape(-1,self.label_len,1)
            #self.y = self.df_y.values[index:].reshape(-1,self.label_len*1)
        else:
            self.y = self.df_y.values[index:].reshape(-1,1,1)
            #self.y = self.df_y.values[index:].reshape(-1,1*1)

        # 排除掉未来值的nan
        self.use_x = self.x[:len(self.x)-self.label_len]
        self.use_y = self.y[:len(self.y)-self.label_len]

        #self.use_x = np.concatenate([self.use_x,self.use_time_x],axis=-1)

        if self.lag_flag:
            self.get_static_data()
            print(self.total_data.shape, self.use_x.shape)
            self.use_x = np.concatenate([self.use_x, self.total_data],axis=-1)

    def get_static_data(self):
        # 获取前一年 前一个月 前一周的cols_data(高中低+low_high mid_high)中的数据
        self.lag_data = self.tools_time.get_lag_value(self.df_data, self.cols_data).values
        # 获得和时间有关的特征(不包含具体date和year) 并转换成独热码返回
        self.time_data = self.tools_time.get_time_code(self.df_data).values

        index = 365*24*4
        self.lag_data = self.lag_data[index:len(self.lag_data)-self.label_len]
        self.time_data = self.time_data[index:len(self.time_data)-self.label_len]

        if self.no_tcode: #不需要time code的flag
            self.lt_data = self.lag_data
        else: #lag_data time_data拼接
            self.lt_data = np.concatenate([self.lag_data,self.time_data],axis=-1)

        n_features = self.lt_data.shape[1]
        self.total_data = None
        for i in range(self.seq_len):
            if self.total_data is None:
                self.total_data = self.lt_data
            else:
                self.total_data = np.concatenate([self.total_data, self.lt_data],axis=-1)
        self.total_data = self.total_data.reshape(-1,self.seq_len,n_features)
        self.lag_data = self.time_data = self.lt_data = None

    def get_train_test(self): # 以70%的比例划分训练集和测试集
        border = int(len(self.use_x)*0.7)+1
        self.train_x = self.use_x[:border]
        self.train_y = self.use_y[:border]
        self.test_x = self.use_x[border:]
        self.test_y = self.use_y[border:]

    def get_steps(self,batch_size,mode='train'):
        if mode=='train':
            return int(len(self.train_x)/batch_size)+1
        elif mode=='test':
            return int(len(self.test_x)/batch_size)+1
            #return len(self.test_x)
    def get_datasize(self):
        return self.train_x.shape[-1],self.train_y.shape[-1]
    def get_label_len(self):
        return self.label_len if self.mult_flag else 1

    def get_batch_data(self,batch_size,mode='train'):
        if mode=='train':
            data_x = self.train_x
            data_y = self.train_y
        elif mode=='test':
            data_x = self.test_x
            data_y = self.test_y

        if (self.count_t+1)*batch_size > len(data_x):
            begin = len(data_x)-batch_size
            end = len(data_x)
            self.count_t = 0
        else:
            begin = self.count_t*batch_size
            end = (self.count_t+1)*batch_size
            self.count_t += 1
        seq = data_x[begin:end]
        res = data_y[begin:end]

        return seq,res



class DataEnv_Ele(object):
    def __init__(self, path, seq_len, label_len, target, mult_flag=True):
        self.path = path
        self.seq_len = seq_len
        self.label_len = label_len
        self.target = target
        self.mult_flag = mult_flag

        self.count_t = 0
        self.scaler = StandardScaler()

        self.read_data()
        self.get_features()
        self.scaler_data()
        self.prepare_data()
        self.get_train_test()

    def read_data(self):
        self.df_raw = pd.read_csv(self.path)

    def get_features(self):
        tools_time = Tools_Time()
        self.df_raw.date = pd.to_datetime(self.df_raw.date)
        self.df_data = tools_time.get_features(self.df_raw.copy())

    def reset_count(self):
        self.count_t = 0
        self.shuffle_data()

    def scaler_data(self):
        data = self.df_data.drop(['date'],axis=1).values
        scaled_data = self.scaler.fit_transform(data)
        cols = self.df_data.columns
        for i,col in enumerate(cols):
            if i!=0:
                self.df_data[col] = scaled_data[:,i-1]

        self.scaler.fit(self.df_raw[self.target].values.reshape(-1,1))

    def inverse_data(self,data):
        return self.scaler.inverse_transform(data)

    def shuffle_data(self):
        index = np.random.choice(len(self.train_x),len(self.train_x),replace=False)
        self.train_x = self.train_x[index]
        self.train_y = self.train_y[index]

    def prepare_data(self):
        past_cols = self.df_data.columns
        cols = self.df_data.columns[1:10]
        for i in range(self.seq_len-1,0-1,-1):
            for col in cols:
                self.df_data[col+'_past_'+str(i)] = self.df_data[col].shift(i)
        for i in range(1,self.label_len+1):
            self.df_data[self.target+'_next_'+str(i)] = self.df_data[self.target].shift(-1*i)

        curr_cols = self.df_data.columns
        x_cols = curr_cols[len(past_cols):len(past_cols)+9*self.seq_len] # next add time onehot
        y_cols = curr_cols[len(curr_cols)-1*self.label_len:]
        self.df_x = self.df_data[x_cols]
        if self.mult_flag:
            self.df_y = self.df_data[y_cols]
        else:
            self.df_y = self.df_data[y_cols[-1]]

        #index = self.df_x[self.df_x.isnull().values==True].index[-1]
        index = self.seq_len-1
        self.x = self.df_x.values[index:].reshape(-1,self.seq_len,9)
        if self.mult_flag:
            self.y = self.df_y.values[index:].reshape(-1,self.label_len,1)
        else:
            self.y = self.df_y.values[index:].reshape(-1,1,1)

        self.use_x = self.x[:len(self.x)-self.label_len]
        self.use_y = self.y[:len(self.y)-self.label_len]

    def get_train_test(self):
        border = int(len(self.use_x)*0.7)+1
        self.train_x = self.use_x[:border]
        self.train_y = self.use_y[:border]
        self.test_x = self.use_x[border:]
        self.test_y = self.use_y[border:]

    def get_steps(self,batch_size,mode='train'):
        if mode=='train':
            return int(len(self.train_x)/batch_size)+1
        elif mode=='test':
            return int(len(self.test_x)/batch_size)+1
            #return len(self.test_x)
    def get_datasize(self):
        return self.train_x.shape[-1],self.train_y.shape[-1]
    def get_label_len(self):
        return self.label_len if self.mult_flag else 1

    def get_batch_data(self,batch_size,mode='train'):
        if mode=='train':
            data_x = self.train_x
            data_y = self.train_y
        elif mode=='test':
            data_x = self.test_x
            data_y = self.test_y

        if (self.count_t+1)*batch_size > len(data_x):
            begin = len(data_x)-batch_size
            end = len(data_x)
            self.count_t = 0
        else:
            begin = self.count_t*batch_size
            end = (self.count_t+1)*batch_size
            self.count_t += 1
        seq = data_x[begin:end]
        res = data_y[begin:end]

        return seq,res

