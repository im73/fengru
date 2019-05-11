import pandas as pd
import numpy as np

import datetime
import time

from multiprocessing import Process,Array
import multiprocessing as mp

import sys
sys.path.append("..")
from transformer_clean import TransformerClean

import warnings
warnings.filterwarnings('ignore')

def get_df(path): # 返回数据的DataFrame和文件夹或文件名
    tc = TransformerClean(path)
    nums = tc.tf_num
    df_all = []
    for num in range(nums):
        data = [item for item in tc.pop(num)]
        df = pd.DataFrame(columns=['date','t1_high_useful','t1_high_useless',
                                   't1_middle_useful','t1_middle_useless', 't1_low_useful', 't1_low_useless','t_T'])
        data = np.array(data); t=0
        for col in df.columns:
            if t!=0:
                df[col] = data[:,t].astype(np.float32)
            else:
                df[col] = data[:,t]
            t = t+1
        df_all.append(df)
    return df_all, tc.name

def get_forecast(df_all):
    date_start = df_all.date[len(df_all)-1] # 数据的最后一天为起始时间
    date_end = date_start+datetime.timedelta(3) # 三天后
    date_range = pd.date_range(start=date_start, end=date_end, freq='15min')
    df_pred = pd.DataFrame(columns=df_all.columns)
    df_pred.date = date_range # 设置预测数据的时间戳
    
    return df_pred

def get_full_df(df_all):
    df_all.date = pd.to_datetime(df_all.date, format='%Y-%m-%d %H:%M:%S')
    df_pred = get_forecast(df_all)
    df_all = pd.concat([df_all, df_pred[1:]]) # 将原始数据的时间戳和需要预测的时间戳合并
    df_all.index = np.arange(len(df_all)) # 行名设为数字??
    
    return df_all
    
def get_time_feature(df_all):
    def get_week_of_month(row):
        end = int(datetime.datetime(row.year, row.month, row.day).strftime("%W"))
        begin = int(datetime.datetime(row.year, row.month, 1).strftime("%W"))
        return end - begin + 1 # 计算月初到当前时间一共几周(向上取整)
    def get_season_of_year(row):
        season_map = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
        return season_map[row.month]
    # 将date转换成年月日等特征
    # apply函数将函数应用到由各列或行形成的一维数组上 默认为列
    df_all['year'] = df_all.date.apply(lambda x: x.year)
    df_all['month'] = df_all.date.apply(lambda x: x.month)
    df_all['day'] = df_all.date.apply(lambda x: x.day)
    df_all['hour'] = df_all.date.apply(lambda x: x.hour)
    df_all['minute'] = df_all.date.apply(lambda x: x.minute)
    df_all['weekday'] = df_all.date.apply(lambda x: x.weekday())
    df_all['week_of_month'] = df_all.date.apply(get_week_of_month)
    df_all['season'] = df_all.date.apply(get_season_of_year)
    
    return df_all

def get_rate_feature(df_all):
    def get_rate(row,col): # 将row[col]归一化
        if row.t1_high_useful!=0:
            return row[col]/row.t1_high_useful
        else:
            return row[col]/(row.t1_low_useful+row.t1_middle_useful+0.001)
    df_all['low_high'] = df_all.apply(lambda row:get_rate(row,'t1_low_useful'),axis=1)
    df_all['mid_high'] = df_all.apply(lambda row:get_rate(row,'t1_middle_useful'),axis=1)
    return df_all

# start 的前k个时刻的值 差值 统计值
def get_statistic(df_all, cols, start, k, step, flag, record, stats):
    for col in cols:
        tmp = pd.DataFrame(); # 记录start的前k个时刻的特征
        for i in range(start, start+k, step):
            tmp['lag'+str(i)] = df_all[col].shift(i) # 'lagi'为df_all[col]向下平移i行 对应start-i时刻的特征
            if record:
                df_all[col+flag+'_lag'+str(i-start)] = tmp['lag'+str(i)] # lag值
                if i>start: # start时刻不记录
                    df_all[col+flag+'_sub'+str(i-start)] = tmp['lag'+str(i)]-tmp['lag'+str(i-1)] # 求差值
        tmp = tmp.T # 将tmp转置 行索引变成lag 列索引变成特征
        
        if stats: # 按列计算统计值 shift产生的NaN咋办??
            df_all[col+flag+'_lag'+str(k)+'sum'] = tmp.sum()
            df_all[col+flag+'_lag'+str(k)+'avg'] = tmp.mean()
            df_all[col+flag+'_lag'+str(k)+'skew'] = tmp.skew()
            df_all[col+flag+'_lag'+str(k)+'kurt'] = tmp.kurt()
            df_all[col+flag+'_lag'+str(k)+'var'] = tmp.var()
            df_all[col+flag+'_lag'+str(k)+'std'] = tmp.std()
            df_all[col+flag+'_lag'+str(k)+'max'] = tmp.max()
            df_all[col+flag+'_lag'+str(k)+'min'] = tmp.min()
            df_all[col+flag+'_lag'+str(k)+'median'] = tmp.median()
            df_all[col+flag+'_lag'+str(k)+'quantile30'] = tmp.quantile(0.3)
            df_all[col+flag+'_lag'+str(k)+'quantile80'] = tmp.quantile(0.8)
    return df_all

def get_statistic_min(df_all, cols, start, k, step, flag='min', record=False):
    for col in cols:
        tmp = pd.DataFrame();
        for i in range(start, start+k, step):
            tmp['lag'+str(i)] = df_all[col].shift(i)
            if record:
                df_all[col+flag+'_lag'+str(i-start)] = tmp['lag'+str(i)] # lag值
                if i>start:
                    df_all[col+flag+'_sub'+str(i-start)] = tmp['lag'+str(i)]-tmp['lag'+str(i-1)] # 求差值
        tmp = tmp.T
        #df_all[col+flag+'_stats'+str(k)+'sum'] = tmp.sum()
        df_all[col+flag+'_stats'+str(k)+'avg'] = tmp.mean() # 只算平均值
    
    return df_all

def get_statistic_cycle(df_all, cols, num, cycle, flag='cycle', record=False):# 前num个周期(cycle)的统计数据
    for col in cols:
        tmp = pd.DataFrame();
        for i in range(1,num+1,1):
            tmp['lag'+str(i)] = df_all[col].shift(i*cycle)
            if record:
                df_all[col+flag+'_cycle'+str(i-start)] = tmp['lag'+str(i)] # lag值
        tmp = tmp.T
        #df_all[col+flag+'_stats'+str(num)+'sum'] = tmp.sum()
        df_all[col+flag+'_cycle'+str(num)+'avg'] = tmp.mean()
    
    return df_all

def get_lag_feature(df_all, cols): # 获得前一年/前一月/前一周的数据
    for col in cols:
        df_all[col+'_lastyear_val'] = df_all[col].shift(365*24*4)
        df_all[col+'_lastmonth_val'] = df_all[col].shift(30*24*4)
        df_all[col+'_lastweek_val'] = df_all[col].shift(7*24*4)
    return df_all

# def get_all_features_thread(df_all, cols, start, queue, name):    
#     s = time.time()
#     # 获取 从 start 时刻开始 过去k个时刻的值
#     df_all = get_statistic(df_all, cols, start=start, k=4*3, step=1, flag='thisyear', record=True, stats=False) # 3个小时 包括值和差值
#     df_all = get_statistic(df_all, cols, start=365*24*4, k=4*3, step=1, flag='lastyear', record=True, stats=False) # 前一年的3个小时
#     # 获取 从 start 时刻开始 过去k个时刻的统计值
#     df_all = get_statistic(df_all, cols, start=start, k=4*12, step=1, flag='thisyear2', record=False, stats=True)
#     df_all = get_statistic(df_all, cols, start=365*24*4, k=4*12, step=1, flag='lastyear2', record=False, stats=True)
#     # 24*2+11*2 = 70
    
#     e = time.time(); print(cols, e-s)
    
#     df_all = get_statistic(df_all, cols, start=start, k=24*4, step=1, flag='day', record=False, stats=True) # 过去一天的统计值
#     df_all = get_statistic(df_all, cols, start=start, k=7*24*4, step=1, flag='week', record=False, stats=True) # 过去一周的统计值
#     # 70+11*2 = 92
    
#     s = time.time(); print(cols, s-e)
    
#     # 获取前一年 前一月 前一周相同时刻的值 以及周围几个时刻的平均值
#     df_all = get_lag_feature(df_all, cols)
#     df_all = get_statistic_min(df_all, cols, start=365*24*4-3, k=6, step=1, flag='year_s') # 年
#     df_all = get_statistic_min(df_all, cols, start=30*24*4-3, k=6, step=1, flag='month_s') # 月
#     df_all = get_statistic_min(df_all, cols, start=7*24*4-3, k=6, step=1, flag='week_s') # 周
#     # 92+4 = 96
    
#     e = time.time(); print(cols, e-s)
    
#     # 获取过去几个月和几周的同一时刻的值的平均值
#     df_all = get_statistic_cycle(df_all, cols, num=3, cycle=30*24*4, flag='month_c') # 过去3个月
#     df_all = get_statistic_cycle(df_all, cols, num=3, cycle=7*24*4, flag='week_c') # 过去3周
#     # 96+2 = 98
    
#     s = time.time(); print(cols, s-e)
    
#     df_save = df_all[df_all.columns[18:]]
#     path = './dataset/save/'+name+cols[0]+'.csv'
#     df_save.to_csv(path,index=False)
    
#     queue.put((cols[0],path))
    
#     return df_all

def get_all_features_thread(df_all, cols, start, queue, name):    
    s = time.time()
    # 获取 从 start 时刻开始 过去k个时刻的值
    if cols[0]=='t_T':
        df_all = get_statistic(df_all, cols, start=start, k=4*3, step=1, flag='thisyear', record=True, stats=False) # 3个小时 包括值和差值
        df_all = get_statistic(df_all, cols, start=365*24*4, k=4*3, step=1, flag='lastyear', record=True, stats=False) # 前一年的3个小时
        # 获取 从 start 时刻开始 过去k个时刻的统计值
        df_all = get_statistic(df_all, cols, start=start, k=4*12, step=1, flag='thisyear2', record=False, stats=True)
        df_all = get_statistic(df_all, cols, start=365*24*4, k=4*12, step=1, flag='lastyear2', record=False, stats=True)
        # 24*2+11*2 = 70
        
        df_all = get_statistic(df_all, cols, start=start, k=24*4, step=1, flag='day', record=False, stats=True) # 过去一天的统计值
        df_all = get_statistic(df_all, cols, start=start, k=7*24*4, step=1, flag='week', record=False, stats=True) # 过去一周的统计值
        # 70+11*2 = 92
    else:
        df_all = get_statistic(df_all, cols, start=start, k=4, step=1, flag='1hour', record=True, stats=False) # start 前的1个小时的值和差值
        # 8
    
    e = time.time(); print(cols, e-s)
    
    s = time.time(); print(cols, s-e)
    
    # 获取前一年 前一月 前一周相同时刻的值 以及周围几个时刻的平均值
    df_all = get_lag_feature(df_all, cols)
    df_all = get_statistic_min(df_all, cols, start=365*24*4-3, k=6, step=1, flag='year_s') # 年
    df_all = get_statistic_min(df_all, cols, start=30*24*4-3, k=6, step=1, flag='month_s') # 月
    df_all = get_statistic_min(df_all, cols, start=7*24*4-3, k=6, step=1, flag='week_s') # 周
    # 92+6 = 98; 8+6 = 14
    
    e = time.time(); print(cols, e-s)
    
    # 获取过去几个月和几周的同一时刻的值的平均值
    df_all = get_statistic_cycle(df_all, cols, num=3, cycle=30*24*4, flag='month_c') # 过去3个月
    df_all = get_statistic_cycle(df_all, cols, num=3, cycle=7*24*4, flag='week_c') # 过去3周
    # 98+2 = 100; 14+2 = 16
    
    s = time.time(); print(cols, s-e)
    
    df_save = df_all[df_all.columns[18:]]
    path = './dataset/save/'+name+cols[0]+'.csv'
    df_save.to_csv(path,index=False)
    
    queue.put((cols[0],path))
    
    return df_all
    
    
def get_all_features(df_all, cols, start, name):
    # 补充预测时刻
    df_all = get_full_df(df_all)
    
    # 补充时间特征和其他特征
    df_all = get_time_feature(df_all)
    df_all = get_rate_feature(df_all) 

    print(name)
    print('get feature start'+name)
    
    df_path = {}
    processes = []
    queue = mp.Queue()
    for col in cols:
        processes += [Process(target=get_all_features_thread, args=[df_all,[col],start,queue,name])]
    [process_one.start() for process_one in processes]
    for i in range(len(cols)):
        col,path = queue.get()
        df_path[col] = path
    [process_one.join() for process_one in processes]
    
    for col in cols:
        df_tmp = pd.read_csv(df_path[col])
        df_all = pd.concat([df_all, df_tmp],axis=1)
    
    #df_all = get_time_onehot(df_all)
    
    df_all.to_csv('../../dataset/save/'+name+'_all.csv',index=False)
    
    return df_all
    
def get_train_features(path,cols):
    df_all,name = get_df(path)
    
    for i,df in enumerate(df_all):
        get_all_features(df,cols,24*4*3,name+str(i))   







