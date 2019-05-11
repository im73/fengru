# 对时间做onehot
def get_time_onehot(df_all):
    df_all.month = df_all.month.apply(lambda row:str(row))
    df_all.week_of_month = df_all.week_of_month.apply(lambda row:str(row))
    df_all.weekday = df_all.weekday.apply(lambda row:str(row))
    df_all.day = df_all.day.apply(lambda row:str(row))
    df_all.hour = df_all.hour.apply(lambda row:str(row))
    df_all.minute = df_all.minute.apply(lambda row:str(row))
    df_all.season = df_all.season.apply(lambda row:str(row))
    
    df_all = pd.get_dummies(df_all)
    
    return df_all

# 判断是否是特殊时间 如节假日 重大需求等
def get_time_special(df_all):
    # 工作日
    df_all['ifweekday'] = df_all.weekday.apply(lambda row:1 if 0<=row<=4 else 0)
    
    # 寒假暑假
    df_all['summer_holiday'] = df_all.month.apply(lambda row: 1 if row==7 or row==8 else 0)
    
    # 阳历假期
    
    # 农历假期
    
    return df_all


def get_nan_index(df_all,col):
    #df_all[col].isnull().values==True 只显示存在缺失值的行列，清楚的确定缺失值的位置
    return df_all[col][df_all[col].isnull().values==True].index[-1] #返回有NaN的行索引的最大值

def get_train_data(df_all):
    df_all = df_all[:len(df_all)-24*4*3] #最近三天的数据留出来
    index = get_nan_index(df_all,col='t_Tlastyear_lag11')
    df_all = df_all[index+1:] #去掉有NaN的行
    
    df_ret = df_all.drop(['date','t1_high_useful', 't1_middle_useful', 't1_low_useful', 't_T',
                          'year', 'low_high', 'mid_high',],axis=1)
    df_label = df_all.t_T
    
    return df_ret,df_label
