import numpy as np
import pandas as pd
import datetime

class Tools_Time(object):
    def __init__(self):
        pass
    
    def get_features(self, df_all):
        df_all = self.get_rate_feature(df_all)
        #df_all = self.get_time_feature(df_all)
        #df_all = self.get_time_special(df_all)
        #df_all = self.get_time_onehot(df_all)
        
        return df_all
    
    def get_time_code(self, df_all): # 获得和时间有关的特征(不包含具体date和year) 并转换成独热码返回
        df_tmp = pd.DataFrame()
        df_tmp['date'] = df_all.date # 单独提取出date列 调用相关函数获得与时间有关的特征并转换成独热码
        df_tmp = self.get_time_feature(df_tmp)
        df_tmp = self.get_time_special(df_tmp)
        df_tmp = self.get_time_onehot(df_tmp)
        return df_tmp.drop(['date','year'],axis=1) # 从中删除时间和年份列
    
    def get_lag_value(self, df_all, cols): # 获取前一年 前一个月 前一周的数据
        df_tmp = pd.DataFrame()
        for col in cols:
            df_tmp[col+'_lastyear'] = df_all[col].shift(365*24*4)
            df_tmp[col+'_lastmonth'] = df_all[col].shift(30*24*4)
            df_tmp[col+'_lastweek'] = df_all[col].shift(7*24*4)
        return df_tmp
    
    def get_time_feature(self, df_all):
        def get_week_of_month(row):
            end = int(datetime.datetime(row.year, row.month, row.day).strftime("%W"))
            begin = int(datetime.datetime(row.year, row.month, 1).strftime("%W"))
            return end - begin + 1 # 计算月初到当前时间一共几周(向上取整)
        def get_season_of_year(row):
            season_map = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
            return season_map[row.month]
        # 将date转换成年月日等特征
        # apply函数将函数应用到由date列形成的一维数组上 获得一组时间有关的特征
        df_all['year'] = df_all.date.apply(lambda x: x.year)
        df_all['month'] = df_all.date.apply(lambda x: x.month)
        df_all['day'] = df_all.date.apply(lambda x: x.day)
        df_all['hour'] = df_all.date.apply(lambda x: x.hour)
        df_all['minute'] = df_all.date.apply(lambda x: x.minute)
        df_all['weekday'] = df_all.date.apply(lambda x: x.weekday())
        df_all['week_of_month'] = df_all.date.apply(get_week_of_month)
        df_all['season'] = df_all.date.apply(get_season_of_year)

        return df_all

    def get_rate_feature(self, df_all): # 将中有低有归一化 结果存在新增的列'low_high'&'mid_high'中
        def get_rate(row,col):
            if row.t1_high_useful!=0:
                return row[col]/row.t1_high_useful
            else:
                return row[col]/(row.t1_low_useful+row.t1_middle_useful+0.001)
        df_all['low_high'] = df_all.apply(lambda row:get_rate(row,'t1_low_useful'),axis=1)
        df_all['mid_high'] = df_all.apply(lambda row:get_rate(row,'t1_middle_useful'),axis=1)
        return df_all
    
    # 对时间做onehot
    def get_time_onehot(self, df_all): # 调用完get_time_feature获得month等列之后再调用 将数值转换成str
        df_all.month = df_all.month.apply(lambda row:str(row))
        df_all.week_of_month = df_all.week_of_month.apply(lambda row:str(row))
        df_all.weekday = df_all.weekday.apply(lambda row:str(row))
        df_all.day = df_all.day.apply(lambda row:str(row))
        df_all.hour = df_all.hour.apply(lambda row:str(row))
        df_all.minute = df_all.minute.apply(lambda row:str(row))
        df_all.season = df_all.season.apply(lambda row:str(row))

        df_all = pd.get_dummies(df_all) # 对df_all进行独热编码

        return df_all

    # 判断是否是特殊时间 如节假日 重大需求等
    def get_time_special(self, df_all):
        # 工作日
        df_all['ifweekday'] = df_all.weekday.apply(lambda row:1 if 0<=row<=4 else 0)

        # 寒假暑假
        df_all['summer_holiday'] = df_all.month.apply(lambda row: 1 if row==7 or row==8 else 0)
        #df_all['winter_holiday'] = df_all.month.apply(lambda row: 1 if row==2 or else 0)

        # 阳历假期
        df_all['national_day'] = df_all.date.apply(lambda row: 1 if row.month==10 and 1<=row.day<=7 else 0)
        df_all['may_day'] =  df_all.date.apply(lambda row: 1 if row.month==5 and row.day==1 else 0)
        df_all['newyear_day'] =  df_all.date.apply(lambda row: 1 if row.month==1 and row.day==1 else 0)
        df_all['christmas_day'] =  df_all.date.apply(lambda row: 1 if row.month==12 and 24<=row.day<=25 else 0)

        # 农历假期
        return df_all