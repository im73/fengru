from transformer_clean import TransformerClean

import pandas as pd
import numpy as np

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

df_all,name = get_df("./liuan/110kV永宁变")

#print(name)

for i,df in enumerate(df_all):
    df.to_csv('./save_csv/df_110kV永宁变'+str(i)+'.csv',index=False)
