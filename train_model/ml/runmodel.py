from get_train_data_min import *
path = "../../dataset/宣城数据整理/110kV永宁变"
cols = ["t1_high_useful","t1_high_useless","t1_middle_useful","t1_middle_useless","t1_low_useful","t1_low_useless","t_T"]
get_train_features(path,cols)
