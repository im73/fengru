#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 18-11-28
# @Author  : haoyi

from train_model.transformer_base import Transformer
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

MAX_DIM = 8  # 时间，高有，高无，中有，中无，低有，低无，温度集合

# DEBUG = True
SAVE = False
DEBUG = False

#Transformer类的子类
class TransformerClean(Transformer):
    def __init__(self, folder_path):
        super(TransformerClean, self).__init__(folder_path)
        self.data = self.__clean__()

    def __clean__(self):
        def check_time_det(time_str_in):
            minute = time_str_in.split(':')[1]
            minute_num = int(minute) if minute.isdigit() else -1
            if minute_num == 0 or minute_num == 15 or minute_num == 30 or minute_num == 45:
                return True
            return False

        def truncate_end(time_in, data_in, te_in):
            count_num = len(time_in) # 数据行数
            for idx in range(len(time_in) - 1, 0, -1): # 从后往前遍历
                # 如果最后一行高有+高无+中有+中无+低有+低无 == 0(总功率是0??) 则总数据行数--
                if np.sum([i for i in data_in[idx, :]]) == 0.0:
                    count_num -= 1
                else: # 遇到符合条件的行 就直接break
                    break
            #直接将数据缩短为count_num行
            return time_in[:count_num], data_in[:count_num, :], te_in[:count_num, :], count_num

        def del_outlier(array_in): # 删除异常值
            # processed by each dim
            dim_num = array_in.shape[1]
            array_out = np.zeros(array_in.shape)
            for idx_dim in range(dim_num):
                feature = array_in[:, idx_dim] # 单独将某一列(idx_dim)提取出来转换成行向量
                mean = np.mean(feature) # 计算该特征的平均值和标准差
                std = np.std(feature)
                # print(idx_dim, ':', mean, std)
                # 该特征中出现分布在mean - 5*std ~ mean + 5*std之外的值 --> 小概率事件 则将该值变为0
                feature[(feature > (mean + 5 * std)) + (feature < (mean - 5 * std))] = 0.0
                array_out[:, idx_dim] = feature
            return array_out

        def fill_zero(array_in):
            dim_num = array_in.shape[1]
            mask = [True for i in range(dim_num)]
            mean = np.mean(array_in, axis=0) # axis=0 --> 按列求平均值 并且返回一个行向量
            std = np.std(array_in, axis=0)
            for idx_dim in range(dim_num): # 如果某一列的均值和方差均为0 则mask设为False
                if mean[idx_dim] == 0.0 and std[idx_dim] == 0.0:
                    mask[idx_dim] = False
            for idx, line in enumerate(array_in): # 逐行遍历输入数据array_in
                for dim, data in enumerate(line):
                    if not mask[dim]:
                        continue
                    pre = array_in[idx - 1, dim] # 上一行中该特征的值
                    # 如果存在mean-10*std ~ mean+10*std之外的数据 将数据修改为上一行中该特征的值
                    if data > (mean[dim] + 10 * std[dim]) or data < (mean[dim] - 10 * std[dim]):
                        # print('[outlier]', idx, ':', array_in[idx, :], 'dim:', dim, '->', pre)
                        array_in[idx, dim] = pre
                    # 如果存在pre-5*std ~ pre+5*std之外的数据 将数据修改为上一行中该特征的值
                    elif data > (pre + 5 * std[dim]) or data < (pre - 5 * std[dim]):
                        # print('[noise]', idx, ':', array_in[idx, :], 'dim:', dim, '->', pre)
                        array_in[idx, dim] = pre
            return array_in

        def merge_temperature(te_in):
            te_row_count, te_num = te_in.shape
            if te_num == 1: # 只有一个油温数据
                #for i in range(te_row_count):
                    #if te_in[i] < 0:
                        #print("temperature " + str(i) + " is lower than 0.")
                return te_in
            else:
                te_selected = []
                for idx_te in range(te_num): # 遍历所有的油温特征
                    feature = te_in[:, idx_te] # 将第idx_te个油温提取出来转换成行向量
                    mean = np.mean(feature)
                    if mean > 5.0: # 油温平均值大于5 该项油温特征才能被使用
                        te_selected.append(idx_te)
                te_out = np.zeros((te_row_count, 1))
                for idx in range(te_row_count):
                    if len(te_selected) == 1: # 如果最终只有一个油温特征被选中 则直接返回油温数据
                        te_out[idx, 0] = np.array(te_in[idx, te_selected[0]])
                    else: #否则,返回被选中的多个油温的平均值
                        te_out[idx, 0] = np.mean([te_in[idx, i] for i in te_selected])
                #for i in range(te_row_count):
                    #if te_out[i] < 0:
                        #print("temperature " + str(i) + " is lower than 0.")
                return te_out

        data_clean = []
        for tf_id in range(self.tf_num):
            time = []
            data_raw = self.data[tf_id]
            # data --> 去掉时间戳和油温的数据
            data = np.zeros((len(data_raw), MAX_DIM - 2), dtype=np.float)
            data_num = 0 # 总的数据行数
            # te --> 油温数据
            te_dim = len(next(self.pop(tf_id))[7])
            te = np.zeros((len(data_raw), te_dim), dtype=np.float)
            for item in data_raw:
                if check_time_det(item[0]): # 检查分钟数是否为0/15/30/45
                    time.append(item[0])
                    data[data_num, :] = item[1:-1]
                    te[data_num, :] = item[7]
                    data_num += 1
            # 缩短数据
            time, data, te, data_num = truncate_end(time, data[:data_num, :], te[:data_num, :])
            # merge_temperature将多个油温特征合并为1个
            # 元组(data, merge_temperature(te)) for迭代先返回data再返回merge_temperature(te)

            # data, te = [del_outlier(array) for array in (data, merge_temperature(te))]
            data, te = [fill_zero(array) for array in (data, merge_temperature(te))]
            data_list = []
            for idx in range(data_num):
                # 将第idx行的data和te在水平方向上平铺 并转换成列表
                data_temp = np.hstack((data[idx, :], te[idx, :])).tolist()
                data_temp[0:0] = [time[idx]] # 列表头插入时间戳
                data_list.append(data_temp)
            data_clean.append(data_list)

        return data_clean

    def draw(self, tf_id, dim): # 画出tf_id号寄存器第dim维特征的图像
        num = len(self.data[tf_id])
        x = range(num)
        y = np.zeros(num)
        for idx, item in enumerate(self.pop(tf_id)):
            y[idx] = float(item[dim])
        # y.sort()
        length = 1000

        # for idx, item in enumerate(self.pop(tf_id)):
        #     if item[1] > 300:
        #         print(item)

        plt.figure(1)
        plt.subplot(211)
        plt.title('transformer %d, dim = %d, max = %f, min = %f' % (tf_id, dim, max(y), min(y)))
        # plt.plot(x[1:length], y[1:length])
        plt.plot(x, y)

        plt.subplot(212)
        plt.hist(y, bins=50, range=[-100, 150]) # 绘制直方图 bins表示直方图的总个数

        plt.show()


if __name__ == '__main__':
    if not DEBUG:
        test_tf = TransformerClean('./liuan/110kV永宁变')
        if SAVE:
            with open('./tmp/test_tf.class', 'wb+') as f:
                pk.dump(test_tf, f)
    else:
        with open('./tmp/test_tf.class', 'rb') as f:
            test_tf = pk.load(f)
    print('Database created. %d transformer in total.' % test_tf.tf_num)

    # for idx, item in enumerate(test_tf.pop(0)):
    #     if 3000 < idx < 3415:
    #         print(idx, ':', item)
    for i in range(test_tf.tf_num):
        for j in range(1, 8):
            test_tf.draw(i, j)
