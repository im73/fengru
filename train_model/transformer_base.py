#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 18-11-28
# @Author  : haoyi

import os
import re
import csv
import codecs
import numpy as np

MAX_TRANSFORMER = 10
MAX_DIM = 8  # 时间，高有，高无，中有，中无，低有，低无，温度集合


class Transformer:
    def __init__(self, folder_path):
        #folder_name = folder_path.split(os.sep) #分割失败 os.sep为\ 传入的如果是相对路径则无\只有/
        # 如果folder_path是dir,则self.name表示最底层文件夹名;否则self.name表示文件名
        #self.name = folder_name[-1] if folder_name[-1] is not '' else folder_name[-2]

        #print(os.sep,folder_path,folder_name,self.name)

        (folder_name,self.name) = os.path.split(folder_path)
        #print(folder_name,self.name)
        
        self.root_path = folder_path
        self.filenames = self.__dirPath__()
        self.filenames.sort()
        # 直接调用read_data 获取每个变压器的数据和变压器数量
        self.data, self.tf_num = self.__read_data__()

    def __dirPath__(self):
        file_list = None
        try:
            file_list = os.listdir(self.root_path)
        except IOError as err:
            print('IOError: ' + err.filename)
        return file_list

    def __read_data__(self):

        def is_number(num):
            pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
            result = pattern.match(num)
            if result:
                return True
            else:
                return False

        def transform_alabo2_roman_num(one_num):
            num_list = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
            str_list = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
            res = ''
            for i in range(len(num_list)):
                while one_num >= num_list[i]:
                    one_num -= num_list[i]
                    res += str_list[i]
            return res

        def transform_alabo2_roman_num_s(one_num):
            if one_num > 10:
                return None
            else:
                str_list = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ', 'X']
                return str_list[one_num-1]

        def exist_transformer(string_in, id_in, speical = False):
            if string_in.find('#' + str(id_in)) > -1:
                return True
            elif string_in.find(str(id_in) + '号') > -1:
                return True
            elif string_in.find(transform_alabo2_roman_num(id_in)) > -1:
                return True
            elif string_in.find(transform_alabo2_roman_num_s(id_in)) > -1:
                return True
            if speical:
                if string_in.find(str(id_in) + '#') > -1:
                    return True
            return False

        def generate(template_in, col_idx, col_in):
            if col_in.find('高有') > -1:
                template_in[col_idx] = 1
            elif col_in.find('高无') > -1:
                template_in[col_idx] = 2
            elif col_in.find('中有') > -1:
                template_in[col_idx] = 3
            elif col_in.find('中无') > -1:
                template_in[col_idx] = 4
            elif col_in.find('低有') > -1:
                template_in[col_idx] = 5
            elif col_in.find('低无') > -1:
                template_in[col_idx] = 6
            elif col_in.find('油温') > -1 or col_in.find('温度') > -1:
                template_in[col_idx] = 7
            return template_in

        def process_head(head_in):
            templates = []
            for id in range(MAX_TRANSFORMER):
                # 表头中#id标识变压器编号 找出变压器的数量num_tf
                if not exist_transformer(head_in, id + 1):
                    num_tf = id
                    break
            head_s = head_in.split('\t')
            if num_tf == 1:# 只包含一台变压器
                template = [-1 for i in range(len(head_s))]
                for col_index, col in enumerate(head_s):
                    # 将表头转换成1-7 col_index-->列的编号 col-->表头内容
                    template = generate(template, col_index, col)
                template[0] = 0  # for time stamp
                templates.append(template)
            else: # 包含多台变压器
                for id in range(num_tf):
                    template = [-1 for i in range(len(head_s))]
                    for col_index, col in enumerate(head_s):
                        # 多加了一种特殊标识 --> id#
                        if exist_transformer(col, id + 1, True):
                            template = generate(template, col_index, col)
                    template[0] = 0  # for time stamp
                    # template的下标为列号 存的为0-7
                    templates.append(template)
            return num_tf, templates

        def process_line(line_in, mask_in):
            line_s = line_in.split('\t')
            data = [0.0 for i in range(MAX_DIM)]
            temperature = []
            for pos, idx in enumerate(mask_in):
                # check for empty case
                data_string = line_s[pos]
                data_cell = 0.0 if data_string is '' else data_string
                # feed data
                if idx == 0: # 时间戳
                    data[0] = data_cell
                else: # idx -1&1~7 7可能有多个
                    data_cell = 0.0 if not is_number(data_string) else data_string
                    if idx == 7: # 油温和功率分开记录
                        temperature.append(float(data_cell))
                    elif idx > 0:
                        data[idx] = float(data_cell)
            data[-1] = temperature # 最后将油温集合(data的第7个元素是一个list)添加到data的最后一个
            return data

        data_list = []
        for filename in self.filenames:
            if filename[0] == '.':
                continue
            file_path = os.path.join(self.root_path, filename)
            try:
                tf_file = codecs.open(file_path, 'r', 'gbk')
                reader = csv.reader(tf_file)
                head = next(reader)[0]
            except UnicodeDecodeError as e:
                print('This transformer data is not gbk encoded.', e)
                tf_file = codecs.open(file_path, 'r', 'utf-8')
                reader = csv.reader(tf_file)
                head = next(reader)[0]
                print(head)
            num_tf, template_list = process_head(head)
            # 处理表头 获得变压器数目和转换后的表头
            # print(num_tf)
            # print(template_list)
            if not data_list: # data_list如果为None 则重新初始化
                data_list = [[] for i in range(num_tf)]
            for row in reader: # 单独提取出每个变压器的数据
                for tf_id in range(num_tf):
                    data_list[tf_id].append(process_line(row[0], template_list[tf_id]))
                    # 时间，高有，高无，中有，中无，低有，低无，温度集合
                    # print('#'+str(tf_id), data_list[tf_id][-1])
            tf_file.close()
        # 返回每个变压器的数据和变压器的数量
        return data_list, num_tf

    def pop(self, transformer_id):
        for record in self.data[transformer_id]:
            yield record #generator 可用于迭代
            # 下一次迭代时,从上一次迭代遇到的yield后面的代码开始执行。


if __name__ == '__main__':
    beita_tf = Transformer('./liuan/110kV北塔变')
    print('Database created.')
    print(beita_tf.tf_num)

    for idx, item in enumerate(beita_tf.pop(0)):
        print(idx, ':', item)
