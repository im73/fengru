import tensorflow as tf
import numpy as np
import pandas as pd

import os

import multiprocessing as mp

import time
import datetime

def get_optimizer(optimizer):
    if optimizer is 'adam':
        opt = tf.train.AdamOptimizer
    elif optimizer is 'adadelta':
        opt = tf.train.AdadeltaOptimizer
    else:
        opt = tf.train.RMSPropOptimizer
    return opt

def init_device(devices, total_occupy):
    device_map = mp.Manager().dict()
    for device in devices:
        device_map[device] = total_occupy
    return device_map
    
def get_device(devices, per_occupy, device_map):
    for device in devices:
        if device_map[device] >= per_occupy:
            device_map[device] -= per_occupy
            print(device, per_occupy)
            print(device_map)
            return device
        
def releas_device(device, per_occupy, device_map):
    print(device,'release',per_occupy)
    device_map[device] += per_occupy
    print(device_map) 
    
def train_model(Model, DataEnv, device_map, train_args):
    env = DataEnv(path=train_args['data_path'],
                  seq_len=train_args['seq_len'],
                  label_len=train_args['label_len'],
                  target='t_T') # 建立数据获取环境
    opt = get_optimizer(train_args['optimizer']) # 获取优化器
    
    device = get_device(train_args['devices'], train_args['gpu_occupy'], device_map)
    print('start',train_args['model_name'],'get device',device)
    for itr in range(train_args['iterations']):
        # 保存模型的路径
        save_path = './save/'+train_args['record_dir']+train_args['dataset']+'_itr'+str(itr)+train_args['model_name']
        # 共用env 获取相关参数
        env.reset_count()
        input_size,output_size = env.get_datasize()
        
        # 关于 graph 的设置
        train_graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=train_args['gpu_occupy'])  
        # gpu_options = tf.GPUOptions(allow_growth = True)
        config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True) 
        sess = tf.Session(graph=train_graph, config=config)
        # graph 的建立
        with train_graph.as_default():
            model = Model(env=env, session=sess,
                          seq_len=train_args['seq_len'],label_len=train_args['label_len'],
                          input_size=input_size, output_size=output_size,
                          batch_size=train_args['batch_size'],
                          **{
                              'learning_rate':train_args['learning_rate'],
                              'train_epochs':train_args['train_epochs'],
                              'hidden_size':train_args['hidden_size'],
                              'attn_size':train_args['attn_size'],
                              'layer_count':1,
                              'optimizer':opt, # 模型参数
                              'device':device,
                              'mode':train_args['mode'], # 和模型训练相关
                              'enable_saver':train_args['enable_saver'],
                              'save_path':save_path,
                              'save_step':train_args['save_step'], # 和模型保存相关
                            })
            start_t = time.clock()
            model.run()
            end_t = time.clock()
            
            print(end_t-start_t)
            #model.eval_and_plot()
        
        # 关闭sess和删除图
        sess.close()
        del sess
        del train_graph
    # 释放GPU显存
    releas_device(device, train_args['gpu_occupy'], device_map)

def grid_search(
    Model = None,
    DataEnv = None,
    
    batch_size_list = [32,64,128],
    hidden_size_list = [32,64,128],
    train_epochs_list = [100],

    seq_len_list = [10],
    label_len_list = [1],
    learning_rate_list = [0.001],
    
    iterations = 5,
    process_num = 5,
    
    data_path = '',
    model_index = '',
    
    dataset = '',
    record_dir = 'NASDAQ/',
    record_tmp_dir = 'NASDAQ/Tmp/',
    addition = '',
    
    devices = ['/gpu:0','/gpu:1','/gpu:2'],
    gpu_occupy = 0.2,
    total_occupy = 0.6,
    mode = 'train',
    enable_saver = False,
    save_step = 10,
    early_stop = 10,
    optimizers = ['adam'],
):
    # 考虑废弃
    # global device_count,device_occupy
    # device_count = 0; device_occupy = 0;
    device_map = init_device(devices, total_occupy)
    
    pool = mp.Pool(processes=process_num)
    results = []
    for batch_size in batch_size_list:
        for hidden_size in hidden_size_list:
            for seq_len in seq_len_list:
                for label_len in label_len_list:
                    for train_epochs in train_epochs_list:
                        learning_rate = learning_rate_list[0]
                        optimizer = optimizers[0]
                        
                        model_name = model_index+'_bs'+str(batch_size)+'_hs'+str(hidden_size)+\
                                    '_te'+str(train_epochs)+'_sql'+str(seq_len)+'_labl'+str(label_len)+\
                                    '_opt'+str(optimizer)
                        train_args = {
                            'batch_size':batch_size, 'hidden_size':hidden_size, 'attn_size':hidden_size,
                            'seq_len':seq_len, 'label_len':label_len,
                            'train_epochs':train_epochs, 'iterations':iterations, #以上是需要主要调整的参数
                            'enable_saver':enable_saver, 'save_step':save_step,
                            'optimizer':optimizer, 'mode':mode, 
                            'early_stop':early_stop, 'learning_rate':learning_rate, # 以上均与模型参数相关
                            'devices':devices,
                            'gpu_occupy':gpu_occupy, 'total_occupy':total_occupy, # 与gpu分配相关
                            'data_path':data_path, 'dataset':dataset, # 与data_env相关
                            'record_dir':record_dir, 'record_tmp_dir':record_tmp_dir, # 与结果记录相关
                            'model_name':model_name, 'addition':addition # 与模型和结果保存相关
                        }
                        print(model_name)
                        #train_model(Model, DataEnv, train_args)
                        
                        results.append(pool.apply_async(train_model, (Model, DataEnv, device_map, train_args)))
    for result in results:
        print(result.get())
    
    pool.close()
    pool.join()
    
    
    
    
    
    
