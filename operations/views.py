#-*- coding:utf-8 -*-
import datetime
import json
import re
import os
import threading
import random
from django.http import HttpResponse, JsonResponse, FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from operations.models import User
from operations.models import data as dt
from operations.models import commit_message
from operations.models import Train_model
import matplotlib.pyplot as plt
from  train_model.dl import model_run
import numpy as np






# Create your views here.
@csrf_exempt
def login(request):

    if request.method=="POST":

        user_name = request.POST.get('username')
        password = request.POST.get('password')
        user = User.objects.filter(username=user_name,password=password)

        if user:
            return JsonResponse({'status': 1,'success': "登录成功"})

    return JsonResponse({'status': 0, 'success': "用户或者密码名错误"})


def test_log(request):
    data={''}
    msg="登陆成功"
    code=200
    # return JsonResponse({msg:'json.dumps(msg, cls=DateEncoder),c'code'})
    # return JsonResponse({'list': json.dumps(list, cls=DateEncoder),'msg':json.dumps(msg, cls=DateEncoder),c'code'})

def upload_data(request):
    try:
        file=request.FILES.get('file')
        new_file=dt(file=file)
        new_file.save()

        return JsonResponse({'status': 1,'image_path': str(new_file.file)})
    except Exception:
        return JsonResponse({'status': 0, 'image_path': 0})

def add_data(request):

    desciption=request.POST.get('description')
    image_path=request.POST.get('image_path')
    dataspecs=request.POST.get('dataSpecs') #历史数据或者
    if ( (desciption==None) | (image_path==None) | (dataspecs==None)):
        return JsonResponse({'status': 0, 'message':"必填项不能为空"})
    user=request.POST.get('user')
    user=User.objects.get(username=user)
    if dataspecs=="predict":
        dataspecs=1
    else:
        dataspecs=0
    image_path=re.sub("[\"\\[\\]]","",image_path)
    image_path=image_path.split(',')
    new_sm=commit_message(owner=user,description=desciption,category=dataspecs)
    new_sm.save()

    for path in image_path:

        file = dt.objects.get(file=path)
        file.addition_meg=new_sm
        file.save()
    return JsonResponse({'status': 1, 'message':"上传成功"})


def get_files(request):

    user="im73"
    user=User.objects.get(username=user)
    data_set=dt.objects.filter(addition_meg__owner=user)
    data_sets={}
    idx='0'
    for data_file in data_set:

        data_dict={}
        print(data_file.file.path)
        data_dict['name']=data_file.file.path.split('/')[-1]
        data_dict['date']=str(data_file.date.date())
        data_dict['detail_message']=data_file.addition_meg.description
        data_dict['type']=data_file.addition_meg.category
        data_sets[idx]=json.dumps(data_dict,ensure_ascii=False)
        idx=idx+'1'

    return JsonResponse(data_sets)


def delete_file(request):
    try:
        user = request.POST.get('username')
        user = User.objects.get(username=user)
        data_set = dt.objects.filter(addition_meg__owner=user)
        idx = int(request.POST.get('index'))

        for data in data_set:
            if idx ==0:
                if os.path.isfile(data.file.path):
                    os.remove(data.file.path)
                data.delete()
            idx=idx-1
        return JsonResponse({'status': 1, 'message': "删除成功"})
    except :
        return JsonResponse({'status': 0, 'message':"删除文件出错"})

def download_file(request):

    try:
        user = "im73"
        user = User.objects.get(username=user)
        data_set = dt.objects.filter(addition_meg__owner=user)
        idx = int(request.GET.get('index'))

        for data in data_set:
            if idx ==0:
                if os.path.isfile(data.file.path):
                    file_name = data.file.path.split("/")[-1]
                    file = open(data.file.path, 'rb')
                    print(data.file.path)
                    response = FileResponse(file)
                    response["Content-Disposition"] = "attachment;filename*=UTF-8''%s"%(file_name);
                    print(response["Content-Disposition"] )

                    return response
            idx=idx-1

    except :
        return JsonResponse({'status': 0, 'message':"下载文件出错"})

def get_models(request):

    user="im73"
    user=User.objects.get(username=user)
    model_set=Train_model.objects.filter(owner=user)
    model_sets={}
    idx='0'

    for model in model_set:

        model_dict={}

        model_dict['name']=model.model_path.split('/')[-2]
        ###!!记得修改
        model_dict['date']=str(model.date.date())
        model_dict['state']=model.state
        model_dict['type']=model.type
        model_sets[idx]=json.dumps(model_dict,ensure_ascii=False)
        print(model_sets[idx])
        idx=idx+'1'

    return JsonResponse(model_sets)

def train_model(request):

    user="im73"
    file_index = request.POST.get('file_index')
    name = request.POST.get('name')
    type= int(request.POST.get('type'))

    user = User.objects.get(username=user)

    file_path = re.sub("[\"\\[\\]]", "", file_index)
    file_path = file_path.split(',')
    idx=0

    for i in range(len(file_path)):
        file_path[i]=int(file_path[i])-1

    paths=[]

    data_set = dt.objects.filter(addition_meg__owner=user)

    for data in data_set:
        if idx in file_path:
            if os.path.isfile(data.file.path):
                paths.append(data.file.path)
        idx = idx + 1


    dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_name =dir_path+"/media/model/"+name+"/"
    #待修改

    if type==1:
        model_name1='lstm'
    elif type==2:
        model_name1='seq2seq'
    elif type==3:
        model_name1='blstm'
    else:
        model_name1='xgboost'
    new_model=Train_model(model_path=model_name,name=name,owner=user,state=int(random.random()*100),type=type)
    new_model.save()

    try:

        if  not os.path.exists(model_name):
            os.makedirs(model_name)
        threading.Thread(target=model_run.read_arr,args=(paths[0],model_name,True,False,model_name1)).start()
        #model_run.read_arr(path=paths[0], model_path=model_name, run_model=True, run_pred=False, model_name=model_name1)

    except:
        print("error in training")


    return JsonResponse({'status': 1, 'message': "模型训练中"})

def delete_model(request):
    try:
        user = request.POST.get('username')
        user = User.objects.get(username=user)
        model_set = Train_model.objects.filter(owner=user)

        idx = int(request.POST.get('index'))

        for model in model_set:
            if idx ==0:
                if os.path.isfile(model.model_path):
                    os.remove(model.model_path)
                model.delete()
            idx=idx-1

        return JsonResponse({'status': 1, 'message': "删除成功"})
    except :

        return JsonResponse({'status': 0, 'message':"删除模型出错"})


def predict(request):


        user = request.POST.get('username')
        user = User.objects.get(username=user)
        data_set = dt.objects.filter(addition_meg__owner=user,addition_meg__category=1)
        model_set = Train_model.objects.filter(owner=user)

        model_idx = int(request.POST.get('model_index'))
        file_index = int(request.POST.get('file_index'))
        print(model_idx)
        model_path=''
        file_path=''

        for model in model_set:
            if model_idx == 0:
                model_path = model.model_path
            model_idx = model_idx - 1



        for data in data_set:
            if file_index == 0:
                if os.path.isfile(data.file.path):
                    file_path = data.file.path
            file_index = file_index - 1

        result=model_run.read_arr(path=file_path,model_path=model_path,run_pred=True,run_model=False,model_name="lstm")
        pred=result[0][0:96]

        return HttpResponse(json.dumps({"realTem":result[1],"predTem":pred},ensure_ascii=False),content_type="application/json,charset=utf-8",status=200)



def get_sets(request):

    user = request.POST.get('username')
    user = User.objects.get(username=user)

    data_set = dt.objects.filter(addition_meg__owner=user,addition_meg__category=1)
    model_set = Train_model.objects.filter(owner=user)

    model_sets=[]
    file_sets=[]

    data_sets={}

    index =1

    for file in data_set:
        file_sets.append(file.file.path.split('/')[-1])
    for model in model_set:
        model_sets.append(model.model_path.split('/')[-2])

    data_sets['file_sets']=file_sets
    data_sets['model_sets']=model_sets

    return JsonResponse(data_sets)


def home(request):

    return render(request,"manage/index.html")
