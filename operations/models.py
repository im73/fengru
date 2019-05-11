from django.db import models
from datetime import datetime

# Create your models here.
class User(models.Model):

    username = models.CharField(max_length=16,verbose_name=u"昵称",null=False,unique=True)
    password = models.CharField(max_length=12,verbose_name=u"密码",null=False)


    class Meta:
        verbose_name = u"用户信息"
        verbose_name_plural = verbose_name

class commit_message(models.Model):

    owner = models.ForeignKey(User, verbose_name=u"所有者", null=True, on_delete=models.CASCADE)
    category = models.IntegerField(null=False,verbose_name=u"文件类型",default=1)

    description = models.CharField(max_length=500,verbose_name=u"上传的附加信息",null=False)


    class Meta:
        verbose_name = u"数据上传信息"
        verbose_name_plural = verbose_name



class data(models.Model):

    file = models.FileField(upload_to="data/%Y/%M",verbose_name=u"数据文件", max_length=300)
    addition_meg = models.ForeignKey(commit_message,null=True,on_delete=models.CASCADE)
    date =  models.DateTimeField(default=datetime.now, verbose_name=u"文件上传时间")

    class Meta:
        verbose_name = u"数据信息"
        verbose_name_plural = verbose_name

class Train_model(models.Model):

    model_path = models.CharField(max_length=400,verbose_name=u"模型文件路径")
    owner = models.ForeignKey(User, verbose_name=u"所有者", null=True, on_delete=models.CASCADE)
    state = models.FloatField(verbose_name="训练进程")
    date = models.DateTimeField(default=datetime.now, verbose_name=u"模型添加时间")
    type = models.IntegerField(null=False,verbose_name="模型的类型")
    name = models.CharField(max_length=60,verbose_name=u"模型名称")

    class Meta:
        verbose_name = u"模型信息"
        verbose_name_plural = verbose_name


