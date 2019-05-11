"""fengru URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from operations import views as opv

urlpatterns = [

    path('admin/', admin.site.urls),
    path(r'/', opv.home),
    path(r'login/',opv.login,name="login"),
    path(r'test_log/',opv.test_log,name="test_log"),
    path(r'upload_data/',opv.upload_data,name="upload_data"),
    path(r'add_data/',opv.add_data,name="add_data"),
    path(r'get_files/',opv.get_files,name="get_files"),
    path(r'delete_file/',opv.delete_file,name="delete_files"),
    path(r'download_file/',opv.download_file,name="download_file"),
    path(r'get_models/',opv.get_models,name="get_models"),
    path(r'train_model/',opv.train_model,name="train_model"),
    path(r'delete_model/',opv.delete_model,name="delete_model"),
    path(r'predict/',opv.predict,name="predict"),
    path(r'get_sets/',opv.get_sets,name="get_sets")


   # path(r'operation/',include('operations.url')),
]
