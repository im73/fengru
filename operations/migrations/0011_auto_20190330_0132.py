# Generated by Django 2.1.5 on 2019-03-30 01:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('operations', '0010_auto_20190325_1945'),
    ]

    operations = [
        migrations.AlterField(
            model_name='train_model',
            name='model_path',
            field=models.CharField(max_length=400, verbose_name='模型文件路径'),
        ),
    ]
