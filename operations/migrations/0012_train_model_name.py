# Generated by Django 2.1.5 on 2019-04-17 21:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('operations', '0011_auto_20190330_0132'),
    ]

    operations = [
        migrations.AddField(
            model_name='train_model',
            name='name',
            field=models.CharField(default='', max_length=60, verbose_name='模型名称'),
            preserve_default=False,
        ),
    ]
