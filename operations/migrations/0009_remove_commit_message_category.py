# Generated by Django 2.1.5 on 2019-03-25 19:44

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('operations', '0008_train_model_type'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='commit_message',
            name='category',
        ),
    ]
