# Generated by Django 5.0.6 on 2024-10-05 11:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cf_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='no_telp',
            field=models.CharField(default=0, max_length=100),
            preserve_default=False,
        ),
    ]
