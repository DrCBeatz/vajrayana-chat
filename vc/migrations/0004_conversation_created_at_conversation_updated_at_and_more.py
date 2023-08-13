# Generated by Django 4.1 on 2023-08-13 20:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vc', '0003_document_html_url_document_youtube_url_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='conversation',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='conversation',
            name='updated_at',
            field=models.DateTimeField(auto_now=True, null=True),
        ),
        migrations.AddField(
            model_name='message',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
    ]
