# Generated by Django 4.1 on 2023-08-06 19:46

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Conversation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='Model',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='GPT-3.5-Turbo', max_length=255)),
                ('context_length', models.IntegerField(default=4096)),
                ('input_token_cost', models.FloatField(default=0.0015)),
                ('output_token_cost', models.FloatField(default=0.002)),
            ],
        ),
        migrations.CreateModel(
            name='Message',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField()),
                ('answer', models.TextField()),
                ('context', models.TextField()),
                ('conversation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vc.conversation')),
            ],
        ),
        migrations.CreateModel(
            name='Expert',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='All You Need Music store handbook', max_length=255)),
                ('prompt', models.TextField(default='Answer the question based on the context below.')),
                ('model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vc.model')),
            ],
        ),
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('content', models.TextField(blank=True, null=True)),
                ('text_file', models.FileField(blank=True, null=True, upload_to='text_files/')),
                ('embeddings_file', models.FileField(blank=True, null=True, upload_to='embeddings/')),
                ('expert', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vc.expert')),
            ],
        ),
        migrations.AddField(
            model_name='conversation',
            name='expert',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='vc.expert'),
        ),
        migrations.AddField(
            model_name='conversation',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
