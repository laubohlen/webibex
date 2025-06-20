# Generated by Django 5.0.2 on 2024-07-22 09:52

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0009_change_ibexchip_to_one_to_one_relationship"),
    ]

    operations = [
        migrations.AlterField(
            model_name="ibexchip",
            name="ibex_image",
            field=models.OneToOneField(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to=settings.FILER_IMAGE_MODEL,
            ),
        ),
    ]
