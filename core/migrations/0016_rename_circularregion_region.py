# Generated by Django 5.0.11 on 2025-02-06 09:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0015_ibeximage_add_FK_to_location"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="CircularRegion",
            new_name="Region",
        ),
    ]
