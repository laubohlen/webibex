# Generated by Django 5.0.11 on 2025-02-05 13:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0013_allow_id_code_of_size_10"),
    ]

    operations = [
        migrations.CreateModel(
            name="CircularRegion",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(blank=True, max_length=50, null=True)),
                ("origin_latitude", models.FloatField(blank=True, null=True)),
                ("origin_longitude", models.FloatField(blank=True, null=True)),
                ("radius", models.IntegerField(blank=True, default=2000, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Location",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("latitude", models.FloatField(blank=True, null=True)),
                ("longitude", models.FloatField(blank=True, null=True)),
            ],
        ),
    ]
