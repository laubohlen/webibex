# Generated by Django 5.0.2 on 2024-06-10 08:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0005_changed_side_choices_ordering"),
    ]

    operations = [
        migrations.AlterField(
            model_name="ibeximage",
            name="side",
            field=models.CharField(
                blank=True,
                choices=[("R", "right"), ("L", "left"), ("O", "other")],
                max_length=1,
                null=True,
            ),
        ),
    ]
