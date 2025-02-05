from django.db import models
from django.contrib.auth.models import AbstractUser

from filer.models.abstract import BaseImage as FilerBaseImage

from collections import OrderedDict


class User(AbstractUser):
    pass


class Animal(models.Model):
    id_code = models.CharField(max_length=10, null=True, blank=True)
    name = models.CharField(max_length=50, null=True, blank=True)
    marked = models.BooleanField(default=False)
    capture_date = models.DateField(null=True, blank=True)
    cohort = models.CharField(max_length=4, null=True, blank=True)
    sex = models.CharField(max_length=1, null=True, blank=True)

    def __str__(self) -> str:
        return self.id_code


class Location(models.Model):
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)


class IbexImage(FilerBaseImage):
    SIDE_CHOICES = OrderedDict(
        [
            ("L", "left"),
            ("R", "right"),
            ("O", "other"),
        ]
    )
    animal = models.ForeignKey(Animal, on_delete=models.SET_NULL, null=True, blank=True)
    side = models.CharField(max_length=1, choices=SIDE_CHOICES, null=True, blank=True)
    location = models.ForeignKey(
        Location, on_delete=models.SET_NULL, null=True, blank=True
    )

    class Meta(FilerBaseImage.Meta):
        # You must define a meta with en explicit app_label
        app_label = "core"
        default_manager_name = "objects"


class IbexChip(FilerBaseImage):
    ibex_image = models.OneToOneField(
        IbexImage, on_delete=models.CASCADE, null=True, blank=True
    )

    class Meta(FilerBaseImage.Meta):
        # You must define a meta with en explicit app_label
        app_label = "core"
        default_manager_name = "objects"


class Embedding(models.Model):
    ibex_chip = models.OneToOneField(
        IbexChip, on_delete=models.CASCADE, null=True, blank=True
    )
    embedding = models.JSONField(null=True, blank=True)  # store as array.tolist()
    time_date = models.DateTimeField(auto_now=True)


class CircularRegion(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True)
    origin_latitude = models.FloatField(null=True, blank=True)
    origin_longitude = models.FloatField(null=True, blank=True)
    radius = models.IntegerField(
        default=2000, null=True, blank=True
    )  # radius in meters
