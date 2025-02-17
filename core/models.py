from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.utils.translation import gettext_lazy as _

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


class Region(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True)
    origin_latitude = models.FloatField(null=True, blank=True)
    origin_longitude = models.FloatField(null=True, blank=True)
    radius = models.IntegerField(
        default=2000, null=True, blank=True
    )  # radius in meters
    owner = models.ForeignKey(
        getattr(settings, "AUTH_USER_MODEL", "auth.User"),
        related_name="owned_%(class)ss",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name=_("owner"),
    )

    # force unique region names per user
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "name"], name="unique_region_name_per_owner"
            )
        ]

    def __str__(self) -> str:
        return self.name


class Location(models.Model):
    SOURCE_CHOICES = OrderedDict(
        [
            ("gps", "location from camera GPS"),
            ("marker", "location marker set manually"),
            ("region", "location somewhere within this region"),
        ]
    )
    source = models.CharField(
        max_length=6, choices=SOURCE_CHOICES, null=True, blank=True
    )
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    exif_latitude = models.FloatField(null=True, blank=True)
    exif_longitude = models.FloatField(null=True, blank=True)
    region = models.ForeignKey(Region, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        # Use hasattr to check if there's an associated IbexImage.
        if hasattr(self, "ibeximage") and self.ibeximage:
            return f"Location for: {self.ibeximage.name}"
        return "Location for: [No IbexImage]"


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
    location = models.OneToOneField(
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
