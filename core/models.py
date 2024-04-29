from django.db import models
from django.contrib.auth.models import AbstractUser
from filer.models.abstract import BaseImage as FilerBaseImage


class User(AbstractUser):
    pass


class Animal(models.Model):
    id_code = models.CharField(max_length=8, null=True, blank=True)
    name = models.CharField(max_length=50, null=True, blank=True)
    marked = models.BooleanField(default=False)
    capture_date = models.DateField(null=True, blank=True)
    cohort = models.CharField(max_length=4, null=True, blank=True)
    sex = models.CharField(max_length=1, null=True, blank=True)

    def __str__(self) -> str:
        return self.id_code


class IbexImage(FilerBaseImage):
    SIDE_CHOICES = {
        ("L", "left"),
        ("R", "right"),
        ("O", "other"),
    }
    animal = models.ForeignKey(Animal, on_delete=models.SET_NULL, null=True, blank=True)
    side = models.CharField(max_length=1, choices=SIDE_CHOICES, null=True, blank=True)

    class Meta(FilerBaseImage.Meta):
        # You must define a meta with en explicit app_label
        app_label = "core"
        default_manager_name = "objects"
