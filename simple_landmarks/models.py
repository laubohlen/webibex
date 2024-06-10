from django.db import models
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey


class Landmark(models.Model):
    label = models.CharField(max_length=20)

    def __str__(self) -> str:
        return self.label


class LandmarkItem(models.Model):
    landmark = models.ForeignKey(Landmark, on_delete=models.CASCADE)
    x_coordinate = models.PositiveBigIntegerField()
    y_coordinate = models.PositiveIntegerField()
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey()
