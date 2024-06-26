import os
import datetime

from django.dispatch import receiver
from django.db.models.signals import post_save
from django.utils.encoding import force_str
from django.utils.timezone import now
from django.utils.text import get_valid_filename as get_valid_filename_django
from django.contrib.contenttypes.models import ContentType

from .models import IbexImage
from simple_landmarks.models import LandmarkItem, Landmark


@receiver(post_save, sender=IbexImage)
def rename_uploaded_image(sender, instance, created, **kwargs):
    image = instance
    location_id = "PNGP"
    unmarked_code = "---"
    season = force_str(now().strftime("%y"))
    if created:
        _, file_extenstion = os.path.splitext(image.file.name)
        if image.exif:  # no exif results in empty dictionary which bool(dict) == False
            created = image.exif["DateTime"]  # format: 2021:06:24 17:48:11
            created = datetime.datetime.strptime(str(created), "%Y:%m:%d %H:%M:%S")
            created = created.strftime("%y_%m_%d_%H%M%S")

            # orientation = image.exif["Orientation"]  # 1 = horizontal, ?? = portrait
            # gps_info = image.exif["GPSInfo"]
        else:  # no exif data
            created = "noexifdata"

        new_filename = "{}{}_{}_{}{}".format(
            location_id, season, unmarked_code, created, file_extenstion
        )
        image.name = get_valid_filename_django(new_filename)
        image.save()
    else:
        pass


@receiver(post_save, sender=IbexImage)
def initialise_landmark_items(sender, instance, created, **kwargs):
    image = instance
    if created:
        # initialise landmark-items for each landmark for the new image
        content_type = ContentType.objects.get_for_model(IbexImage)
        landmarks = Landmark.objects.all()
        for lm in landmarks:
            LandmarkItem.objects.create(
                content_type=content_type, object_id=image.id, landmark=lm
            )
    else:
        pass
