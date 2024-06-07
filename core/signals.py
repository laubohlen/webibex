import os
import datetime

from django.dispatch import receiver
from django.db.models.signals import post_save
from django.utils.encoding import force_str
from django.utils.timezone import now
from django.utils.text import get_valid_filename as get_valid_filename_django

from .models import IbexImage


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
