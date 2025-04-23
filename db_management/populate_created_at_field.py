import os
import datetime
from core.models import IbexImage

images = IbexImage.objects.all()

for image in images:
    filename, file_extenstion = os.path.splitext(str(image))
    parts = filename.split("_")[-4:]
    dt_string = "_".join(parts)
    dt_object = datetime.datetime.strptime(dt_string, "%y_%m_%d_%H%M%S")
    image.created_at = dt_object
    image.save(update_fields=["created_at"])