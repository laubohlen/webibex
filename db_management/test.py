import datetime

from core.models import IbexImage

images = IbexImage.objects.all()

for image in images:
    dt = image.exif.get("DateTime", None)
    if isinstance(dt, str):
        dt_object = datetime.datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
        if isinstance(dt_object, datetime.datetime):
            print(dt_object)
        else:
            print("no datetime exif")
    else:
        print("no exif")