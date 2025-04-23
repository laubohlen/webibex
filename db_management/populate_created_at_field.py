import os
import sys
import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webibex.settings') 
django.setup()

from core.models import IbexImage

images = IbexImage.objects.all()

for image in images:
    dt = image.exif.get("DateTime", None)
    if isinstance(dt, str):
        dt_object = datetime.datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
        if isinstance(dt_object, datetime.datetime):
            image.created_at = dt_object
        else:
            image.created_at = datetime.datetime.now()
    else:
        image.created_at = datetime.datetime.now()
    image.save(update_fields=["created_at"])
    print(f"Updated image {image.id}")
