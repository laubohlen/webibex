import os
import sys
from datetime import datetime

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
    if isinstance(dt, datetime):
        image.created_at = dt
    else:
        image.created_at = datetime.now()
    image.save(update_fields=["created_at"])
    print(f"Updated image {image.id}")
