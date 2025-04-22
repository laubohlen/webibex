import os
import datetime

from django.dispatch import receiver
from django.db.models.signals import post_save, post_delete, pre_save
from django.utils.encoding import force_str
from django.utils.timezone import now
from django.utils.text import get_valid_filename as get_valid_filename_django
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from allauth.account.signals import user_signed_up

from .models import IbexImage, IbexChip, Location
from simple_landmarks.models import LandmarkItem, Landmark
from filer.models import Folder


User = get_user_model()


@receiver(user_signed_up)
def user_signed_up_callback(request, user, **kwargs):
    group, created = Group.objects.get_or_create(name="public_users")
    user.groups.add(group)
    # TODO: create templates for accessing filer filemanagement outside the admin page
    # Set the user as staff to access filer folders
    # risky but my temprary.
    # user.is_staff = True
    # user.save()


@receiver(post_save, sender=User)
def create_user_folders(sender, instance, created, **kwargs):
    if created:
        # The user has been created
        user = instance
        folder_name = f"{user.username}_files"

        # Check if the main folder already exists
        if not Folder.objects.filter(name=folder_name, owner=user).exists():
            # Create the main folder
            main_folder = Folder.objects.create(name=folder_name, owner=user)

            # Create subfolders "_left" and "_right" inside the main folder
            Folder.objects.create(name="_left_upload", owner=user, parent=main_folder)
            Folder.objects.create(name="_right_upload", owner=user, parent=main_folder)
            Folder.objects.create(name="_other_upload", owner=user, parent=main_folder)


from PIL import Image, ExifTags


def get_decimal_from_dms(dms, ref):
    """
    Convert degrees, minutes, seconds to decimal degrees.

    dms: tuple of three values representing degrees, minutes, seconds.
         Each value can be either a float or a tuple (numerator, denominator).
    ref: 'N', 'S', 'E', or 'W'. South and West are negative.
    """

    def to_float(value):
        # If the value is a tuple (num, den), perform the division.
        if isinstance(value, tuple):
            try:
                return value[0] / value[1]
            except Exception as e:
                print("Error converting tuple to float:", e)
                return None
        # Otherwise, try to convert directly to float.
        try:
            return float(value)
        except Exception as e:
            print("Error converting value to float:", e)
            return None

    try:
        degrees = to_float(dms[0])
        minutes = to_float(dms[1])
        seconds = to_float(dms[2])
    except Exception as e:
        print("Decimal conversion error:", e)
        return None  # Return 0.0 if any error occurs in conversion

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref.upper() in ["S", "W"]:
        decimal = -decimal
    return decimal


def extract_gps_coords(filer_image):
    # get EXIF data
    exif = filer_image.exif

    # Get the GPSInfo data; if missing, return (None, None)
    gps_info = exif.get("GPSInfo")
    print("gps info:", gps_info)
    if not gps_info:
        print("No GPS information in EXIF data.")
        return None, None

    # Decode the GPSInfo keys to human-readable form.
    gps_data = {}
    for key in gps_info.keys():
        decoded_key = ExifTags.GPSTAGS.get(key, key)
        gps_data[decoded_key] = gps_info[key]

    # Check that all necessary fields are present.
    required_keys = ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]
    if not all(key in gps_data for key in required_keys):
        print("Missing GPS entries in EXIF data.")
        return None, None

    try:
        lat = get_decimal_from_dms(gps_data["GPSLatitude"], gps_data["GPSLatitudeRef"])
        lng = get_decimal_from_dms(
            gps_data["GPSLongitude"], gps_data["GPSLongitudeRef"]
        )
    except Exception:
        # In case something goes wrong during conversion
        return None, None

    return lat, lng


@receiver(post_save, sender=IbexImage)
def process_uploaded_image(sender, instance, created, **kwargs):
    image = instance
    location_id = "PNGP"
    unmarked_code = "---"
    season = force_str(now().strftime("%y"))
    if created:
        _, file_extenstion = os.path.splitext(image.file.name)
        if image.exif:  # no exif results in empty dictionary which bool(dict) == False
            createtime = image.exif["DateTime"]  # format: 2021:06:24 17:48:11
            createtime = datetime.datetime.strptime(str(createtime), "%Y:%m:%d %H:%M:%S")
            createtime = createtime.strftime("%y_%m_%d_%H%M%S")

            # orientation = image.exif["Orientation"]  # 1 = horizontal, ?? = portrait
            # gps_info = image.exif["GPSInfo"]
        else:  # no exif data
            createtime = "noexifdata"

        new_filename = "{}{}_{}_{}{}".format(
            location_id, season, unmarked_code, createtime, file_extenstion
        )
        image.name = get_valid_filename_django(new_filename)

        # tag left or right side if parent folder indicates this
        if image.folder.name == "_left_upload":
            image.side = "L"
        elif image.folder.name == "_right_upload":
            image.side = "R"
        elif image.folder.name == "_other_upload":
            image.side = "O"

        # extract location from exif if available
        if image.exif:  # no exif results in empty dictionary which bool(dict) == False
            latitude, longitude = extract_gps_coords(image)
        else:
            latitude, longitude = None, None

        # Create a new Location instance if one doesn't exist
        if image.location is None:
            location = Location.objects.create(
                latitude=latitude,
                longitude=longitude,
                exif_latitude=latitude,
                exif_longitude=longitude,
            )
            image.location = location
        else:
            # Otherwise, update the existing location.
            image.location.latitude = latitude
            image.location.longitude = longitude
        print("Created location object for image")

        # save "created_at" datetime for image
        if isinstance(image.exif["DateTime"], datetime): # Check if it's a datetime object
            image.created_at = image.exif["DateTime"]
            print("Updated file created_at field")
        else:
            image.created_at = datetime.now()
        
        # Save the image to update the relationship, if needed
        image.save()

    else:
        pass


@receiver(post_save, sender=IbexImage)
def initialise_landmark_items(sender, instance, created, **kwargs):
    image = instance
    if created:
        # initialise landmark-items as 0 for each landmark for the new image
        content_type = ContentType.objects.get_for_model(IbexImage)
        landmarks = Landmark.objects.all()
        if landmarks:
            for lm in landmarks:
                LandmarkItem.objects.create(
                    content_type=content_type,
                    object_id=image.id,
                    landmark=lm,
                    y_coordinate=0,
                    x_coordinate=0,
                )
            print("Created landmarkitem objects for image")
        else:
            print("No Landmarks available, please create Landmarks first")
    else:
        pass


@receiver(post_delete, sender=IbexImage)
def delete_landmark_items(sender, instance, **kwargs):
    image = instance
    content_type = ContentType.objects.get_for_model(IbexImage)
    landmark_items = LandmarkItem.objects.filter(
        content_type=content_type, object_id=image.id
    )
    landmark_items.delete()


@receiver(post_delete, sender=IbexImage)
def delete_associated_location(sender, instance, **kwargs):
    if instance.location:
        instance.location.delete()


@receiver(post_delete, sender=IbexChip)
def delete_ibexchip_file(sender, instance, **kwargs):
    # Delete the associated file when the IbexChip instance is deleted
    if instance.file:
        instance.file.delete(
            save=False
        )  # Delete the file without saving the model again


@receiver(pre_save, sender=IbexImage)
def check_animal_id_change(sender, instance, **kwargs):
    # If the instance exists, get the current animal_id from the database
    if instance.pk:
        original_instance = IbexImage.objects.get(pk=instance.pk)
        instance._original_animal_id = original_instance.animal_id
    else:
        # New instance, no previous animal_id
        instance._original_animal_id = None


@receiver(post_save, sender=IbexImage)
def create_folder_for_animal_on_change(sender, instance, **kwargs):
    # Compare the original animal_id with the current one
    if instance.animal_id != instance._original_animal_id:
        if instance.animal:
            # Get animal ID and user
            animal_id = instance.animal.id_code
            try:
                user = instance.owner
            except User.DoesNotExist:
                # If the user isn't found, exit early.
                return
            username = user.username
            user_main_folder_name = f"{username}_files"
            user_main_foler = Folder.objects.filter(
                name=user_main_folder_name, owner=user
            ).first()

            animal_folder, _ = Folder.objects.get_or_create(
                name=animal_id, owner=user, parent=user_main_foler
            )
            left_folder, _ = Folder.objects.get_or_create(
                name=f"left_{animal_id}", owner=user, parent=animal_folder
            )
            right_folder, _ = Folder.objects.get_or_create(
                name=f"right_{animal_id}", owner=user, parent=animal_folder
            )
            other_folder, _ = Folder.objects.get_or_create(
                name=f"other_{animal_id}", owner=user, parent=animal_folder
            )

            # Determine which subfolder the image should go to
            if instance.side == "L":
                target_folder = left_folder
            elif instance.side == "R":
                target_folder = right_folder
            elif instance.side == "O":
                target_folder = other_folder
            else:
                pass

            # Move the image to the correct folder
            instance.folder = target_folder

            # create new filename
            old_filename = instance.name
            parts = old_filename.split("_")
            if (
                len(parts) >= 3
            ):  # old_name = "PNGP_---_yy_mm_dd_HHMMSS.ext" or "PNGP_---_noexif.ext"
                new_filename = f"{animal_id}_{"_".join(parts[2:])}"
            else:  # old_name = "V01O_noexif.ext"
                new_filename = f"{animal_id}_{parts[1]}"

            # rename and finally save all changes
            instance.name = new_filename
            instance.save()
