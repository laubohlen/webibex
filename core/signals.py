import os
import datetime
import tensorflow as tf

from django.dispatch import receiver
from django.db.models.signals import post_save, post_delete, pre_save
from django.utils.encoding import force_str
from django.utils.timezone import now
from django.utils.text import get_valid_filename as get_valid_filename_django
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.conf import settings
from allauth.account.signals import user_signed_up

from .models import IbexImage, IbexChip, Embedding, Animal
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
            Folder.objects.create(name="_left", owner=user, parent=main_folder)
            Folder.objects.create(name="_right", owner=user, parent=main_folder)
            Folder.objects.create(name="_other", owner=user, parent=main_folder)


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


@receiver(post_delete, sender=IbexImage)
def delete_landmark_items(sender, instance, **kwargs):
    image = instance
    content_type = ContentType.objects.get_for_model(IbexImage)
    landmark_items = LandmarkItem.objects.filter(
        content_type=content_type, object_id=image.id
    )
    landmark_items.delete()


@receiver(post_delete, sender=IbexChip)
def delete_ibexchip_file(sender, instance, **kwargs):
    # Delete the associated file when the IbexChip instance is deleted
    if instance.file:
        instance.file.delete(
            save=False
        )  # Delete the file without saving the model again


@receiver(post_save, sender=IbexChip)
def embed_new_chip(sender, instance, created, **kwargs):
    ibexchip = instance
    if created:
        chip_path = os.path.join(settings.MEDIA_ROOT, ibexchip.file.name)
        chip_size = (288, 144)
        chip_encoded = tf.io.read_file(chip_path)
        chip_decoded = tf.image.decode_jpeg(chip_encoded, channels=3)
        chip_resized = tf.image.resize(chip_decoded, chip_size)
        chip = tf.expand_dims(chip_resized, axis=0)
        model = tf.saved_model.load("core/embedding_model/")
        embedder = model.signatures["serving_default"]

        output = embedder(chip)["output_tensor"].numpy().tolist()[0]
        print(output)
        print(type(output))

        Embedding.objects.create(ibex_chip=ibexchip, embedding=output)
    else:
        pass


@receiver(pre_save, sender=IbexImage)
def check_animal_id_change(sender, instance, **kwargs):
    # If the instance exists, get the current animal_id from the database
    if instance.pk:
        original_instance = IbexImage.objects.get(pk=instance.pk)
        print("*original_instance", original_instance)
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
            user = User.objects.get(pk=instance.owner_id)
            username = user.username
            user_main_folder_name = f"{username}_files"
            user_main_foler = Folder.objects.filter(
                name=user_main_folder_name, owner=user
            ).first()

            # Check if a folder with the animal ID exists for the user
            if not Folder.objects.filter(name=animal_id, owner=user).exists():
                print("****folder doesnt exist", instance.animal)
                # Create the folder if it does not exist
                animal_folder = Folder.objects.create(
                    name=animal_id, owner=user, parent=user_main_foler
                )

                # Create subfolders "left" and "right" inside the new animal folder
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
                if len(parts) >= 3: # old_name = "PNGP_---_yy_mm_dd_HHMMSS.ext" or "PNGP_---_noexif.ext"
                    new_filename = f"{animal_id}_{"_".join(parts[2:])}"
                else: # old_name = "V01O_noexif.ext"
                    new_filename = f"{animal_id}_{parts[1]}"

                # rename and finally save all changes
                instance.name = new_filename
                instance.save()
