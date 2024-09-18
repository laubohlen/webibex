import os
import cv2
import shutil
import numpy as np

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.contenttypes.models import ContentType
from django.db.models.aggregates import Count
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import render, get_object_or_404
from django.core.files.base import ContentFile

from pathlib import Path
from io import BytesIO
from PIL import Image
from core.models import IbexImage, IbexChip, Animal, Embedding
from simple_landmarks.models import LandmarkItem, Landmark
from . import utils, b2_utils


def welcome_view(request):
    return render(request, "core/welcome.html")


def upload_view(request):
    # link to django-filer folder of the user
    url = reverse("admin:filer_folder_changelist")
    return HttpResponseRedirect(url)


@login_required
def unidentified_images_view(request):
    # get all images that are not linked to any animal
    unidentified_images = IbexImage.objects.filter(animal_id__isnull=True)
    return render(
        request,
        "core/unidentified_images.html",
        {"images": unidentified_images},
    )


@login_required
def saved_animal_selection_view(request):
    # catching forms from selecting animal in show_result_view
    if request.method == "POST":
        oid = request.POST.get("selectedAnimalId")
        query_chip_id = request.POST.get("query_chip_id")
        # save the animal selection to the chip and image
        img = get_object_or_404(IbexImage, ibexchip=query_chip_id)
        img.animal = get_object_or_404(Animal, pk=oid)
        img.save()
        print("Saved selected animal to IbexImage.")
    else:
        pass

    # get all images of a specific animal
    images = IbexImage.objects.filter(animal_id=oid)
    # in case images is empty, get the animal name
    if not images:
        animal_id_code = get_object_or_404(Animal, pk=oid).id_code
    else:
        animal_id_code = images.first().animal.id_code

    return render(
        request,
        "core/animal_images.html",
        {"images": images, "animal_id_code": animal_id_code},
    )


@login_required
def animal_images_view(request, oid):
    # get all images of a specific animal
    images = IbexImage.objects.filter(animal_id=oid)
    # in case images is empty, get the animal name
    if not images:
        animal_id_code = get_object_or_404(Animal, pk=oid).id_code
    else:
        animal_id_code = images.first().animal.id_code
    return render(
        request,
        "core/animal_images.html",
        {"images": images, "animal_id_code": animal_id_code},
    )


@login_required
def observed_animal_view(request):
    url_name = request.resolver_match.url_name

    if url_name == "observed-animals":
        # get all animals that are linked to one or more images
        animals = Animal.objects.annotate(image_count=Count("ibeximage")).filter(
            image_count__gt=0
        )
    else:  # url_name == unobserved-animals
        # get all animals that are not featured in any images
        animals = Animal.objects.annotate(image_count=Count("ibeximage")).filter(
            image_count=0
        )
    # get all images that are not linked to any animal
    nr_unidentified_images = len(IbexImage.objects.filter(animal_id__isnull=True))

    return render(
        request,
        "core/animal_table.html",
        {
            "animals": animals,
            "no_id_count": nr_unidentified_images,
            "url_name": url_name,
        },
    )


@login_required
def unobserved_animal_view(request):
    # get all animals that are not featured in any images
    animals = Animal.objects.annotate(image_count=Count("ibeximage")).filter(
        image_count=0
    )
    return render(
        request,
        "core/animal_table.html",
        {"animals": animals},
    )


@login_required
def to_landmark_images_view(request):
    # get all images that have no animal ID
    user_id = request.user.id
    images_to_landmark = IbexImage.objects.filter(owner_id=user_id).filter(
        animal_id__isnull=True
    )
    return render(
        request,
        "core/to_landmark.html",
        {"images": images_to_landmark},
    )


@login_required
def landmark_horn_view(request, oid):
    image = get_object_or_404(IbexImage, id=oid)
    return render(
        request,
        "simple_landmarks/horn_landmark.html",
        {"image": image, "display_width": settings.LANDMARK_IMAGE_WIDTH},
    )


@login_required
def landmark_eye_view(request, oid):
    image = get_object_or_404(IbexImage, id=oid)

    x_horn_scaled, y_horn_scaled = utils.parse_coordinates(request)
    x_horn, y_horn = utils.scale_coordinate(
        x_horn_scaled, y_horn_scaled, image.width, settings.LANDMARK_IMAGE_WIDTH
    )
    # save horn-landmark for that image
    landmark_id = get_object_or_404(Landmark, label="horn_tip").id
    content_type = ContentType.objects.get_for_model(IbexImage)
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=landmark_id,
    )
    horn_landmark.x_coordinate = x_horn
    horn_landmark.y_coordinate = y_horn
    horn_landmark.save()

    # render eye_landmark page
    return render(
        request,
        "simple_landmarks/eye_landmark.html",
        {
            "image": image,
            "x_horn_scaled": x_horn_scaled,
            "y_horn_scaled": y_horn_scaled,
            "display_width": settings.LANDMARK_IMAGE_WIDTH,
        },
    )


@login_required
def finished_landmark_view(request, oid):
    image = get_object_or_404(IbexImage, id=oid)
    x_eye_scaled, y_eye_scaled = utils.parse_coordinates(request)
    x_eye, y_eye = utils.scale_coordinate(
        x_eye_scaled, y_eye_scaled, image.width, settings.LANDMARK_IMAGE_WIDTH
    )
    # save eye-landmark for that image
    eye_landmark_id = get_object_or_404(Landmark, label="eye_corner").id
    content_type = ContentType.objects.get_for_model(IbexImage)
    eye_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=eye_landmark_id,
    )
    eye_landmark.x_coordinate = x_eye
    eye_landmark.y_coordinate = y_eye
    eye_landmark.save()

    # render landmarks on image
    horn_landmark_id = get_object_or_404(Landmark, label="horn_tip").id
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=horn_landmark_id,
    )
    x_horn = horn_landmark.x_coordinate
    y_horn = horn_landmark.y_coordinate
    x_horn_scaled, y_horn_scaled = utils.scale_coordinate(
        x_horn, y_horn, settings.LANDMARK_IMAGE_WIDTH, image.width
    )
    return render(
        request,
        "simple_landmarks/finished_landmarks.html",
        {
            "image": image,
            "x_horn_scaled": x_horn_scaled,
            "y_horn_scaled": y_horn_scaled,
            "x_eye_scaled": x_eye_scaled,
            "y_eye_scaled": y_eye_scaled,
            "display_width": settings.LANDMARK_IMAGE_WIDTH,
            # "x_horn_scaled": x_horn,
            # "y_horn_scaled": y_horn,
            # "x_eye_scaled": x_eye,
            # "y_eye_scaled": y_eye,
            # "display_width": image.width,
        },
    )


@login_required
def chip_view(request, oid):
    image = get_object_or_404(IbexImage, id=oid)
    print("image -", image)
    chip_name = utils.get_chip_filename(image.file.name, settings.CHIP_FILETYPE)
    print("chip name -", chip_name)

    # Determine media storage environment
    if settings.ENVIRONMENT == "production" or settings.POSTGRES_LOCALLY == True:
        is_local = False
    else:
        is_local = True
    print("is_local -", is_local)

    if is_local:
        image_path = os.path.join(settings.MEDIA_ROOT, image.file.name)
        chip_url = os.path.join(os.path.split(image.url)[0], chip_name)
        chip_path = Path(os.path.join(os.path.split(image_path)[0], chip_name))

        # if a chip exists already, delete it before continuing
        if chip_path.is_file():
            chip_path.unlink()
            # also update database
            ibex_chip = get_object_or_404(IbexChip, ibex_image_id=image.id)
            ibex_chip.delete()
            print(
                "IbexChip already existed on local storage, deleted successfully before continueing."
            )

        # create new chip from original image and try to preserve all metadata
        shutil.copy2(image_path, chip_path)

        # load image
        img = utils.load_image(chip_path)

    else:
        # if a chip exists already, delete it before continuing
        try:
            ibex_chip = IbexChip.objects.get(ibex_image_id=image.id)
            # If the object is found, continue with your logic here
            print(f"IbexChip found in database with ibex_image_id: {image.id}")
            print("Deleting previous ibex chip media file on backblaze..")
            # chip_public_id = ibex_chip.file.name
            image_bucket_path = os.path.join(settings.AWS_LOCATION, image.file.name)
            chip_bucket_path = os.path.dirname(image_bucket_path)
            chip_bucket_path = os.path.join(chip_bucket_path, chip_name)
            # Check if the file exists in the B2 bucket
            file_exists = b2_utils.check_file_exists(chip_bucket_path)

            # If the file exists, proceed with deletion
            if file_exists:
                # Delete the file from Backblaze B2 bucket
                b2_utils.delete_files([chip_bucket_path])

                # Delete the associated IbexChip object from the database
                ibex_chip = get_object_or_404(IbexChip, ibex_image_id=image.id)
                ibex_chip.delete()
                print(
                    f"File {chip_bucket_path} deleted from B2 bucket and IbexChip deleted from the database."
                )
            else:
                print(
                    "IbexChip not deleted because the file was not found in the B2 bucket."
                )

        except IbexChip.DoesNotExist:
            # Handle the case where the object does not exist
            print("IbexChip does not exist already, continueing normally..")
            pass

        # Download the image from cloud
        bucket_file_path = os.path.join(settings.AWS_LOCATION, image.file.name)
        img_object = b2_utils.download_file(bucket_file_path=bucket_file_path)
        # Convert the response content to a NumPy array
        img_array = np.frombuffer(img_object, np.uint8)

        # Decode the image from the NumPy array
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to load image from cloud.")

        print("Downloaded image from cloud.")

    # load landmarks
    content_type = ContentType.objects.get_for_model(IbexImage)
    horn_landmark_id = get_object_or_404(Landmark, label="horn_tip").id
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=horn_landmark_id,
    )
    eye_landmark_id = get_object_or_404(Landmark, label="eye_corner").id
    eye_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=eye_landmark_id,
    )
    eyehorn_src = [
        [eye_landmark.x_coordinate, eye_landmark.y_coordinate],
        [horn_landmark.x_coordinate, horn_landmark.y_coordinate],
    ]

    # check animal side
    # flip image if it is right taged
    if image.side == "R":
        img = cv2.flip(img, 1)  # along x-axis = around y-axis
        # flip x-coordinates
        eyehorn_src[0][0] = utils.mirror_coordinate(eyehorn_src[0][0], image.width)
        eyehorn_src[1][0] = utils.mirror_coordinate(eyehorn_src[1][0], image.width)

    # calculate coordniates where horn and eye should be in the output image
    width_dst = settings.CHIP_WIDTH
    height_dst = settings.CHIP_HEIGHT
    eye_dst = (np.round(width_dst * 0.98), np.round(height_dst * 0.95))
    tip_dst = (np.round(width_dst * 0.98), np.round(height_dst * 0.05))
    eyehorn_dst = [eye_dst, tip_dst]

    # affine transform image
    tform = utils.similarityTransform(eyehorn_src, eyehorn_dst)
    # note, height and width are exchanged here because we want a
    # horizontal image first
    shape_dst = (width_dst, height_dst)
    img_transformed = cv2.warpAffine(img, tform, shape_dst)

    # save
    if is_local:
        cv2.imwrite(chip_path, cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR))
        print("path", chip_path)
        chip_file = os.path.join(os.path.split(str(image.file.name))[0], chip_name)
        print("file", chip_file)
        IbexChip.objects.create(file=chip_file, ibex_image_id=image.id)
        print("Chip saved locally using open-cv, database updated.")
        ibex_chip = get_object_or_404(IbexChip, ibex_image_id=image.id)
        print("Loaded ibex chip object")
    else:
        # Convert the image to the correct format for backblaze
        buffer = BytesIO()
        img_pil = Image.fromarray(cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR))
        img_pil.save(buffer, format="png")
        buffer.seek(0)

        # Use Django's FileField to handle the upload
        chip_content = ContentFile(buffer.getvalue())

        # Create the IbexChip instance
        ibex_chip = IbexChip(ibex_image_id=image.id)
        ibex_chip.file.save(chip_name, chip_content)
        print("Chip saved on Backblaze using Django's FileField.")
        chip_url = ibex_chip.file.url

    # Call the custom method to process and embed the chip
    utils.embed_new_chip(ibex_chip)

    # eye_x_scaled, eye_y_scaled = scale_coordinate(
    #     eye_landmark.x_coordinate,
    #     eye_landmark.y_coordinate,
    #     settings.LANDMARK_IMAGE_WIDTH,
    #     image.width,
    # )
    # horn_x_scaled, horn_y_scaled = scale_coordinate(
    #     horn_landmark.x_coordinate,
    #     horn_landmark.y_coordinate,
    #     settings.LANDMARK_IMAGE_WIDTH,
    #     image.width,
    # )

    return render(
        request,
        "simple_landmarks/chip.html",
        {"chip": chip_url, "side": image.side},
    )


def results_over_view(request):
    # images that have an animal associated
    identified_images = IbexImage.objects.filter(animal_id__isnull=False)
    return render(
        request,
        "core/results_overview.html",
        {"images": identified_images},
    )


def show_result_view(request, oid):
    query = get_object_or_404(IbexChip, id=oid)
    query_embedding = query.embedding.embedding

    # query chips of all previously known animals:
    # Step 1: Filter Animals that have related IbexImages
    # The distinct() call ensures that each Animal is only returned once, even if they have multiple images.
    animals_with_images = Animal.objects.filter(ibeximage__isnull=False).distinct()

    # Step 2: Query IbexChips related to those animals via the IbexImage model
    gallery_chips = IbexChip.objects.filter(ibex_image__animal__in=animals_with_images)

    known_animals = Animal.objects.all()
    if gallery_chips:
        gallery_embeddings = Embedding.objects.filter(ibex_chip_id__in=gallery_chips)
        # Extract all embedding vectors as a list of lists (or arrays)
        gallery_vectors = [i.embedding for i in gallery_embeddings]
        gallery_ids = [i.ibex_chip_id for i in gallery_embeddings]

        # Convert the list of embedding vectors to a NumPy array
        gallery_vectors_array = np.array(gallery_vectors)
        distances = utils.cdist_np(
            np.array([query_embedding]), gallery_vectors_array, metric="euclidean"
        )
        distances = distances[0]
        gallery_and_distances = zip(gallery_chips, distances)
        # Sort the zipped list based on the distance (second element in each tuple)
        sorted_gallery = sorted(gallery_and_distances, key=lambda x: x[1])
        top5_sorted_gallery = sorted_gallery[:5]
        # round distances
        top5_sorted_gallery = [
            (chip, round(distance, 2)) for chip, distance in top5_sorted_gallery
        ]

    else:
        top5_sorted_gallery = []

    threshold_distance = 9.3

    return render(
        request,
        "core/result.html",
        {
            "query_chip": query,
            "gallery_and_distances": top5_sorted_gallery,
            "threshold": threshold_distance,
            "known_animals": known_animals,
        },
    )


def rerun_view(request, oid):
    query = get_object_or_404(IbexChip, id=oid)
    query_embedding = query.embedding.embedding

    # query chips of all previously known animals:
    # Step 1: Filter Animals that have related IbexImages
    # The distinct() call ensures that each Animal is only returned once, even if they have multiple images.
    animals_with_images = Animal.objects.filter(ibeximage__isnull=False).distinct()
    # Step 2: Query IbexChips related to those animals via the IbexImage model
    # exclude the query chip that has already been run
    gallery_chips = IbexChip.objects.filter(
        ibex_image__animal__in=animals_with_images
    ).exclude(pk=oid)

    known_animals = Animal.objects.all()
    if gallery_chips:
        gallery_embeddings = Embedding.objects.filter(ibex_chip_id__in=gallery_chips)
        # Extract all embedding vectors as a list of lists (or arrays)
        gallery_vectors = [i.embedding for i in gallery_embeddings]
        gallery_ids = [i.ibex_chip_id for i in gallery_embeddings]

        # Convert the list of embedding vectors to a NumPy array
        gallery_vectors_array = np.array(gallery_vectors)
        distances = utils.cdist_np(
            np.array([query_embedding]), gallery_vectors_array, metric="euclidean"
        )
        distances = distances[0]
        gallery_and_distances = zip(gallery_chips, distances)
        # Sort the zipped list based on the distance (second element in each tuple)
        sorted_gallery = sorted(gallery_and_distances, key=lambda x: x[1])
        top5_sorted_gallery = sorted_gallery[:5]
        # round distances
        top5_sorted_gallery = [
            (chip, round(distance, 2)) for chip, distance in top5_sorted_gallery
        ]

    else:
        top5_sorted_gallery = []

    threshold_distance = 9.3

    return render(
        request,
        "core/result.html",
        {
            "query_chip": query,
            "gallery_and_distances": top5_sorted_gallery,
            "threshold": threshold_distance,
            "known_animals": known_animals,
        },
    )


@login_required
def created_animal_view(request, oid):
    query_chip = get_object_or_404(IbexChip, id=oid)

    # parse ibeximage filename from query chip object id
    filename = query_chip.ibex_image.name
    # get new animal id_code
    new_code = utils.generate_animal_id_code(filename)
    # check if there is a datetime in the filename from the exif data
    date_of_image = utils.parse_datetime_from_filename(
        filename
    )  # returns datetime.date object
    # create new animal
    if date_of_image:
        Animal.objects.create(id_code=new_code, capture_date=date_of_image)
    else:
        Animal.objects.create(id_code=new_code)
    # link image to that animal
    original_image = get_object_or_404(IbexImage, id=query_chip.ibex_image_id)
    original_image.animal = get_object_or_404(Animal, id_code=new_code)
    original_image.save()
    # get all images of a specific animal
    images = IbexImage.objects.filter(animal__id_code=new_code)

    return render(
        request,
        "core/animal_images.html",
        {"images": images, "animal_id_code": new_code},
    )


def test_view(request):
    from environ import Env

    env = Env()
    Env.read_env()
    ENVIRONMENT = env("ENVIRONMENT", default="production")

    # If not a POST request, just render the form
    return render(request, "test.html")
