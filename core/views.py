import os
import re
import cv2
import math
import shutil
import datetime
import numpy as np

from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect, HttpResponse
from django.db.models.aggregates import Count
from django.db.models import Exists, OuterRef
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.decorators import login_required
from django.conf import settings

from core.models import IbexImage, IbexChip, Animal, Embedding
from simple_landmarks.models import LandmarkItem, Landmark
from filer.models import Folder

from pathlib import Path
from PIL import Image


# snipet from https://github.com/krasch/simple_landmarks
# coordinates are sent as slightly weird URL parameters (e.g. 0.png?214,243)
# parse them, will crash server if they are coming in unexpected format
def parse_coordinates(request):
    keys = list(request.GET.keys())
    assert len(keys) == 1
    coordinates = keys[0]

    assert len(coordinates.split(",")) == 2
    x, y = coordinates.split(",")
    x = int(x)
    y = int(y)
    return x, y


# image was not displayed in original size -> need to convert the coordinates
def scale_coordinate(x: int, y: int, dst_image_width: int, src_image_width: int):
    scale = dst_image_width / src_image_width
    return round(x * scale), round(y * scale)


# mirror coordinate along the x-axis when right horn is flipped to be normalised as a left horn
def mirror_coordinate(x: int, src_image_width: int):
    return round(src_image_width - x)


def welcome_view(request):
    return render(request, "core/welcome.html")


def upload_view(request):
    # link to django-filer main folder of the user istead of
    # the root folder of the user because the folder hierarchy isn't displayed correctly
    user = request.user
    main_folder_name = f"_{user.username}_files"
    main_user_folder = get_object_or_404(Folder, name=main_folder_name, owner=user)
    # Construct the URL to the folder's listing page in the admin
    url = reverse(
        "admin:filer-directory_listing", kwargs={"folder_id": main_user_folder.id}
    )
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
        img = IbexImage.objects.get(ibexchip=query_chip_id)
        img.animal = Animal.objects.get(pk=oid)
        img.save()
        print("Saved selected animal to IbexImage.")
    else:
        pass

    # get all images of a specific animal
    images = IbexImage.objects.filter(animal_id=oid)
    # in case images is empty, get the animal name
    if not images:
        animal_id_code = Animal.objects.get(pk=oid).id_code
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
        animal_id_code = Animal.objects.get(pk=oid).id_code
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
    images_to_landmark = IbexImage.objects.filter(animal_id__isnull=True)
    return render(
        request,
        "core/to_landmark.html",
        {"images": images_to_landmark},
    )


@login_required
def landmark_horn_view(request, oid):
    image = IbexImage.objects.filter(id=oid).first()
    return render(
        request,
        "simple_landmarks/horn_landmark.html",
        {"image": image, "display_width": settings.LANDMARK_IMAGE_WIDTH},
    )


@login_required
def landmark_eye_view(request, oid):
    image = IbexImage.objects.filter(id=oid).first()

    x_horn_scaled, y_horn_scaled = parse_coordinates(request)
    x_horn, y_horn = scale_coordinate(
        x_horn_scaled, y_horn_scaled, image.width, settings.LANDMARK_IMAGE_WIDTH
    )
    # save horn-landmark for that image
    landmark_id = Landmark.objects.get(label="horn_tip").id
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
    image = IbexImage.objects.filter(id=oid).first()
    x_eye_scaled, y_eye_scaled = parse_coordinates(request)
    x_eye, y_eye = scale_coordinate(
        x_eye_scaled, y_eye_scaled, image.width, settings.LANDMARK_IMAGE_WIDTH
    )
    # save eye-landmark for that image
    eye_landmark_id = Landmark.objects.get(label="eye_corner").id
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
    horn_landmark_id = Landmark.objects.get(label="horn_tip").id
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=horn_landmark_id,
    )
    x_horn = horn_landmark.x_coordinate
    y_horn = horn_landmark.y_coordinate
    x_horn_scaled, y_horn_scaled = scale_coordinate(
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


# load an image as an rgb numpy array
def load_image(filename):
    # load image from file
    cv2image = cv2.imread(filename)
    # convert to RGB
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)  # cv2 loads as BGR
    return cv2image


def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = (
        c60 * (inPts[0][0] - inPts[1][0])
        - s60 * (inPts[0][1] - inPts[1][1])
        + inPts[1][0]
    )
    yin = (
        s60 * (inPts[0][0] - inPts[1][0])
        + c60 * (inPts[0][1] - inPts[1][1])
        + inPts[1][1]
    )

    inPts.append([round(xin), round(yin)])

    xout = (
        c60 * (outPts[0][0] - outPts[1][0])
        - s60 * (outPts[0][1] - outPts[1][1])
        + outPts[1][0]
    )
    yout = (
        s60 * (outPts[0][0] - outPts[1][0])
        + c60 * (outPts[0][1] - outPts[1][1])
        + outPts[1][1]
    )

    outPts.append([round(xout), round(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))

    return tform[0]


def get_chip_filename(filename: str, dst_ext: str):  # path of file or filename
    chip_name = os.path.split(filename)[1]
    name, ext = os.path.splitext(chip_name)
    chip_name = name + "_chip" + "." + dst_ext
    return chip_name


@login_required
def chip_view(request, oid):
    image = IbexImage.objects.filter(id=oid).first()
    image_path = os.path.join(settings.MEDIA_ROOT, image.file.name)
    chip_name = get_chip_filename(image.file.name, settings.CHIP_FILETYPE)
    chip_url = os.path.join(os.path.split(image.url)[0], chip_name)
    chip_path = Path(os.path.join(os.path.split(image_path)[0], chip_name))

    # if a chip exists already, delete it before continuing
    if chip_path.is_file():
        chip_path.unlink()
        # also update database
        ibex_chip = IbexChip.objects.filter(ibex_image_id=image.id)
        ibex_chip.delete()

    # create new chip from original image and try to preserve all metadata
    shutil.copy2(image_path, chip_path)

    # load image
    img = load_image(chip_path)

    # load landmarks
    content_type = ContentType.objects.get_for_model(IbexImage)
    horn_landmark_id = Landmark.objects.get(label="horn_tip").id
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=horn_landmark_id,
    )
    eye_landmark_id = Landmark.objects.get(label="eye_corner").id
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
        eyehorn_src[0][0] = mirror_coordinate(eyehorn_src[0][0], image.width)
        eyehorn_src[1][0] = mirror_coordinate(eyehorn_src[1][0], image.width)

    # calculate coordniates where horn and eye should be in the output image
    width_dst = settings.CHIP_WIDTH
    height_dst = settings.CHIP_HEIGHT
    eye_dst = (np.round(width_dst * 0.98), np.round(height_dst * 0.95))
    tip_dst = (np.round(width_dst * 0.98), np.round(height_dst * 0.05))
    eyehorn_dst = [eye_dst, tip_dst]

    # affine transform image
    tform = similarityTransform(eyehorn_src, eyehorn_dst)
    # note, height and width are exchanged here because we want a
    # horizontal image first
    shape_dst = (width_dst, height_dst)
    img_transformed = cv2.warpAffine(img, tform, shape_dst)

    # save
    cv2.imwrite(chip_path, cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR))

    # update database
    chip_file = os.path.join(os.path.split(str(image.file.name))[0], chip_name)
    IbexChip.objects.create(file=chip_file, ibex_image_id=image.id)

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


def all_diffs_np(a, b):
    """
    Returns a NumPy array of all combinations of a - b.

    Args:
        a (2D array): A batch of vectors shaped (B1, F).
        b (2D array): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in `b`,
        will be of shape (B1, B2, F).
    """
    return np.expand_dims(a, axis=1) - np.expand_dims(b, axis=0)


def cdist_np(a, b, metric="euclidean"):
    """
    Similar to scipy.spatial's cdist, but implemented in NumPy.

    Args:
        a (2D array): The left-hand side, shaped (B1, F).
        b (2D array): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in `b`,
        will be of shape (B1, B2).
    """
    a = a.astype(np.float32)  # Ensure float32 precision
    b = b.astype(np.float32)  # Ensure float32 precision
    diffs = all_diffs_np(a, b)

    if metric == "sqeuclidean":
        # Squared Euclidean distance
        return np.sum(np.square(diffs), axis=-1)
    elif metric == "euclidean":
        # Euclidean distance
        return np.sqrt(
            np.sum(np.square(diffs), axis=-1) + 1e-12
        )  # Adding a small epsilon for numerical stability
    elif metric == "cityblock":
        # Manhattan or L1 distance
        return np.sum(np.abs(diffs), axis=-1)
    else:
        raise NotImplementedError(
            f"The following metric is not implemented by `cdist` yet: {metric}"
        )


def results_over_view(request):
    # images that are linked from Embedding model
    chips = IbexChip.objects.filter(embedding__isnull=False)
    return render(request, "core/results_overview.html", {"chips": chips})


def show_result_view(request, oid):
    query = IbexChip.objects.filter(id=oid).first()
    query_embedding = query.embedding.embedding

    # query chips of all previously known animals:
    # Step 1: Filter Animals that have related IbexImages
    # The distinct() call ensures that each Animal is only returned once, even if they have multiple images.
    animals_with_images = Animal.objects.filter(ibeximage__isnull=False).distinct()

    # Step 2: Query IbexChips related to those animals via the IbexImage model
    gallery_chips = IbexChip.objects.filter(ibex_image__animal__in=animals_with_images)

    # gallery_chips = IbexChip.objects.exclude(id=oid)
    known_animals = Animal.objects.all()
    if gallery_chips:
        gallery_embeddings = Embedding.objects.filter(ibex_chip_id__in=gallery_chips)
        # Extract all embedding vectors as a list of lists (or arrays)
        gallery_vectors = [i.embedding for i in gallery_embeddings]
        gallery_ids = [i.ibex_chip_id for i in gallery_embeddings]

        # Convert the list of embedding vectors to a NumPy array
        gallery_vectors_array = np.array(gallery_vectors)
        distances = cdist_np(
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


def generate_animal_id_code(filename: str):
    # ensure filename is only the basename and not the file path
    filename = os.path.basename(filename)
    # get location and year indicators e.g. 'PNGP24'
    prefix = filename.split("_")[0]
    # find all newly generated animal codes, earlier codes don't contain "_"
    new_animals = Animal.objects.filter(id_code__contains="_")
    if new_animals:
        # convert to list
        previous_generated_codes = [i.id_code for i in new_animals]
        # Regular expression pattern to find a three-digit number
        pattern = r"\d{3}"
        code_number_list = [
            re.findall(pattern, i) for i in previous_generated_codes
        ]  # returns list of list
        # convert to actual numbers
        code_number_list = [int(i[0]) for i in code_number_list]
        largest_number = max(code_number_list)
        new_code = f"{prefix}_{largest_number+1:03}"  # -> 'prefix_014'
    # first new animal
    else:
        id_number = 1
        new_code = f"{prefix}_{id_number:03}"  # -> 'prefix_001'

    return new_code


def parse_datetime_from_filename(filename: str):
    # Ensure filename is only the basename and not the file path
    filename = os.path.basename(filename)

    # Check if the filename contains the string "noexifdata"
    if "noexifdata" in filename:
        return None

    # Regular expression to match the datetime string format: yy_mm_dd_HHMMSS
    datetime_pattern = r"\d{2}_\d{2}_\d{2}_\d{6}"

    # Search for the pattern in the filename
    match = re.search(datetime_pattern, filename)
    if match:
        datetime_str = match.group()  # Extract the matched datetime string
        try:
            # Parse the datetime string into a datetime object
            datetime_obj = datetime.datetime.strptime(datetime_str, "%y_%m_%d_%H%M%S")
            # Return the date part of the datetime object
            return datetime_obj.date()
        except ValueError:
            # If parsing fails, return None
            return None

    # If no valid datetime string is found, return None
    return None


@login_required
def created_animal_view(request, oid):
    query_chip = IbexChip.objects.filter(id=oid).first()
    # parse ibeximage filename from query chip object id
    filename = query_chip.ibex_image.name
    # get new animal id_code
    new_code = generate_animal_id_code(filename)
    # check if there is a datetime in the filename from the exif data
    date_of_image = parse_datetime_from_filename(
        filename
    )  # returns datetime.date object
    # create new animal
    if date_of_image:
        Animal.objects.create(id_code=new_code, capture_date=date_of_image)
    else:
        Animal.objects.create(id_code=new_code)
    # link image to that animal
    original_image = IbexImage.objects.get(id=query_chip.ibex_image_id)
    original_image.animal = Animal.objects.get(id_code=new_code)
    original_image.save()
    # get all images of a specific animal
    images = IbexImage.objects.filter(animal__id_code=new_code)

    return render(
        request,
        "core/animal_images.html",
        {"images": images, "animal_id_code": new_code},
    )


def test_view(request):
    if request.method == "POST":
        print("**********")
        selected_image = request.POST.get("selectedImage")
        print(selected_image)

        if selected_image:
            # Process the selected image value here
            # For example, you could save it to the database or perform some logic
            return HttpResponse(f"Selected Animal: {selected_image}")
        else:
            selected_image = None
            return HttpResponse("No Animal was selected.")

    # If not a POST request, just render the form
    return render(request, "test.html", {"selected_image": selected_image})
