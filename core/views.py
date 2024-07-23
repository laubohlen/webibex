import os
import cv2
import math
import shutil
import numpy as np

from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.db.models.aggregates import Count
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.decorators import login_required
from django.conf import settings

from core.models import IbexImage, IbexChip, Animal
from simple_landmarks.models import LandmarkItem, Landmark

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
    # link to django-filer admin page
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
def observed_animal_view(request):
    # get all animals that are linked to one or more images
    animals = Animal.objects.annotate(image_count=Count("ibeximage")).filter(
        image_count__gt=0
    )
    # get all images that are not linked to any animal
    nr_unidentified_images = len(IbexImage.objects.filter(animal_id__isnull=True))
    return render(
        request,
        "core/animal_table.html",
        {"animals": animals, "no_id_count": nr_unidentified_images},
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

    eye_x_scaled, eye_y_scaled = scale_coordinate(
        eye_landmark.x_coordinate,
        eye_landmark.y_coordinate,
        settings.LANDMARK_IMAGE_WIDTH,
        image.width,
    )
    horn_x_scaled, horn_y_scaled = scale_coordinate(
        horn_landmark.x_coordinate,
        horn_landmark.y_coordinate,
        settings.LANDMARK_IMAGE_WIDTH,
        image.width,
    )

    return render(
        request,
        "simple_landmarks/chip.html",
        {"chip": chip_url, "side": image.side},
    )


def test_view(request):
    queryset = IbexImage.objects.all()
    return render(request, "core/test.html", {"queryset": queryset})
