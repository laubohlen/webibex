from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.db.models.aggregates import Count
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.decorators import login_required

from core.models import IbexImage, Animal
from simple_landmarks.models import LandmarkItem, Landmark


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
    # get all images that dont feature any landmarks AND are not featured to any animal
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
        {"image": image},
    )


@login_required
def landmark_eye_view(request, oid):
    x_horn, y_horn = parse_coordinates(request)

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
    image = IbexImage.objects.filter(id=oid).first()
    return render(
        request,
        "simple_landmarks/eye_landmark.html",
        {"image": image, "horn_landmark": horn_landmark},
    )


@login_required
def finished_landmark_view(request, oid):
    x_eye, y_eye = parse_coordinates(request)

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
    image = IbexImage.objects.filter(id=oid).first()
    horn_landmark_id = Landmark.objects.get(label="horn_tip").id
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=horn_landmark_id,
    )
    image_ratio = image.width / image.height
    # displayed_width = 1500  # set at base.html > .fixedContainer
    # displayed_height = displayed_width / image_ratio  # set to auto
    return render(
        request,
        "simple_landmarks/finished_landmarks.html",
        {"image": image, "horn_landmark": horn_landmark, "eye_landmark": eye_landmark},
    )


def test_view(request):
    content_type = ContentType.objects.get_for_model(IbexImage)

    queryset = LandmarkItem.objects.select_related("landmark").filter(
        content_type=content_type,
        object_id=198,
    )
    return render(request, "core/test.html", {"queryset": queryset})
