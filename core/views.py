from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.db.models.aggregates import Count
from django.contrib.contenttypes.models import ContentType

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


def unidentified_images_view(request):
    # get all images that are not linked to any animal
    unidentified_images = IbexImage.objects.filter(animal_id__isnull=True)
    return render(
        request,
        "core/unidentified_images.html",
        {"images": unidentified_images},
    )


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


def to_landmark_images_view(request):
    # get all images that dont feature any landmarks AND are not featured to any animal
    images_to_landmark = IbexImage.objects.filter(animal_id__isnull=True)
    return render(
        request,
        "core/to_landmark.html",
        {"images": images_to_landmark},
    )


def landmark_horn_view(request, oid):
    image = IbexImage.objects.filter(id=oid).first()
    return render(
        request,
        "simple_landmarks/horn_landmark.html",
        {"image": image},
    )


def landmark_eye_view(request, oid):
    x_horn, y_horn = parse_coordinates(request)

    # save horn-landmark for that image
    landmark_id = Landmark.objects.get(label="horn_tip").id
    content_type = ContentType.objects.get_for_model(IbexImage)
    landmark_item = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=landmark_id,
    )
    landmark_item.x_coordinate = x_horn
    landmark_item.y_coordinate = y_horn
    landmark_item.save()

    # render eye_landmark page
    image = IbexImage.objects.filter(id=oid).first()
    return render(
        request,
        "simple_landmarks/eye_landmark.html",
        {"image": image},
    )


def finished_landmark_view(request, oid):
    x_eye, y_eye = parse_coordinates(request)

    # save eye-landmark for that image
    landmark_id = Landmark.objects.get(label="eye_corner").id
    content_type = ContentType.objects.get_for_model(IbexImage)
    landmark_item = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=landmark_id,
    )
    landmark_item.x_coordinate = x_eye
    landmark_item.y_coordinate = y_eye
    landmark_item.save()

    # return view of the landmarks on the image
    # together with confirmation button
    pass


def test_view(request):
    content_type = ContentType.objects.get_for_model(IbexImage)

    queryset = LandmarkItem.objects.select_related("landmark").filter(
        content_type=content_type,
        object_id=198,
    )
    return render(request, "core/test.html", {"queryset": queryset})
