from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.db.models.aggregates import Count
from django.contrib.contenttypes.models import ContentType

from core.models import IbexImage, Animal
from simple_landmarks.models import LandmarkItem


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


def test_view(request):
    content_type = ContentType.objects.get_for_model(IbexImage)

    queryset = LandmarkItem.objects.select_related("landmark").filter(
        content_type=content_type
    )
    return render(request, "core/test.html", {"queryset": queryset})
