from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.db.models.aggregates import Count
from .models import IbexImage, Animal


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
    return render(
        request,
        "core/animal_table.html",
        {"animals": animals},
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
    queryset = Animal.objects.annotate(image_count=Count("ibeximage")).filter(
        image_count=1
    )
    return render(request, "core/test.html", {"queryset": queryset})
