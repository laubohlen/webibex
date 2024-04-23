from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponseRedirect
from .models import IbexImage, Animal


def welcome_view(request):
    return render(request, "core/welcome.html")


def upload_view(request):
    # link to django-filer admin page
    url = reverse("admin:filer_folder_changelist")
    return HttpResponseRedirect(url)


def animal_view(request):
    # get all images that have an animal linked
    identified_images = IbexImage.objects.filter(animal_id__isnull=False).values_list(
        "animal_id"
    )
    # get all images that have no animal linked
    # unidentified_images = Image.objects.filter(animal_id__isnull=True).values_list(
    #     "animal_id"
    # )
    # get all animals that are linked to one or more images
    animals = Animal.objects.filter(pk__in=identified_images)
    return render(
        request,
        "core/animal_list.html",
        {"animals": animals},
    )
