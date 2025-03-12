import numpy as np

from django.conf import settings
from django.http import HttpResponseForbidden
from django.shortcuts import render, redirect, get_object_or_404
from django.core.exceptions import PermissionDenied
from django.db.models import Count, Q
from django.contrib.auth.decorators import login_required
from django.contrib.contenttypes.models import ContentType

from . import utils
from core.models import IbexImage, IbexChip, Animal, Embedding, Region, Location

from filer.models import Folder
from simple_landmarks.models import LandmarkItem, Landmark


def welcome_view(request):
    return render(request, "core/welcome.html")


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
        return redirect("unidentified-images")
    else:
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
def animal_images_owner_view(request, oid):
    # get all images of a specific animal
    images = IbexImage.objects.filter(animal_id=oid, owner=request.user)
    # in case images is empty, get the animal name
    if not images:
        animal_id_code = get_object_or_404(Animal, pk=oid).id_code
    else:
        animal_id_code = images.first().animal.id_code
    return render(
        request,
        "core/animal_images_owner.html",
        {"images": images, "animal_id_code": animal_id_code},
    )


@login_required
def animals_overview(request):
    observed_animals = Animal.objects.annotate(image_count=Count("ibeximage")).filter(
        image_count__gt=0
    )
    unobserved_animals = animals = Animal.objects.annotate(
        image_count=Count("ibeximage")
    ).filter(image_count=0)
    # get all images that are not linked to any animal
    nr_unidentified_images = len(IbexImage.objects.filter(animal_id__isnull=True))
    context = {
        "observed_animals": observed_animals,
        "unobserved_animals": unobserved_animals,
        "nr_unidentified_images": nr_unidentified_images,
    }
    return render(request, "core/animal_overview.html", context)


def save_landmarks_view(request):
    if request.method == "POST":
        image_id = request.POST.get("image-id")
        image = get_object_or_404(IbexImage, id=image_id)

        # get landmarks relative to displayed image size
        x_horn_scaled = int(request.POST.get("horn_x"))
        y_horn_scaled = int(request.POST.get("horn_y"))
        x_eye_scaled = int(request.POST.get("eye_x"))
        y_eye_scaled = int(request.POST.get("eye_y"))

        # calculate landmark back relative to original image size
        x_horn, y_horn = utils.scale_coordinate(
            x_horn_scaled, y_horn_scaled, image.width, settings.LANDMARK_IMAGE_WIDTH
        )
        x_eye, y_eye = utils.scale_coordinate(
            x_eye_scaled, y_eye_scaled, image.width, settings.LANDMARK_IMAGE_WIDTH
        )

        # save horn landmark
        landmark_id = get_object_or_404(Landmark, label="horn_tip").id
        content_type = ContentType.objects.get_for_model(IbexImage)
        landmark = get_object_or_404(
            LandmarkItem,
            content_type=content_type,
            object_id=image_id,
            landmark_id=landmark_id,
        )
        landmark.x_coordinate = x_horn
        landmark.y_coordinate = y_horn
        landmark.save()

        # save eye landmark
        landmark_id = get_object_or_404(Landmark, label="eye_corner").id
        content_type = ContentType.objects.get_for_model(IbexImage)
        landmark = get_object_or_404(
            LandmarkItem,
            content_type=content_type,
            object_id=image_id,
            landmark_id=landmark_id,
        )
        landmark.x_coordinate = x_eye
        landmark.y_coordinate = y_eye
        landmark.save()

        print("about to process chip")

        # crop, save and embed horn chip
        utils.process_horn_chip(image, x_horn, y_horn, x_eye, y_eye)

        # check if there is another image to landmark
        next_id_index = request.POST.get("next_id_index")
        if next_id_index not in (None, "None"):
            return multi_task_view(request)
        # if no next image, return where the user requested the task
        else:
            return redirect("unidentified-images")

    else:
        pass


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

    id_to_color = utils.id_color_mapping(top5_sorted_gallery)
    print(id_to_color)

    return render(
        request,
        "core/result.html",
        {
            "query_chip": query,
            "gallery_and_distances": top5_sorted_gallery,
            "threshold": threshold_distance,
            "known_animals": known_animals,
            "id_to_color": id_to_color,
        },
    )


def rerun_view(request, oid):
    """TODO: figure out way to best show the comparison that was shown
    during the identification. Maybe just store ID's of images that where shown
    together with the embedding? But what if images get deleted that where previously
    used?
    """
    query = get_object_or_404(IbexChip, id=oid)
    query_embedding = query.embedding.embedding

    # query chips of all previously known animals:
    # Step 1: Filter Animals that have related IbexImages
    # The distinct() call ensures that each Animal is only returned once, even if they have multiple images.
    animals_with_images = Animal.objects.filter(ibeximage__isnull=False).distinct()
    # Step 2: Query IbexChips related to those animals via the IbexImage model
    # and exclude the query chip that has already been run
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

    return redirect("images-overview")


@login_required
def save_region(request):
    if request.method == "POST":
        region_name = request.POST.get("region-name")
        radius = request.POST.get("radius")
        latitude = request.POST.get("latitude")
        longitude = request.POST.get("longitude")
        region_id = request.POST.get("region-id")  # Will be present if updating

        # If a region id is present, update the existing region.
        if region_id:
            region = get_object_or_404(Region, pk=region_id, owner=request.user)

            # Optionally, if the name is being changed, check for duplicates.
            if (
                region.name != region_name
                and Region.objects.filter(name__iexact=region_name).exists()
            ):
                print("Region name already exists")
                return render(
                    request,
                    "core/region_create_naming_error.html",
                    {"name_taken": region_name, "radius": radius},
                )

            # Update the region.
            region.name = region_name
            region.radius = radius
            region.origin_latitude = latitude
            region.origin_longitude = longitude
            region.save()

            return render(request, "core/region_read.html", {"region": region})

        else:  # Otherwise, we're creating a new region.
            # check if region name already exists
            if Region.objects.filter(name__iexact=region_name).exists():
                print("Region name already exists")
                return render(
                    request,
                    "core/region_create_naming_error.html",
                    {"name_taken": region_name, "radius": radius},
                )

        # Create the new region.
        Region.objects.create(
            name=region_name,
            radius=radius,
            origin_latitude=latitude,
            origin_longitude=longitude,
            owner=request.user,
        )
        return redirect("region-overview")
    # For GET requests, simply return to overview
    return render(request, "core/region_overview.html")


@login_required
def create_region(request):
    return render(request, "core/region_create.html")


@login_required
def read_region(request, oid):
    region = get_object_or_404(Region, pk=oid)
    return render(request, "core/region_read.html", {"region": region})


@login_required
def delete_region(request, oid):
    region = get_object_or_404(Region, pk=oid, owner=request.user)
    if request.method == "POST":
        if region.owner != request.user:
            return HttpResponseForbidden("You are not allowed to delete this object.")
        region.delete()
        return redirect("region-overview")
    return render(request, "core/region_delete.html", {"region": region})


@login_required
def update_region(request, oid):
    region = get_object_or_404(Region, pk=oid)
    if region.owner != request.user:
        return HttpResponseForbidden("You are not allowed to edit this object.")

    return render(request, "core/region_update.html", {"region": region})


@login_required
def region_overview(request):
    region_qs = Region.objects.all()
    return render(request, "core/region_overview.html", {"region_qs": region_qs})


@login_required
def save_image_location(request):
    if request.method == "POST":
        region_id = request.POST.get("region-id")
        latitude = request.POST.get("latitude")
        longitude = request.POST.get("longitude")
        location_id = request.POST.get("location-id")
        location_source = request.POST.get("location-source")
        image_id = request.POST.get("image-id")

        # Update location
        region = get_object_or_404(Region, pk=region_id)
        location = get_object_or_404(Location, pk=location_id)
        location.latitude = latitude
        location.longitude = longitude
        location.source = location_source
        location.region = region
        location.save()
        print(f"Location updated.")

        next_id_index = request.POST.get("next_id_index")
        post_task_dst = request.GET.get("next")
        print("post-task destination from url next-parameter:", post_task_dst)
        if next_id_index not in (None, "None"):
            return multi_task_view(request)
        # if no next image, try to return where the user requested the task
        elif post_task_dst not in (None, "None", ""):
            return redirect(post_task_dst)
        else:
            return redirect("images-overview")

    pass


@login_required
def create_loaction(request, oid):
    image = get_object_or_404(IbexImage, id=oid)
    # check if image has a location associated
    image_location = image.location
    if not image_location:
        init_location = Location.objects.create()
        image.location = init_location
        image.save()
        image_location = image.location
    location_id = image_location.id
    # check if GPS is available, else return None
    if None in [image_location.latitude, image_location.longitude]:
        image_location = None
    region_qs = Region.objects.filter(owner=request.user)
    return render(
        request,
        "core/location_create.html",
        {
            "image": image,
            "image_location": image_location,
            "location_id": location_id,
            "regions": region_qs,
        },
    )


@login_required
def images_overview(request):
    # get all animals that are linked to one or more images
    animals = Animal.objects.annotate(
        image_count=Count("ibeximage", filter=Q(ibeximage__owner=request.user))
    ).filter(image_count__gt=0)
    # get all images that are not linked to any animal
    nr_unidentified_images = len(
        IbexImage.objects.filter(animal_id__isnull=True, owner=request.user)
    )

    return render(
        request,
        "core/images_overview.html",
        {
            "animals": animals,
            "no_id_count": nr_unidentified_images,
        },
    )


@login_required
def image_read(request, oid):
    image = get_object_or_404(IbexImage, pk=oid)

    return render(
        request,
        "core/image_read_new.html",
        {
            "image": image,
        },
    )


@login_required
def image_update(request, oid):
    image = get_object_or_404(IbexImage, pk=oid)

    # get landmarks for  the image
    eye_landmark_id = get_object_or_404(Landmark, label="eye_corner").id
    horn_landmark_id = get_object_or_404(Landmark, label="horn_tip").id
    content_type = ContentType.objects.get_for_model(IbexImage)
    eye_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=eye_landmark_id,
    )
    horn_landmark = get_object_or_404(
        LandmarkItem,
        content_type=content_type,
        object_id=oid,
        landmark_id=horn_landmark_id,
    )

    x_eye_percent, y_eye_percent = utils.percentage_coordinate(
        eye_landmark.x_coordinate,
        eye_landmark.y_coordinate,
        image.width,
        image.height,
    )

    x_horn_percent, y_horn_percent = utils.percentage_coordinate(
        horn_landmark.x_coordinate,
        horn_landmark.y_coordinate,
        image.width,
        image.height,
    )

    if request.method == "POST":
        if image.owner != request.user:
            return HttpResponseForbidden("You are not allowed to edit this object.")
        side = request.POST.get("horn-side")
        image.side = side
        image.save()
        print("Image updated")
        return redirect("read-image", oid=image.id)

    return render(
        request,
        "core/image_update_new.html",
        {
            "image": image,
            "x_horn_percent": x_horn_percent,
            "y_horn_percent": y_horn_percent,
            "x_eye_percent": x_eye_percent,
            "y_eye_percent": y_eye_percent,
        },
    )


@login_required
def image_delete(request, oid):
    image = get_object_or_404(IbexImage, pk=oid)
    if request.method == "POST":
        if image.owner != request.user:
            return HttpResponseForbidden("You are not allowed to delete this object.")
        image.delete()
        print("image deleted")
        return redirect("images-overview")
    return render(request, "core/image_delete_new.html", {"image": image})


@login_required
def image_upload(request):
    if request.method == "POST":
        files = request.FILES.getlist("images")
        side = request.POST.get("horn-side")
        if side == "later":
            side = None
        # Retrieve the user's main folder (created by signal)
        upload_folder = Folder.objects.get(
            name=f"{request.user.username}_files", owner=request.user
        )

        for f in files:
            IbexImage.objects.create(
                original_filename=f.name,
                file=f,
                folder=upload_folder,
                side=side,
                owner=request.user,
            )

        return redirect("images-overview")  # Redirect back
    return render(request, "core/image_upload.html")


@login_required
def unidentified_images_view(request):
    if request.method == "POST":
        return multi_task_view(request)

    else:
        # get all images that are not linked to any animal
        unidentified_images = IbexImage.objects.filter(
            animal_id__isnull=True, owner=request.user
        )
        return render(
            request,
            "core/unidentified_images.html",
            {"images": unidentified_images},
        )


@login_required
def multi_task_view(request):
    if request.method == "POST":
        # what task is asked
        task = request.POST.get("task")

        # what files to use for the task
        selected_img_ids = request.POST.getlist("selected-files")
        selected_img_ids = [int(i) for i in selected_img_ids[0].split(",")]

        if task == "tag_left":
            IbexImage.objects.filter(id__in=selected_img_ids).update(side="L")
            return redirect("unidentified-images")
        elif task == "tag_right":
            IbexImage.objects.filter(id__in=selected_img_ids).update(side="R")
            return redirect("unidentified-images")
        elif task == "tag_other":
            IbexImage.objects.filter(id__in=selected_img_ids).update(side="O")
            return redirect("unidentified-images")

        else:
            # current image
            try:
                current_id_index = int(request.POST.get("next_id_index", "0"))
            except (TypeError, ValueError):
                current_id_index = 0
            current_image_id = selected_img_ids[current_id_index]
            image = get_object_or_404(IbexImage, pk=current_image_id)

            # next image
            if current_id_index + 1 < len(selected_img_ids):
                next_id_index = current_id_index + 1
                next_image_id = selected_img_ids[next_id_index]
            else:
                next_id_index = None
                next_image_id = None

            # convert list of ints back to string
            selected_img_ids = ",".join(map(str, selected_img_ids))

            context = {
                "image": image,
                "current_id_index": current_id_index,
                "next_image_id": next_image_id,
                "next_id_index": next_id_index,
                "selected_img_ids": selected_img_ids,
            }

            # get the correct url-pattern depending on the task
            template, task_context = utils.multi_task_url(
                task, image=image, user=request.user
            )

            # concatenate context
            if task_context:
                context = {**context, **task_context}

            return render(request, template, context)

    else:
        unidentified_images = IbexImage.objects.filter(animal_id__isnull=True)
        return render(
            request,
            "core/unidentified_images.html",
            {"images": unidentified_images},
        )


@login_required
def test_view(request):
    return render(request, "core/test.html")
