import os
import re
import cv2
import math
import shutil
import base64
import requests
import datetime
import numpy as np

from django.urls import resolve
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.core.files.base import ContentFile

from core.models import Animal, Embedding, IbexChip, Region

from io import BytesIO
from PIL import Image
from environ import Env
from pathlib import Path
from urllib.parse import urlparse
from geopy.distance import distance

from . import b2_utils

# Initialize environment variables
env = Env()
Env.read_env()

ENVIRONMENT = env("ENVIRONMENT", default="production")

# model_is_local = settings.ENVIRONMENT != "production" and not settings.GCP_MODEL_LOCALLY
# if model_is_local:
#     import tensorflow as tf

#     print(tf.__version__)

_tf = None


def get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf  # type: ignore (supress VSCode warning)

        print(tf.__version__)
        _tf = tf
    return _tf


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


def percentage_coordinate(x: int, y: int, src_image_width: int, src_image_height: int):
    left_percentage = (x / src_image_width) * 100
    top_percentage = (y / src_image_height) * 100
    return left_percentage, top_percentage


# mirror coordinate along the x-axis when right horn is flipped to be normalised as a left horn
def mirror_coordinate(x: int, src_image_width: int):
    return round(src_image_width - x)


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


# from https://github.com/VisualComputingInstitute/triplet-reid loss.py
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


# from https://github.com/VisualComputingInstitute/triplet-reid loss.py
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


def endpoint_inference(
    input_b64_img,
    endpoint_id: str = env("RUNPOD_ENDPOINT_ID"),
    endpoint_api_key: str = env("RUNPOD_API_KEY"),
):
    # Make the POST request to the RunPod endpoint
    endpoint_url = f"https://api.runpod.ai/v2/{env("RUNPOD_ENDPOINT_ID")}/runsync"
    headers = {
        "accept": "application/json",
        "authorization": env("RUNPOD_API_KEY"),
        "content-type": "application/json",
    }

    try:
        response = requests.post(endpoint_url, headers=headers, json=input_b64_img)
        response.raise_for_status()  # Raise an error for HTTP status codes >= 400
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to reach RunPod endpoint: {e}")

    # Parse the response from the endpoint
    response_data = response.json()
    if "error" in response_data:
        raise ValueError(f"RunPod error: {response_data['error']}")

    output = response_data.get("output").get("output").get("output_tensor")[0]
    if not output:
        raise ValueError("No output received from RunPod endpoint.")

    print("Embedded on endpoint.")
    return output


def embed_new_chip(ibex_chip):
    chip_size = (288, 144)

    # Determine if working locally or in production
    database_is_local = not (
        settings.ENVIRONMENT == "production" or settings.POSTGRES_LOCALLY == True
    )
    model_is_local = not (
        settings.ENVIRONMENT == "production" or settings.ENDPOINT_LOCALLY == True
    )

    if (not database_is_local) and (not model_is_local):
        # get image from cloud storage and run on embedding endpoint as in production
        chip_bucket_path = os.path.join(settings.AWS_LOCATION, ibex_chip.file.name)
        img_object = b2_utils.download_file(bucket_file_path=chip_bucket_path)
        if img_object is None:
            raise ValueError("Failed to download image from Backblaze B2.")
        # Convert the downloaded content (bytes) to a NumPy array
        img_array = np.frombuffer(img_object, np.uint8)
        # Decode the image from the NumPy array to check that it's valid
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                "Failed to decode image. The file may be corrupted or invalid."
            )

        chip_base64 = base64.b64encode(img_object).decode("utf-8")
        print("decoded image as base64")
        # Prepare the instance dictionary to match the model's expected input schema
        model_input = {"input": {"b64": chip_base64}}
        output = endpoint_inference(
            input_b64_img=model_input,
        )
        print("Embedded on model endpoint.")

    elif database_is_local and model_is_local:
        tf = get_tf()
        # Everything runs locally (complete dev environment)
        chip_path = os.path.join(settings.MEDIA_ROOT, ibex_chip.file.name)
        chip_bytes = tf.io.read_file(chip_path)
        chip_image = tf.image.decode_jpeg(chip_bytes, channels=3)
        print("Image loaded from local storage.")
        chip_resized = tf.image.resize(chip_image, chip_size)
        chip_expanded = tf.expand_dims(chip_resized, axis=0)
        model = tf.saved_model.load("core/embedding_model/")
        embedder = model.signatures["serving_default"]
        output = embedder(chip_expanded)["output_tensor"].numpy().tolist()[0]
        print("Embedded on local model.")

    elif (not database_is_local) and model_is_local:
        tf = get_tf()
        # get image from cloud storage and run with local model
        chip_bucket_path = os.path.join(settings.AWS_LOCATION, ibex_chip.file.name)
        img_object = b2_utils.download_file(bucket_file_path=chip_bucket_path)
        if img_object is None:
            raise ValueError("Failed to download image from Backblaze B2.")
        # Convert the downloaded content (bytes) to a NumPy array
        chip_array = np.frombuffer(img_object, np.uint8)
        # Decode the image from the NumPy array to check that it's valid
        chip_image = cv2.imdecode(chip_array, cv2.IMREAD_COLOR)
        chip_resized = tf.image.resize(chip_image, chip_size)
        chip_expanded = tf.expand_dims(chip_resized, axis=0)
        model = tf.saved_model.load("core/embedding_model/")
        embedder = model.signatures["serving_default"]
        output = embedder(chip_expanded)["output_tensor"].numpy().tolist()[0]
        print("Embedded on local model.")

    else:
        # get image from local storage but run on embedding endpoint
        chip_path = os.path.join(settings.MEDIA_ROOT, ibex_chip.file.name)
        with open(chip_path, "rb") as img_file:
            chip_image = img_file.read()
        print("Image loaded from local storage.")
        chip_base64 = base64.b64encode(chip_image).decode("utf-8")
        # Prepare the instance dictionary to match the model's expected input schema
        model_input = {"input": {"b64": chip_base64}}
        output = endpoint_inference(
            input_b64_img=model_input,
        )
        print("Embedded on model endpoint.")

    # Save the embedding to the database
    Embedding.objects.create(ibex_chip=ibex_chip, embedding=output)
    print("Embedding created and saved.")


# return img
def multi_task_url(tool, image=None, user=None):
    if tool == "view":
        print("Viewing")
        return "core/multi_view.html", None
    elif tool == "locate":
        print("Locating images")
        image_location = image.location
        location_id = image_location.id
        # check if GPS is available, else return None
        if None in [image_location.latitude, image_location.longitude]:
            image_location = None
        region_qs = Region.objects.filter(owner=user)
        template = "core/multi_location_create.html"
        task_specific_context = {
            "image_location": image_location,
            "location_id": location_id,
            "regions": region_qs,
        }
        return template, task_specific_context
    elif tool == "landmark":
        print("Landmarking images")
        template = "simple_landmarks/multi_landmarking.html"
        task_specific_context = {"display_width": settings.LANDMARK_IMAGE_WIDTH}
        return template, task_specific_context
    elif tool == "delete":
        print("Deleting images")
    else:
        print("No valid tool selected.")


def process_horn_chip(image, x_horn, y_horn, x_eye, y_eye):
    print("image -", image)
    chip_name = get_chip_filename(image.file.name, settings.CHIP_FILETYPE)
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
            # also update database
            ibex_chip = get_object_or_404(IbexChip, ibex_image_id=image.id)
            ibex_chip.delete()
            # chip_path.unlink()
            print(
                "IbexChip already existed on local storage, deleted successfully before continueing."
            )

        # create new chip from original image and try to preserve all metadata
        shutil.copy2(image_path, chip_path)

        # load image
        img = load_image(chip_path)

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
                # Delete the associated IbexChip object from the database
                ibex_chip = get_object_or_404(IbexChip, ibex_image_id=image.id)
                ibex_chip.delete()

                # # Delete the file from Backblaze B2 bucket
                # b2_utils.delete_files([chip_bucket_path])
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

    # setup landmarks
    eyehorn_src = [
        [x_eye, y_eye],
        [x_horn, y_horn],
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
    embed_new_chip(ibex_chip)
    return


# catch url where user is coming from to redirect there after task
def get_task_request_origin(request):
    """
    This function catches where the post request came from in order to redirect
    the user to that origin after the task is complete.
    Tasks, such as locating an image, or landmarking can be requested from
    different pages (unidentified-images, image-update) but they are handled
    by the same view function.
    """
    referer = request.META.get("HTTP_REFERER")
    if referer:
        parsed_url = urlparse(referer)
        # parsed_url.path will be something like "/unidentified/"
        try:
            match = resolve(parsed_url.path)
            task_request_url_name = match.url_name  # e.g., "unidentified-images"
            print("Matched URL pattern:", task_request_url_name)
        except Exception as e:
            task_request_url_name = None
            print("No matching URL pattern:", e)
    return task_request_url_name


def id_color_mapping(gallery_and_distances):
    """Map colors to the different ID's when showing the 5 best matching horn chips."""
    # Define your five color classes (adjust these as needed)
    # need to be safelisted for tailwind at 'note/tailwind.config.js'
    color_classes = [
        "bg-blue-400",
        "bg-purple-400",
        "bg-orange-400",
        "bg-slate-400",
        "bg-emerald-400",
    ]

    # Create a dictionary mapping animal IDs to color classes.
    # We'll iterate over gallery_and_distances (or a list of chips) and assign each new animal ID the next color.
    id_to_color = {}
    for chip, distance in gallery_and_distances:
        animal_id = chip.ibex_image.animal.id
        if animal_id not in id_to_color:
            # Use modulo in case there are more than 5, though you mentioned max 5.
            id_to_color[animal_id] = color_classes[
                len(id_to_color) % len(color_classes)
            ]
    return id_to_color


def get_gallery(query_embedding, gallery_chips):
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
    return top5_sorted_gallery


def overlapping_regions(single_region, regions_qs):
    """
    Given a single region and a queryset of regions,
    return a list of regions that overlap with the single region.

    Overlap is defined as when the distance between centers is less than the sum of the radii.
    """
    overlaps = []
    for region in regions_qs:
        # Ensure both regions have valid coordinates and radii.
        if (single_region.origin_latitude is None or single_region.origin_longitude is None or single_region.radius is None or
            region.origin_latitude is None or region.origin_longitude is None or region.radius is None):
            continue

        center_single = (single_region.origin_latitude, single_region.origin_longitude)
        center_other = (region.origin_latitude, region.origin_longitude)
        dist = distance(center_single, center_other).meters

        # Check if the regions overlap.
        if dist < (single_region.radius + region.radius):
            overlaps.append(region)

    return overlaps