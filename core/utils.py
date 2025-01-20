import os
import re
import cv2
import math
import base64
import requests
import datetime
import numpy as np

from django.conf import settings

from core.models import Animal, Embedding

from environ import Env

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
        import tensorflow as tf

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
