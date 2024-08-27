import os
import re
import cv2
import math
import datetime
import numpy as np

from core.models import Animal


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
