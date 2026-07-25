"""T39-T49: I/O-boundary scenarios (image decode, RunPod HTTP, B2 download)."""

from unittest import mock

import numpy as np
import pytest
import requests
from django.test import override_settings

from core.utils import embed_new_chip, endpoint_inference, load_image


# T39 -------------------------------------------------------------------
def test_load_image_png_happy(tmp_path, tiny_png_bytes):
    path = tmp_path / "tiny.png"
    path.write_bytes(tiny_png_bytes)

    result = load_image(str(path))

    assert result.shape == (3, 4, 3)  # (height, width, channels)
    assert result.dtype == np.uint8
    # source PNG is pure red in RGB -- after load_image's BGR2RGB conversion,
    # channel order must be RGB: red channel high, green/blue low.
    np.testing.assert_array_equal(result[0, 0], [255, 0, 0])


def test_load_image_jpeg_happy(tmp_path, tiny_jpeg_bytes):
    path = tmp_path / "tiny.jpeg"
    path.write_bytes(tiny_jpeg_bytes)

    result = load_image(str(path))

    assert result.shape == (3, 4, 3)
    assert result.dtype == np.uint8
    # JPEG is lossy -- assert red channel dominance instead of exact equality.
    pixel = result[0, 0]
    assert int(pixel[0]) > int(pixel[1])
    assert int(pixel[0]) > int(pixel[2])


# T40 -----------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Bug B5 (pinned, not fixed): load_image() lacks a None-guard after "
        "cv2.imread() -- an undecodable file (corrupted/AVIF-without-plugin) "
        "makes cv2.imread() return None, and cv2.cvtColor(None, ...) raises "
        "cv2.error instead of a clear domain error."
    ),
)
def test_load_image_corrupted_file_raises_cv2_error(tmp_path):
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not a real image")

    # No inner pytest.raises: it would itself catch cv2.error and let the
    # test function return normally (a plain PASS), which under
    # xfail(strict=True) registers as XPASS -- the opposite of pinning. Let
    # the exception propagate so the xfail marker is what turns the (still
    # buggy) raise into "expected failure."
    load_image(str(path))


# T41 -------------------------------------------------------------------
def test_endpoint_inference_happy_path(mock_runpod):
    response_mock = mock.Mock()
    response_mock.raise_for_status = mock.Mock()
    response_mock.json.return_value = {
        "output": {"output": {"output_tensor": [[0.1, 0.2, 0.3]]}}
    }
    mock_runpod.return_value = response_mock

    result = endpoint_inference(input_b64_img={"input": {"b64": "abc"}})

    assert result == [0.1, 0.2, 0.3]
    mock_runpod.assert_called_once()


# T42 -------------------------------------------------------------------
def test_endpoint_inference_request_exception_raises_value_error(mock_runpod):
    mock_runpod.side_effect = requests.exceptions.RequestException("boom")

    with pytest.raises(ValueError, match="Failed to reach RunPod"):
        endpoint_inference(input_b64_img={"input": {"b64": "abc"}})


# T43 -------------------------------------------------------------------
def test_endpoint_inference_response_error_field_raises_value_error(mock_runpod):
    response_mock = mock.Mock()
    response_mock.raise_for_status = mock.Mock()
    response_mock.json.return_value = {"error": "bad request"}
    mock_runpod.return_value = response_mock

    with pytest.raises(ValueError, match="RunPod error"):
        endpoint_inference(input_b64_img={"input": {"b64": "abc"}})


# T44 -------------------------------------------------------------------
def test_endpoint_inference_empty_output_raises_value_error(mock_runpod):
    response_mock = mock.Mock()
    response_mock.raise_for_status = mock.Mock()
    response_mock.json.return_value = {"output": {"output": {"output_tensor": [[]]}}}
    mock_runpod.return_value = response_mock

    with pytest.raises(ValueError, match="No output received"):
        endpoint_inference(input_b64_img={"input": {"b64": "abc"}})


# T45 -----------------------------------------------------------------------
def test_embed_new_chip_cloud_storage_and_cloud_endpoint_branch(
    mock_b2, tiny_png_bytes, ibex_chip_stub_factory
):
    mock_b2.return_value = tiny_png_bytes
    chip = ibex_chip_stub_factory(name="chip1.png")

    with (
        override_settings(POSTGRES_LOCALLY=True, AWS_LOCATION="media"),
        mock.patch("core.utils.endpoint_inference", return_value=[0.1, 0.2]) as inference_mock,
        mock.patch("core.utils.Embedding.objects.create") as create_mock,
        mock.patch("core.utils.get_tf") as get_tf_mock,
    ):
        embed_new_chip(chip)

    mock_b2.assert_called_once()
    inference_mock.assert_called_once()
    create_mock.assert_called_once_with(ibex_chip=chip, embedding=[0.1, 0.2])
    get_tf_mock.assert_not_called()


# T46 -------------------------------------------------------------------------
def test_embed_new_chip_local_storage_and_cloud_endpoint_branch(
    tmp_path, tiny_png_bytes, ibex_chip_stub_factory, mock_b2
):
    (tmp_path / "x.png").write_bytes(tiny_png_bytes)
    chip = ibex_chip_stub_factory(name="x.png")

    with (
        override_settings(MEDIA_ROOT=str(tmp_path)),
        mock.patch("core.utils.endpoint_inference", return_value=[0.2, 0.3]) as inference_mock,
        mock.patch("core.utils.Embedding.objects.create") as create_mock,
        mock.patch("core.utils.get_tf") as get_tf_mock,
    ):
        embed_new_chip(chip)

    inference_mock.assert_called_once()
    create_mock.assert_called_once_with(ibex_chip=chip, embedding=[0.2, 0.3])
    get_tf_mock.assert_not_called()
    mock_b2.assert_not_called()


# T47 -------------------------------------------------------------------
def test_endpoint_inference_url_override_unset_uses_real_url(monkeypatch, mock_runpod):
    monkeypatch.delenv("INFERENCE_ENDPOINT_URL_OVERRIDE", raising=False)
    response_mock = mock.Mock()
    response_mock.raise_for_status = mock.Mock()
    response_mock.json.return_value = {
        "output": {"output": {"output_tensor": [[0.1, 0.2, 0.3]]}}
    }
    mock_runpod.return_value = response_mock

    endpoint_inference(input_b64_img={"input": {"b64": "abc"}})

    assert (
        mock_runpod.call_args.args[0]
        == "https://api.runpod.ai/v2/test-runpod-endpoint-id/runsync"
    )


# T48 -------------------------------------------------------------------
def test_endpoint_inference_url_override_set_used_verbatim(monkeypatch, mock_runpod):
    monkeypatch.setenv("INFERENCE_ENDPOINT_URL_OVERRIDE", "http://localhost:8001/runsync")
    response_mock = mock.Mock()
    response_mock.raise_for_status = mock.Mock()
    response_mock.json.return_value = {
        "output": {"output": {"output_tensor": [[0.1, 0.2, 0.3]]}}
    }
    mock_runpod.return_value = response_mock

    endpoint_inference(input_b64_img={"input": {"b64": "abc"}})

    assert mock_runpod.call_args.args[0] == "http://localhost:8001/runsync"
    assert "api.runpod.ai" not in mock_runpod.call_args.args[0]


# T49 -------------------------------------------------------------------
def test_endpoint_inference_url_override_empty_string_treated_as_unset(
    monkeypatch, mock_runpod
):
    monkeypatch.setenv("INFERENCE_ENDPOINT_URL_OVERRIDE", "")
    response_mock = mock.Mock()
    response_mock.raise_for_status = mock.Mock()
    response_mock.json.return_value = {
        "output": {"output": {"output_tensor": [[0.1, 0.2, 0.3]]}}
    }
    mock_runpod.return_value = response_mock

    endpoint_inference(input_b64_img={"input": {"b64": "abc"}})

    assert (
        mock_runpod.call_args.args[0]
        == "https://api.runpod.ai/v2/test-runpod-endpoint-id/runsync"
    )
