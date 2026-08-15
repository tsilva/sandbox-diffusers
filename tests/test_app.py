from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image
from starlette.testclient import TestClient

from sandbox_diffusers.app import _decode_image, build_app


def test_local_ui_and_health_report_security_boundary() -> None:
    with TestClient(build_app()) as client:
        response = client.get("/")
        health = client.get("/healthz")

    assert response.status_code == 200
    assert "Text to image" in response.text
    assert "Image to image" in response.text
    assert "Compare" in response.text
    assert "share" not in response.text.lower()
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert health.json()["remote_code"] is False


def test_api_rejects_invalid_input_without_loading_a_model() -> None:
    with TestClient(build_app()) as client:
        invalid_steps = client.post("/api/txt2img", json={"steps": 31})
        missing_image = client.post("/api/img2img", json={})
        invalid_shape = client.post("/api/compare", json=[])

    assert invalid_steps.status_code == 400
    assert invalid_steps.json()["error"] == "steps must be between 1 and 30"
    assert missing_image.status_code == 400
    assert missing_image.json()["error"] == "Upload a source image first"
    assert invalid_shape.status_code == 400


def test_small_uploaded_image_is_decoded_safely() -> None:
    image = Image.new("RGB", (6, 4), "purple")
    encoded = BytesIO()
    image.save(encoded, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

    decoded = _decode_image(data_url)

    assert decoded.mode == "RGB"
    assert decoded.size == (6, 4)
