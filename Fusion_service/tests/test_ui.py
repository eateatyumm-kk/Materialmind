import json
import re
from pathlib import Path

import pytest
from pymatgen.core import Composition, Structure

from app.ensemble import REFERENCE_TEST_RMSE_LOG10
from app.schemas import HealthResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"
EXAMPLES = json.loads((STATIC_DIR / "examples.json").read_text(encoding="utf-8"))
HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")


# ---------------- static assets (no trained weights needed) ----------------
def test_every_element_id_used_by_the_script_exists_in_the_page():
    # Catches HTML/JS drift, e.g. renaming an id in one file only.
    js_ids = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    html_ids = set(re.findall(r'\bid="([\w-]+)"', HTML))
    assert js_ids, "expected the script to look up elements by id"
    assert js_ids <= html_ids, f"ids used in app.js but missing from index.html: {sorted(js_ids - html_ids)}"


def test_page_references_only_local_assets():
    # The UI must work offline and inside the Docker image: no CDN scripts, fonts or images.
    # (The w3.org SVG namespace is an identifier, not a network request.)
    for name, text in {"index.html": HTML, "app.js": JS, "style.css": CSS}.items():
        urls = [u for u in re.findall(r"https?://[^\s\"')]+", text) if "www.w3.org/2000/svg" not in u]
        assert not urls, f"{name} references external resources: {urls}"


def test_page_links_the_bundled_script_and_stylesheet():
    assert 'href="/static/style.css"' in HTML
    assert 'src="/static/app.js"' in HTML


def test_script_never_writes_server_data_with_innerhtml():
    # Response text is rendered with textContent only, so a hostile formula can't inject markup.
    assert "innerHTML" not in JS
    assert "insertAdjacentHTML" not in JS


def test_examples_are_well_formed():
    assert len(EXAMPLES) >= 3
    assert len({ex["id"] for ex in EXAMPLES}) == len(EXAMPLES)
    for ex in EXAMPLES:
        assert set(ex) == {"id", "formula", "label", "filename", "cif"}
        assert ex["filename"].lower().endswith(".cif")


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda ex: ex["id"])
def test_example_formula_matches_its_structure(example):
    structure = Structure.from_str(example["cif"], fmt="cif")
    assert structure.composition.reduced_formula == Composition(example["formula"]).reduced_formula


# ---------------- served by the app (needs the real checkpoints) ----------------
@pytest.mark.weights
def test_index_serves_the_web_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form id="predict-form"' in response.text


@pytest.mark.weights
@pytest.mark.parametrize("asset", ["style.css", "app.js", "examples.json"])
def test_static_assets_are_served(client, asset):
    assert client.get(f"/static/{asset}").status_code == 200


@pytest.mark.weights
def test_static_mount_does_not_expose_source_files(client):
    for path in ("/static/../main.py", "/static/%2e%2e/main.py", "/static/..%2fmain.py"):
        response = client.get(path)
        assert response.status_code != 200 or "FastAPI(" not in response.text, path


@pytest.mark.weights
def test_health_reports_ensemble_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    health = HealthResponse.model_validate(response.json())
    assert health.status == "ok"
    assert health.ensemble_size == 5
    assert health.seeds == [42, 123, 456, 789, 2024]
    assert health.reference_test_rmse_log10 == REFERENCE_TEST_RMSE_LOG10


@pytest.mark.weights
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda ex: ex["id"])
def test_each_ui_example_predicts_through_the_api(client, example):
    files = {"cif_file": (example["filename"], example["cif"].encode(), "text/plain")}
    response = client.post("/predict", data={"formula": example["formula"]}, files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["formula"] == example["formula"]
    assert body["bulk_modulus_gpa_mean"] > 0
    assert body["ensemble_size"] == 5


@pytest.mark.weights
def test_openapi_documents_the_service(client):
    spec = client.get("/openapi.json").json()
    assert "MaterialMind" in spec["info"]["title"]
    assert {tag["name"] for tag in spec["tags"]} >= {"prediction", "system"}
    assert "/" not in spec["paths"]  # the web page is not part of the API schema
    assert "/health" in spec["paths"]

    predict = spec["paths"]["/predict"]["post"]
    assert predict["summary"]
    assert {"200", "400", "422", "500"} <= set(predict["responses"])
