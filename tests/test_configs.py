"""
Reading the configurations of the editor (/configs/parse, and the
endpoints that use the same reader): spectrojotometer's
parse_spin_configurations.
"""

import pytest
from fastapi.testclient import TestClient

from visualbondweb import api

from .test_evaluate import ring_model

client = TestClient(api.app)


@pytest.fixture
def session():
    sid = client.post("/session/new").json()["session_id"]
    api.sessions[sid]["model"] = ring_model()
    return sid


def num_sites(sid):
    return api.sessions[sid]["model"].lattice_properties["cell_size"]


def parse(sid, text):
    return client.post("/configs/parse", json={"session_id": sid, "configs_text": text})


def test_configurations_labels_and_unknown_energies(session):
    """An unknown energy (nan) is null: NaN is not valid JSON (it was a 500)."""
    n = num_sites(session)
    res = parse(session, f"# header\n1.5 [1]  #  first  \nnan {'0' * (n - 1)}1\n")
    assert res.status_code == 200, res.text
    data = res.json()
    assert (data["total"], data["with_energy"], data["without_energy"]) == (2, 1, 1)
    first, second = data["configs"]
    assert first == {"energy": 1.5, "config": [1] + [0] * (n - 1), "label": "first"}
    assert second["energy"] is None
    assert second["label"] == str(2 ** (n - 1))  # without a label: the index


@pytest.mark.parametrize("endpoint", ["/configs/parse", "/configs/equations"])
def test_more_values_than_sites_is_an_error(session, endpoint):
    """They were accepted, and the extra values ignored."""
    n = num_sites(session)
    text = f"1.0 {'0' * n}\n\n2.0 {'0' * (n + 2)}\n"
    res = client.post(endpoint, json={"session_id": session, "configs_text": text})
    assert res.status_code == 422
    assert res.json()["detail"] == f"Parse error at line 3: {n + 2} values for {n} sites"


@pytest.mark.parametrize(
    "text, detail",
    [
        ("# header\nabc [0, 1]\n", "Parse error at line 2: invalid energy 'abc'"),
        ("1.0 [0, 1]\n2.0\n", "Parse error at line 2: expected '<energy> <config>'"),
        ("1.0 # only a label\n", "Parse error at line 1: expected '<energy> <config>'"),
    ],
)
def test_invalid_lines(session, text, detail):
    res = parse(session, text)
    assert res.status_code == 422
    assert res.json()["detail"] == detail


def test_the_session_keeps_the_last_valid_configurations(session):
    assert parse(session, "1.0 [1]\n").status_code == 200
    before = api.sessions[session]["configurations"]
    assert parse(session, "1.0 [1]\nxyz [0]\n").status_code == 422
    assert api.sessions[session]["configurations"] == before
