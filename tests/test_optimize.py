"""
The endpoints that report the error bound of a set of configurations:
/configs/optimize, /configs/optimal_independent_set and /configs/equations.
"""

import math

import pytest
from fastapi.testclient import TestClient

from visualbondweb import api

from .test_evaluate import ALL_CONFS, ring_model, spin_text

client = TestClient(api.app)


@pytest.fixture
def session():
    sid = client.post("/session/new").json()["session_id"]
    api.sessions[sid]["model"] = ring_model()
    return sid


def equations(sid, text):
    return client.post(
        "/configs/equations",
        json={"session_id": sid, "configs_text": text, "output_format": "plain"},
    )


def test_equations_report_the_exact_error_bound(session):
    model = api.sessions[session]["model"]
    confs = ALL_CONFS[:8]
    res = equations(session, spin_text(model, confs))
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["error_bound"] == pytest.approx(model.cost(confs))
    assert float(data["condition_number"]) == pytest.approx(data["error_bound"])
    assert f"|ΔJ|/|ΔE| < {data['error_bound']:.4g}" in data["equations"]


def test_undetermined_couplings_give_null(session):
    model = api.sessions[session]["model"]
    res = equations(session, spin_text(model, ALL_CONFS[:1]))
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["error_bound"] is None
    assert data["condition_number"] == "inf"
    assert "some couplings are not determined" in data["equations"]


def optimize(sid, **kwargs):
    body = {"session_id": sid, "iterations": 5, "bunch_size": 3}
    body.update(kwargs)
    return client.post("/configs/optimize", json=body)


def test_optimize_with_a_given_number(session):
    res = optimize(session, num_configs=4)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["num_new_configs"] == 4
    assert math.isfinite(data["error_bound"])
    assert "|ΔJ|/|ΔE| <" in data["new_configs_text"]


@pytest.mark.parametrize("num_configs", [None, 0])
def test_optimize_chooses_the_number(session, num_configs):
    body = {} if num_configs is None else {"num_configs": num_configs}
    res = optimize(session, **body)
    assert res.status_code == 200, res.text
    data = res.json()
    # Two couplings and the base energy: at least three configurations.
    assert data["num_new_configs"] >= 3
    assert math.isfinite(data["error_bound"])


def test_optimize_takes_the_known_configurations_into_account(session):
    model = api.sessions[session]["model"]
    known = ALL_CONFS[:8]
    equations(session, spin_text(model, known))  # stores them in the session
    data = optimize(session).json()
    new = [line for line in data["new_configs_text"].splitlines() if line.startswith("nan")]
    assert len(new) == data["num_new_configs"]
    # The bound covers the known configurations together with the new ones.
    assert data["error_bound"] <= model.cost(known) * (1 + 1e-9)


def test_min_improvement_is_validated(session):
    assert optimize(session, min_improvement=1.5).status_code == 422
    assert optimize(session, min_improvement=-0.1).status_code == 422


def test_optimal_independent_set(session):
    model = api.sessions[session]["model"]
    equations(session, spin_text(model, ALL_CONFS))
    res = client.post("/configs/optimal_independent_set", json={"session_id": session})
    assert res.status_code == 200, res.text
    data = res.json()
    assert "sqrt(l)" not in data["subset_text"]
    assert f"|ΔJ|/|ΔE| < {data['error_bound']:.4g}" in data["subset_text"]
    sized = client.post(
        "/configs/optimal_independent_set", json={"session_id": session, "num_configs": 5}
    ).json()
    assert sized["num_configs"] == 5
