"""Tests for the /evaluate endpoint."""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectrojotometer.magnetic_model import MagneticModel
from visualbondweb import api

client = TestClient(api.app)

J_TRUE = np.array([-0.01, 0.004])
TOL = 1.0e-3


def ring_model():
    """Ring of 6 sites, nearest and next-nearest neighbour couplings."""
    n = 6
    pos = [np.array([float(i), 0.0, 0.0]) for i in range(n)]
    lists = [[(i, (i + d) % n, (0,)) for i in range(n)] for d in (1, 2)]
    return MagneticModel(
        pos, [np.array([float(n), 0.0, 0.0])],
        bond_lists=lists, bond_distances=[1.0, 2.0],
    )


def spin_text(model, confs, seed=0, shift=0.0):
    coeffs = model.coefficient_matrix(confs, False)
    rng = np.random.default_rng(seed)
    energs = coeffs.dot(np.r_[J_TRUE, -100.0]) + rng.uniform(-TOL / 2, TOL / 2, len(confs))
    energs = energs + shift * np.where(np.arange(len(confs)) % 2, 1, -1)
    return "\n".join(f"{e:.8f}\t{c}\t# {i}" for i, (e, c) in enumerate(zip(energs, confs)))


ALL_CONFS = [[0] + [int(b) for b in np.binary_repr(k, 5)] for k in range(32)]


@pytest.fixture
def session():
    sid = client.post("/session/new").json()["session_id"]
    api.sessions[sid]["model"] = ring_model()
    return sid


def evaluate(sid, text, **kw):
    body = {"session_id": sid, "configs_text": text, "energy_tolerance": TOL}
    body.update(kw)
    return client.post("/evaluate", json=body)


NAMES = list(ring_model().bonds)


def bars(data):
    return {p["name"]: p for p in data["parameters"]}


def test_default_method_is_exact(session):
    r = evaluate(session, spin_text(api.sessions[session]["model"], ALL_CONFS))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["method"] == "exact"
    assert data["acceptance_rate"] is None
    assert not data["incompatible"]
    p = bars(data)
    for name, j in zip(NAMES, J_TRUE):
        assert abs(p[name]["value"] - j) <= p[name]["error"]
        assert p[name]["undetermined"] is False


def test_methods_are_ordered(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS)
    err = {}
    for method in ("montecarlo", "exact", "ellipse"):
        data = evaluate(session, text, method=method, mc_steps=3000).json()
        assert data["method"] == method
        err[method] = np.array([bars(data)[n]["error"] for n in NAMES])
    assert np.all(err["montecarlo"] <= err["exact"] * (1 + 1e-9))
    assert np.all(err["exact"] <= err["ellipse"] * (1 + 1e-9))


def test_legacy_use_montecarlo_flag(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS)
    data = evaluate(session, text, use_montecarlo=True).json()
    assert data["method"] == "montecarlo"
    assert 0 < data["acceptance_rate"] < 1
    assert evaluate(session, text, use_montecarlo=False).json()["method"] == "ellipse"


def test_unknown_method_rejected(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS)
    assert evaluate(session, text, method="bogus").status_code == 422


def test_inequalities_lower_upper(session):
    data = evaluate(session, spin_text(api.sessions[session]["model"], ALL_CONFS)).json()
    assert len(data["inequalities"]) == 2
    for iq in data["inequalities"]:
        assert math.isclose(iq["lower"], iq["center"] - iq["half_width"])
        assert math.isclose(iq["upper"], iq["center"] + iq["half_width"])
        assert iq["lower"] < iq["upper"]
        value = float(np.dot(iq["coefficients"], J_TRUE))
        assert iq["lower"] <= value <= iq["upper"]


def test_undetermined_couplings_are_valid_json(session):
    # Repeating the same configurations cannot determine the couplings.
    confs = [ALL_CONFS[0]] * 2 + [ALL_CONFS[1]]
    r = evaluate(session, spin_text(api.sessions[session]["model"], confs))
    assert r.status_code == 200, r.text
    p = bars(r.json())
    assert any(p[n]["undetermined"] for n in NAMES)
    for n in NAMES:
        if p[n]["undetermined"]:
            assert p[n]["value"] is None and p[n]["error"] is None


def test_incompatible_model(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS, shift=1.0)
    data = evaluate(session, text).json()
    assert data["incompatible"]
    assert all(bars(data)[n]["incompatible"] for n in NAMES)


# /model/add_bonds: every bond within the range (spectrojotometer fdf406e).

def add_bonds(sid, **params):
    return client.post("/model/add_bonds", json={"session_id": sid, **params})


def upload_fixture(name, primitive_cell):
    path = Path(__file__).parent / "fixtures" / name
    with path.open("rb") as f:
        res = client.post(
            "/model/upload",
            files={"file": (path.name, f)},
            data={"primitive_cell": str(primitive_cell).lower()},
        )
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def test_add_bonds_finds_every_bond_within_the_range():
    """A primitive bcc cell of 2.86 A and a range of 7 A: the search was
    limited to the neighbouring cells, and 55 of its 68 bonds were missed."""
    sid = upload_fixture("synthetic_bcc.struct", primitive_cell=True)
    res = add_bonds(sid, rmin=0.0, rmax=7.0)
    assert res.status_code == 200, res.text
    model = api.sessions[sid]["model"]
    pos = np.array(model.site_properties["coord_atomos"], dtype=float)
    lat = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    expected = sum(
        1
        for n in itertools.product(range(-6, 7), repeat=3)
        if 0 < np.linalg.norm(np.array(n) @ lat) < 7.0
    ) // 2  # one site: (0, 0, n) and (0, 0, -n) are the same bond
    assert len(pos) == 1
    assert sum(len(b["bonds"]) for b in model.bonds.values()) == expected == 68


def test_add_bonds_ignores_the_supercell_size():
    sids = [upload_fixture("synthetic_bcc.struct", primitive_cell=True) for _ in range(2)]
    a = add_bonds(sids[0], rmin=0.0, rmax=5.0).json()["cif_text"]
    b = add_bonds(sids[1], rmin=0.0, rmax=5.0, supercell_size=[3, 3, 3]).json()["cif_text"]
    assert a == b


def test_a_range_too_long_for_the_cell_is_an_error():
    """It was a 500; the offset was stored wrapped before."""
    sid = upload_fixture("synthetic_bcc.struct", primitive_cell=True)
    res = add_bonds(sid, rmin=0.0, rmax=20.0)
    assert res.status_code == 422
    assert "cell offset" in res.json()["detail"]
