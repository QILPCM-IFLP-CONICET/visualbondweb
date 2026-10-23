"""
Importing configurations from another model, exporting the transformed
model, and Monte Carlo simulations.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.model_io import magnetic_model_from_file
from visualbondweb import api

client = TestClient(api.app)

LATTICE = np.diag([4.0, 5.0, 6.0])
SITES = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])  # fractional


def cell_model():
    model = MagneticModel(list(SITES @ LATTICE), list(LATTICE), magnetic_species=["Cr", "Cr"])
    model.generate_bonds(0.02, [[0, 4.5]])
    return model


def double_model():
    frac = np.vstack([SITES * [0.5, 1, 1], SITES * [0.5, 1, 1] + [0.5, 0, 0]])
    lattice = LATTICE * [[2], [1], [1]]
    model = MagneticModel(list(frac @ lattice), list(lattice), magnetic_species=["Cr"] * 4)
    model.generate_bonds(0.02, [[0, 4.5]])
    return model


def upload(path):
    with open(path, "rb") as f:
        res = client.post("/model/upload", files={"file": (path.name, f)})
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def import_model(sid, path, **form):
    with open(path, "rb") as f:
        return client.post(
            "/import/model",
            data={"session_id": sid, **{k: str(v) for k, v in form.items()}},
            files={"file": (path.name, f)},
        )


@pytest.fixture
def files(tmp_path):
    cell, double = tmp_path / "cell.cif", tmp_path / "double.cif"
    cell_model().save_cif(str(cell))
    double_model().save_cif(str(double))
    return cell, double


def test_import_from_a_cell_into_its_supercell(files):
    cell, double = files
    sid = upload(double)
    res = import_model(sid, cell)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["method"] == "position" and data["missing_sites"] == []
    assert data["energy_scale"] == 2.0
    sm = data["site_map"]
    res = client.post(
        "/import/map",
        json={"session_id": sid, "configs_text": "1.5 [0, 1] #a\n# note\n2.5 [1, 1] # b\n"},
    )
    assert res.status_code == 200, res.text
    out = res.json()
    assert out["imported"] == 2 and out["skipped"] == 0
    assert f"3.0\t{[[0, 1][j] for j in sm]}\t# a" in out["mapped_text"]
    assert "# note" in out["mapped_text"]


def test_import_from_a_supercell_skips_non_periodic(files):
    cell, double = files
    sid = upload(cell)
    data = import_model(sid, double).json()
    assert data["energy_scale"] == 0.5
    sm = data["site_map"]
    periodic = [0, 1, 0, 1] if sm == [0, 1] else None
    configs = "4.0 [0, 1, 0, 1] # periodic\n8.0 [0, 1, 1, 1] # not\n"
    out = client.post("/import/map", json={"session_id": sid, "configs_text": configs}).json()
    assert out["imported"] == 1 and out["skipped"] == 1
    assert "# skipped, not periodic with the current cell: 8.0" in out["mapped_text"]
    if periodic:
        assert "2.0\t[0, 1]\t# periodic" in out["mapped_text"]


def test_import_errors(files):
    cell, double = files
    sid = upload(cell)
    res = client.post("/import/map", json={"session_id": sid, "configs_text": "1 [0, 1]"})
    assert res.status_code == 400  # no other model yet
    import_model(sid, cell)
    res = client.post(
        "/import/map",
        json={"session_id": sid, "configs_text": "1 [0, 1]", "site_map": [0, -1]},
    )
    assert res.status_code == 422 and "no counterpart" in res.json()["detail"]


def test_transformed_model(files, tmp_path):
    cell, double = files
    sid = upload(double)
    import_model(sid, cell)
    res = client.get(f"/import/{sid}/transformed_model")
    assert res.status_code == 200, res.text
    assert "cell_transformed.cif" in res.headers["content-disposition"]
    path = tmp_path / "transformed.cif"
    path.write_text(res.text)
    transformed = magnetic_model_from_file(str(path))
    assert len(transformed.site_properties["coord_atomos"]) == 4
    assert list(transformed.bonds) == list(cell_model().bonds)
    sid_small = upload(cell)
    import_model(sid_small, double)
    assert client.get(f"/import/{sid_small}/transformed_model").status_code == 422


# ── Monte Carlo ──────────────────────────────────────────────────────────────


def session_with_couplings(files, value=-1.0):
    cell, _ = files
    sid = upload(cell)
    model = api.sessions[sid]["model"]
    for bond in model.bonds.values():
        bond["value"] = value
    return sid


def wait(job_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        data = client.get(f"/montecarlo/{job_id}").json()
        if data["status"] != "running":
            return data
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_couplings_endpoint(files):
    cell, _ = files
    sid = upload(cell)
    data = client.get(f"/montecarlo/{sid}/couplings").json()
    assert data["num_sites"] == 2
    assert all(c["value"] is None for c in data["couplings"])  # not estimated
    for bond in api.sessions[sid]["model"].bonds.values():
        bond["value"] = 0.5
    assert all(c["value"] == 0.5 for c in client.get(f"/montecarlo/{sid}/couplings").json()["couplings"])


def test_run_a_simulation(files):
    sid = session_with_couplings(files)
    res = client.post(
        "/montecarlo/run",
        json={"session_id": sid, "supercell": [2, 2, 2], "n_temps": 4,
              "n_equil": 100, "n_sweeps": 300, "seed": 1},
    )
    assert res.status_code == 200, res.text
    assert res.json()["num_sites"] == 16
    data = wait(res.json()["job_id"])
    assert data["status"] == "done", data
    results = data["results"]
    assert len(results["T"]) == 4 and len(results["E"]) == 4
    assert results["algorithm"] in ("wolff", "metropolis")
    assert data["progress"]["done"] == data["progress"]["total"] == 4


def test_couplings_override_and_missing_values(files):
    cell, _ = files
    sid = upload(cell)  # no values
    res = client.post("/montecarlo/run", json={"session_id": sid, "n_temps": 2})
    assert res.status_code == 422 and "no value" in res.json()["detail"]
    names = list(api.sessions[sid]["model"].bonds)
    res = client.post(
        "/montecarlo/run",
        json={"session_id": sid, "n_temps": 2, "n_sweeps": 50, "n_equil": 10,
              "couplings": {n: 1.0 for n in names}},
    )
    assert res.status_code == 200, res.text
    assert wait(res.json()["job_id"])["status"] == "done"


def test_limits(files, monkeypatch):
    sid = session_with_couplings(files)
    monkeypatch.setattr(api, "MC_MAX_SPIN_UPDATES", 1000)
    res = client.post("/montecarlo/run", json={"session_id": sid})
    assert res.status_code == 422 and "limit" in res.json()["detail"]
    assert client.post(
        "/montecarlo/run", json={"session_id": sid, "supercell": [9, 1, 1]}
    ).status_code == 422
    assert client.post(
        "/montecarlo/run", json={"session_id": sid, "T_min": 3, "T_max": 1}
    ).status_code == 422


def test_cancel(files):
    sid = session_with_couplings(files)
    res = client.post(
        "/montecarlo/run",
        json={"session_id": sid, "supercell": [4, 4, 4], "n_temps": 50,
              "n_equil": 2000, "n_sweeps": 20000, "algorithm": "metropolis"},
    )
    job_id = res.json()["job_id"]
    client.post(f"/montecarlo/{job_id}/cancel")
    assert wait(job_id)["status"] == "cancelled"
    assert client.get("/montecarlo/unknown").status_code == 404
