"""
Importing configurations from another model, exporting the transformed
model, and Monte Carlo simulations (the server builds the lattice; the
simulation runs in the browser, static/mc_engine.js, checked here with
Node against the library).
"""

import itertools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectrojotometer.magnetic_model import MagneticModel
from spectrojotometer.model_io import magnetic_model_from_file
from spectrojotometer.monte_carlo import detect_frustration, run_ising_mc
from spectrojotometer.monte_carlo.adapter import magnetic_model_to_ising
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


def test_transformed_model_aligned(tmp_path):
    """align=true: the other model is moved onto the session's model,
    each site at the image closest to its site, so each bond of the
    session's model is the same bond in it. Here the session has a
    centred square lattice (4 x 4 x 6 A, two sites, the second one stored
    outside the cell, at (-2, 2, 0), as after a symmetry expansion) and
    the other model its primitive cell (2.83 x 2.83 x 6 A, one site,
    another origin): read from a CIF, with a along x, it is rotated 45
    degrees with respect to the session, as the tetragonal cell of a
    spinel with respect to the cubic one, and StructureMatcher aligns
    them. Without align, the sites are wrapped into the cell.

    The CIF keeps the cell parameters, not their orientation: read back,
    a lies along x whatever the frame of the export. What align changes
    in the file is the origin and the image of each site."""
    from spectrojotometer.tools import unpack_offset

    square = MagneticModel(
        [np.zeros(3), np.array([-2.0, 2.0, 0.0])],
        [np.array([4.0, 0, 0]), np.array([0, 4.0, 0]), np.array([0, 0, 6.0])],
        magnetic_species=["Cr", "Cr"],
    )
    square.generate_bonds(0.02, [[0, 3.0]])
    side = 2 * np.sqrt(2)
    primitive = MagneticModel(
        [np.array([0.13 * side, 0.21 * side, 0.07 * 6])],
        [np.array([side, 0, 0]), np.array([0, side, 0]), np.array([0, 0, 6.0])],
        magnetic_species=["Cr"],
    )
    primitive.generate_bonds(0.02, [[0, 3.0]])
    session_file, other = tmp_path / "square.cif", tmp_path / "primitive.cif"
    square.save_cif(str(session_file))
    primitive.save_cif(str(other))
    sid = upload(session_file)
    imported = import_model(sid, other).json()
    assert imported["method"] == "structure"
    session = api.sessions[sid]["model"]
    targets = np.array(session.site_properties["coord_atomos"], dtype=float)
    np.testing.assert_allclose(targets[1], [-2, 2, 0], atol=1e-6)  # kept outside

    def exported(**params):
        res = client.get(f"/import/{sid}/transformed_model", params=params)
        assert res.status_code == 200, res.text
        path = tmp_path / "out.cif"
        path.write_text(res.text)
        model = magnetic_model_from_file(str(path))
        positions = np.array(model.site_properties["coord_atomos"], dtype=float)
        return res, model, np.linalg.norm(positions - targets, axis=1)

    res, _, distances = exported()
    assert "primitive_transformed.cif" in res.headers["content-disposition"]
    assert distances.max() > 0.5
    res, model, distances = exported(align="true")
    assert "primitive_aligned.cif" in res.headers["content-disposition"]
    np.testing.assert_allclose(distances, 0, atol=1e-3)
    positions = np.array(model.site_properties["coord_atomos"], dtype=float)
    lattice = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    session_lattice = np.array(session.lattice_properties["bravais_vectors"], dtype=float)
    np.testing.assert_allclose(lattice, session_lattice, atol=1e-3)
    assert session.bonds
    for coupling in session.bonds.values():
        for i, j, tag in coupling["bonds"]:
            ours = np.linalg.norm(targets[j] + unpack_offset(tag) @ lattice - targets[i])
            theirs = np.linalg.norm(positions[j] + unpack_offset(tag) @ lattice - positions[i])
            assert theirs == pytest.approx(ours, abs=1e-3)
    # With the site map given explicitly (the automatic one, written out).
    site_map = ",".join(str(s) for s in imported["site_map"])
    _, _, distances = exported(align="true", site_map=site_map)
    np.testing.assert_allclose(distances, 0, atol=1e-3)


# ── Monte Carlo ──────────────────────────────────────────────────────────────


def session_with_couplings(files, value=-1.0):
    cell, _ = files
    sid = upload(cell)
    model = api.sessions[sid]["model"]
    for bond in model.bonds.values():
        bond["value"] = value
    return sid


def test_couplings_endpoint(files):
    cell, _ = files
    sid = upload(cell)
    data = client.get(f"/montecarlo/{sid}/couplings").json()
    assert data["num_sites"] == 2
    assert all(c["value"] is None for c in data["couplings"])  # not estimated
    for bond in api.sessions[sid]["model"].bonds.values():
        bond["value"] = 0.5
    assert all(c["value"] == 0.5 for c in client.get(f"/montecarlo/{sid}/couplings").json()["couplings"])


def lattice(sid, **kw):
    return client.post("/montecarlo/lattice", json={"session_id": sid, **kw})


@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_the_lattice_is_the_one_of_the_library(files, value):
    sid = session_with_couplings(files, value)
    res = lattice(sid, supercell=[2, 2, 2])
    assert res.status_code == 200, res.text
    data = res.json()
    spins, bonds, j_vals, e0 = magnetic_model_to_ising(api.sessions[sid]["model"], (2, 2, 2))
    assert data["num_sites"] == len(spins) == 16
    assert data["bonds"] == np.asarray(bonds).tolist()
    assert data["J"] == pytest.approx(list(j_vals))
    assert data["E0"] == pytest.approx(e0)
    frustrated, gauge, cycle = detect_frustration(np.asarray(bonds), np.asarray(j_vals), 16)
    assert data["frustrated"] == frustrated
    if frustrated:  # the antiferromagnet of this model
        assert value < 0 and data["gauge"] is None and len(data["frustration_cycle"]) >= 3
    else:
        assert data["gauge"] == list(gauge) and data["frustration_cycle"] is None


def test_couplings_override_and_missing_values(files):
    cell, _ = files
    sid = upload(cell)  # no values
    res = lattice(sid)
    assert res.status_code == 422 and "no value" in res.json()["detail"]
    names = list(api.sessions[sid]["model"].bonds)
    res = lattice(sid, couplings={n: 0.25 for n in names})
    assert res.status_code == 200, res.text
    assert set(res.json()["J"]) == {0.25}


def test_limits(files, monkeypatch):
    sid = session_with_couplings(files)
    assert lattice(sid, supercell=[9, 1, 1]).status_code == 422
    assert lattice(sid, supercell=[0, 1, 1]).status_code == 422
    # The number of bonds is checked before building the lattice.
    per_cell = sum(len(b["bonds"]) for b in api.sessions[sid]["model"].bonds.values())
    monkeypatch.setattr(api, "MC_MAX_BONDS", 8 * per_cell - 1)

    def not_called(*args, **kwargs):
        raise AssertionError("the lattice was built")

    monkeypatch.setattr(api, "magnetic_model_to_ising", not_called)
    res = lattice(sid, supercell=[2, 2, 2])
    assert res.status_code == 422 and "limit of this server" in res.json()["detail"]


def test_the_lattice_is_compressed(files):
    sid = session_with_couplings(files)
    res = lattice(sid, supercell=[6, 6, 6])
    assert res.status_code == 200
    assert res.headers["content-encoding"] == "gzip"


# The JavaScript engine, run by Node on the lattice of the server, against
# the library on the same lattice.

ENGINE = Path(api.__file__).parent / "static" / "mc_engine.js"
NODE = shutil.which("node")


def run_engine(lattice_data, params, tmp_path):
    (tmp_path / "lattice.json").write_text(json.dumps(lattice_data))
    script = (
        f"const mc = require({json.dumps(str(ENGINE))});"
        f"const lat = require({json.dumps(str(tmp_path / 'lattice.json'))});"
        f"console.log(JSON.stringify(mc.runIsingMC(lat, {json.dumps(params)})));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True, timeout=120)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
@pytest.mark.parametrize(
    "value, algorithm, T_min",
    [
        (1.0, "auto", 0.5),  # unfrustrated: Wolff
        # Frustrated: Metropolis, above the temperatures where single flips
        # get trapped in metastable states (as the library does, with
        # some seeds, at T < 1), and parallel tempering at any temperature.
        (-1.0, "auto", 1.5),
        (-1.0, "pt", 0.5),
    ],
)
def test_the_browser_engine_agrees_with_the_library(files, value, algorithm, T_min, tmp_path):
    sid = session_with_couplings(files, value)
    data = lattice(sid, supercell=[2, 2, 2]).json()
    params = {"T_min": T_min, "T_max": 6.0, "n_temps": 5, "n_equil": 500, "n_sweeps": 100000,
              "algorithm": algorithm, "seed": 3, "swap_every": 2}
    out = run_engine(data, params, tmp_path)
    spins, bonds, j_vals, e0 = magnetic_model_to_ising(api.sessions[sid]["model"], (2, 2, 2))
    ref = run_ising_mc(spins, bonds, j_vals, e0, T_min, params["T_max"], params["n_temps"],
                       n_equil=params["n_equil"], n_sweeps=params["n_sweeps"],
                       algorithm=algorithm, seed=3, swap_every=2)
    js = out["results"]
    expected = algorithm if algorithm != "auto" else "metropolis" if data["frustrated"] else "wolff"
    assert js["algorithm"] == ref["algorithm"] == expected
    assert bool(out["warnings"]) == (data["frustrated"] and algorithm == "auto")
    assert js["T"] == pytest.approx(ref["T"])
    # Against the exact averages of the 16 sites: unbiased...
    exact = exact_averages(data, js["T"])
    scale = sum(abs(j) for j in j_vals) / len(spins)
    assert np.allclose(js["E"], exact["E"], atol=0.02 * scale)
    assert np.allclose(js["C"], exact["C"], rtol=0.1, atol=0.01)
    # ... and against the library: the same lattice and conventions.
    assert np.allclose(js["E"], ref["E"], atol=0.03 * scale)
    assert np.allclose(js["M"], ref["M"], atol=0.03)


def exact_averages(data, temperatures):
    """Energy and specific heat per site of a small lattice, by enumeration."""
    n = data["num_sites"]
    bonds = np.asarray(data["bonds"])
    states = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)
    energy = data["E0"] - (states[:, bonds[:, 0]] * states[:, bonds[:, 1]]) @ np.asarray(data["J"])
    out = {"E": [], "C": []}
    for T in temperatures:
        w = np.exp(-(energy - energy.min()) / T)
        w /= w.sum()
        e1, e2 = w @ energy, w @ energy**2
        out["E"].append(e1 / n)
        out["C"].append((e2 - e1 * e1) / (T * T * n))
    return out


def asymmetric_chain(xs, period):
    """Gaps 0.7, 0.9, 1.4: exchanging two sites is not a symmetry."""
    model = MagneticModel(
        [np.array([x, 0.0, 0.0]) for x in xs],
        [np.array([period, 0.0, 0.0]), np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 5.0])],
        magnetic_species=["Cr"] * len(xs),
    )
    model.generate_bonds(0.02, [[0, 1.5]])
    return model


def test_transformed_model_with_an_edited_site_map(tmp_path):
    """The exported model follows the edited map, as /import/map (it used
    the automatic one)."""
    asymmetric_chain([0, 0.7, 1.6, 3, 3.7, 4.6], 6.0).save_cif(str(tmp_path / "double.cif"))
    asymmetric_chain([0, 0.7, 1.6], 3.0).save_cif(str(tmp_path / "cell.cif"))
    sid = upload(tmp_path / "double.cif")
    assert import_model(sid, tmp_path / "cell.cif").json()["site_map"] == [0, 1, 2, 0, 1, 2]
    edited = [1, 0, 2, 1, 0, 2]
    res = client.get(
        f"/import/{sid}/transformed_model", params={"site_map": ",".join(map(str, edited))}
    )
    assert res.status_code == 200, res.text
    (tmp_path / "out.cif").write_text(res.text)
    exported = magnetic_model_from_file(str(tmp_path / "out.cif"))
    source = magnetic_model_from_file(str(tmp_path / "cell.cif"))
    for conf in ([0, 1, 1], [1, 0, 1], [0, 0, 1]):
        mapped = client.post(
            "/import/map",
            json={"session_id": sid, "configs_text": f"1.0 {conf}", "site_map": edited},
        ).json()["mapped_text"]
        mapped_conf = [int(c) for c in mapped.split("[")[1].split("]")[0].split(",")]
        assert mapped_conf == [conf[j] for j in edited]
        np.testing.assert_allclose(
            exported.coefficient_matrix([mapped_conf])[0][:-1],
            2 * source.coefficient_matrix([conf])[0][:-1],
        )


def test_transformed_model_with_an_invalid_site_map(files):
    cell, double = files
    sid = upload(double)
    import_model(sid, cell)
    res = client.get(f"/import/{sid}/transformed_model", params={"site_map": "0,0,0,1"})
    assert res.status_code == 422
    assert "the cell of the target holds each one 2 times" in res.json()["detail"]
    res = client.get(f"/import/{sid}/transformed_model", params={"site_map": "0,a,1"})
    assert res.status_code == 422 and "list of site indices" in res.json()["detail"]
    # An empty map: the automatic one.
    assert client.get(f"/import/{sid}/transformed_model", params={"site_map": ""}).status_code == 200
