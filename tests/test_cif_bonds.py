"""
Bonds to sites in other cells, through the CIF text of the session
(spectrojotometer 6448ef4).

The server keeps each model as a CIF file and reads it back whenever its
text is edited. spectrojotometer wrote the cell shift of a bond with the
shift itself in each digit (1_100) and read it as the standard CIF code
(1_655 for the same shift): after an edit, such bonds were 4 or 5 cells
away (69.7 A instead of 2.94 A in a spinel). And pressing "Add bonds"
again added every bond a second time, doubling its coefficients.
"""

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from spectrojotometer.tools import unpack_offset
from visualbondweb import api

client = TestClient(api.app)
FIXTURES = Path(__file__).parent / "fixtures"


def upload(name, primitive_cell=False):
    path = FIXTURES / name
    with path.open("rb") as f:
        res = client.post(
            "/model/upload",
            files={"file": (path.name, f)},
            data={"primitive_cell": str(primitive_cell).lower()},
        )
    assert res.status_code == 200, res.text
    return res.json()


def add_bonds(sid, **params):
    res = client.post("/model/add_bonds", json={"session_id": sid, **params})
    assert res.status_code == 200, res.text
    return res.json()


def bonds(sid):
    """{coupling: sorted bond lengths} of the model of the session."""
    model = api.sessions[sid]["model"]
    pos = np.array(model.site_properties["coord_atomos"], dtype=float)
    lat = np.array(model.lattice_properties["bravais_vectors"], dtype=float)
    return {
        name: sorted(
            round(float(np.linalg.norm(pos[j] + unpack_offset(tag) @ lat - pos[i])), 4)
            for i, j, tag in coupling["bonds"]
        )
        for name, coupling in model.bonds.items()
    }


def bond_codes(cif_text):
    loop = cif_text.split("_geom_bond_site_symmetry_2", 1)[1]
    return sorted(line.split()[-1] for line in loop.strip().split("\n") if line.strip())


def test_editing_the_cif_keeps_the_bonds_across_cells():
    sid = upload("synthetic_bcc.struct")["session_id"]
    cif_text = add_bonds(sid, rmin=0.0, rmax=3.0)["cif_text"]
    before = bonds(sid)
    assert any(code != "." for code in bond_codes(cif_text))
    # The CIF text, as the frontend sends it back after an edit.
    res = client.post(f"/model/{sid}/validate", json={"cif_text": cif_text})
    assert res.status_code == 200, res.text
    assert bonds(sid) == before
    # And uploaded as an edited file.
    res = client.post(f"/model/{sid}/cif", files={"file": ("model.cif", cif_text.encode())})
    assert res.status_code == 200, res.text
    assert bonds(sid) == before


def test_the_cif_uses_standard_codes():
    sid = upload("synthetic_bcc.struct", primitive_cell=True)["session_id"]
    codes = bond_codes(add_bonds(sid, rmin=0.0, rmax=2.6)["cif_text"])
    # One site, the eight neighbours of a bcc lattice: four bonds, each to
    # a neighbouring cell (n_klm, each digit the shift + 5).
    assert len(codes) == 4
    for code in codes:
        n, klm = code.split("_")
        assert n == "1" and all(abs(int(d) - 5) <= 1 for d in klm), code


def test_a_cif_in_the_old_convention_is_read_with_a_warning():
    data = upload("old_convention_bonds.cif")
    assert any("old convention" in w for w in data["warnings"])
    assert bonds(data["session_id"]) == {"J1": [2.5, 2.5, 2.5]}
    # The CIF of the session is written in the standard convention.
    assert bond_codes(data["cif_text"]) == ["1_556", "1_565", "1_655"]
    res = client.post(
        f"/model/{data['session_id']}/validate", json={"cif_text": data["cif_text"]}
    )
    assert res.status_code == 200, res.text
    assert not any("old convention" in w for w in res.json().get("warnings", []))
    assert bonds(data["session_id"]) == {"J1": [2.5, 2.5, 2.5]}


def test_add_bonds_twice_does_not_duplicate_them():
    sid = upload("synthetic_bcc.struct")["session_id"]
    first = add_bonds(sid, rmin=0.0, rmax=3.0)
    counts = {name: len(lengths) for name, lengths in bonds(sid).items()}
    second = add_bonds(sid, rmin=0.0, rmax=3.0)
    assert {name: len(lengths) for name, lengths in bonds(sid).items()} == counts
    assert second["cif_text"] == first["cif_text"]
