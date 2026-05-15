"""
Visualbond — FastAPI backend for spectrojotometer
"""
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

# ── NumPy 2.0 compatibility patch ────────────────────────────────────────────
# spectrojotometer uses np.Infinity which was removed in NumPy 2.0
if not hasattr(np, 'Infinity'):
    np.Infinity = np.inf
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from spectrojotometer.model_io import (
    confindex,
    magnetic_model_from_file,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualbond")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Visualbond API",
    description="Web API for the spectrojotometer library",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id -> { model, cif_path, configs }
sessions: dict = {}

TMPDIR = Path(tempfile.gettempdir()) / "visualbond"
TMPDIR.mkdir(exist_ok=True)

# ── Static files (frontend) ──────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "visualbondweb" / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class AddBondsRequest(BaseModel):
    session_id: str
    rmin: float = 0.0
    rmax: float = 4.9
    discretization: float = 0.02


class OptimizeConfigsRequest(BaseModel):
    session_id: str
    num_configs: int = 10
    bunch_size: int = 10
    iterations: int = 100


class OptimalIndepSetRequest(BaseModel):
    session_id: str


class EvaluateRequest(BaseModel):
    session_id: str
    configs_text: str          # raw .spin file content with energies
    energy_tolerance: float = 0.001
    use_montecarlo: bool = True
    mc_steps: int = 1000
    mc_size_factor: float = 1.0
    output_format: str = "plain"   # plain | latex | wolfram


class EquationsRequest(BaseModel):
    session_id: str
    configs_text: str
    output_format: str = "plain"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Load a model first.")
    return sessions[session_id]


def _parse_configs(configs_text: str, model) -> tuple[list, list, list]:
    """
    Parse a .spin text block into (energies, configs, labels).
    Lines starting with '#' or empty are ignored.
    Format: <energy|nan>  [0,1,0,...] # label
    """
    cell_size = model.lattice_properties["cell_size"]
    energies, confs, labels = [], [], []

    for lineno, line in enumerate(configs_text.splitlines(), 1):
        ls = line.strip()
        if not ls or ls.startswith("#"):
            continue
        fields = ls.split(maxsplit=1)
        if len(fields) < 2:
            raise HTTPException(
                status_code=422,
                detail=f"Parse error at line {lineno}: expected '<energy> <config>'",
            )
        try:
            energy = float(fields[0])
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid energy value at line {lineno}: '{fields[0]}'",
            )

        rest = fields[1].strip()
        comment = ""
        newconf = []
        for pos, ch in enumerate(rest):
            if ch == "#":
                comment = rest[pos + 1:].strip()
                break
            elif ch == "0":
                newconf.append(0)
            elif ch == "1":
                newconf.append(1)

        while len(newconf) < cell_size:
            newconf.append(0)

        if not comment:
            comment = str(confindex(newconf))

        energies.append(energy)
        confs.append(newconf)
        labels.append(comment)

    return energies, confs, labels


def _confs_to_text(energies, confs, labels) -> str:
    lines = ["# Spin configurations definition file",
             "# Energy\t[config]\t\t# label"]
    for i, conf in enumerate(confs):
        lines.append(f"{energies[i]}\t{_fmt_conf(conf)}\t\t# {labels[i]}")
    return "\n".join(lines)


def _fmt_conf(c) -> str:
    """Format a config list as plain ints, e.g. [0, 1, 0] (no np.int64)."""
    return '[' + ', '.join(str(int(x)) for x in c) + ']'


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the Visualbond frontend."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(
            content="<h2>Frontend not found.</h2>"
            "<p>Copy <code>visualbond_web.html</code> to "
            "<code>static/index.html</code> next to this file.</p>",
            status_code=404,
        )
    return HTMLResponse(content=index.read_text())


@app.get("/health")
def health():
    return {"app": "Visualbond API", "version": "0.2.0", "status": "ok"}


# ── Session management ────────────────────────────────────────────────────────

@app.post("/session/new")
def new_session():
    """Create a new empty session, return its ID."""
    sid = str(uuid.uuid4())
    sessions[sid] = {"model": None, "cif_path": None, "configurations": ([], [], [])}
    return {"session_id": sid}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Remove a session and its temp files."""
    sess = _get_session(session_id)
    if sess["cif_path"] and Path(sess["cif_path"]).exists():
        os.remove(sess["cif_path"])
    del sessions[session_id]
    return {"deleted": session_id}


# ── Model loading ─────────────────────────────────────────────────────────────

@app.post("/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a .cif or .struct file.
    Returns session_id + the CIF text of the parsed model.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".cif", ".struct"):
        raise HTTPException(status_code=400, detail="Only .cif and .struct files are supported.")

    # Save upload to a temp file
    tmp = TMPDIR / f"{uuid.uuid4()}{suffix}"
    content = await file.read()
    tmp.write_bytes(content)

    try:
        model = magnetic_model_from_file(filename=str(tmp))
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Cannot parse model: {exc}")

    # Save normalised CIF
    cif_path = str(TMPDIR / f"{uuid.uuid4()}.cif")
    model.save_cif(cif_path)
    with open(cif_path) as f:
        cif_text = f.read()

    sid = str(uuid.uuid4())
    sessions[sid] = {
        "model": model,
        "cif_path": cif_path,
        "configurations": ([], [], []),
    }
    tmp.unlink(missing_ok=True)

    bonds_info = [
        {"name": name, "distance": float(model.bonds[name].get("distance", 0))}
        for name in model.bonds
    ] if model.bonds else []

    num_atoms = len(model.site_properties.get("coord_atomos", []))

    return {
        "session_id": sid,
        "filename": file.filename,
        "cif_text": cif_text,
        "num_atoms": num_atoms,
        "cell_size": model.lattice_properties.get("cell_size", 0),
        "bonds": bonds_info,
    }


@app.get("/model/{session_id}/cif", response_class=PlainTextResponse)
def get_model_cif(session_id: str):
    """Return the current CIF text for the session."""
    sess = _get_session(session_id)
    if not sess["cif_path"] or not Path(sess["cif_path"]).exists():
        raise HTTPException(status_code=404, detail="No CIF file for this session.")
    return Path(sess["cif_path"]).read_text()


@app.post("/model/{session_id}/cif")
async def update_model_cif(session_id: str, file: UploadFile = File(...)):
    """Replace the model CIF from an edited upload."""
    sess = _get_session(session_id)
    content = (await file.read()).decode()
    tmp = TMPDIR / f"{uuid.uuid4()}.cif"
    tmp.write_text(content)
    try:
        model = magnetic_model_from_file(filename=str(tmp))
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Cannot parse updated CIF: {exc}")
    if sess["cif_path"]:
        Path(sess["cif_path"]).unlink(missing_ok=True)
    model.save_cif(str(tmp))
    sess["model"] = model
    sess["cif_path"] = str(tmp)
    return {"status": "updated", "cell_size": model.lattice_properties.get("cell_size")}


class ValidateCifRequest(BaseModel):
    cif_text: str


@app.post("/model/{session_id}/validate")
def validate_and_update_cif(session_id: str, req: ValidateCifRequest):
    """
    Validate the CIF text from the editor and update the session model.
    Called when the user leaves tab 1. Returns model info on success,
    or raises 422 with a human-readable detail on parse failure.
    """
    sess = _get_session(session_id)
    tmp = TMPDIR / f"{uuid.uuid4()}.cif"
    tmp.write_text(req.cif_text)
    try:
        model = magnetic_model_from_file(filename=str(tmp))
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    if sess["cif_path"]:
        Path(sess["cif_path"]).unlink(missing_ok=True)
    model.save_cif(str(tmp))
    sess["model"] = model
    sess["cif_path"] = str(tmp)
    num_atoms = len(model.site_properties.get("coord_atomos", []))
    bonds_info = [{"name": name} for name in model.bonds] if model.bonds else []
    return {
        "status": "ok",
        "num_atoms": num_atoms,
        "cell_size": model.lattice_properties.get("cell_size", 0),
        "bonds": bonds_info,
    }


# ── Bond generation ───────────────────────────────────────────────────────────

@app.post("/model/add_bonds")
def add_bonds(req: AddBondsRequest):
    """
    Call model.generate_bonds(), save the updated CIF, return new CIF text.
    Corresponds to the 'Add bonds' button (bond_generator step).
    """
    sess = _get_session(req.session_id)
    model = sess["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded in session.")

    try:
        model.generate_bonds(
            ranges=[[req.rmin, req.rmax]],
            discretization=req.discretization,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bond generation failed: {exc}")

    model.save_cif(sess["cif_path"])
    cif_text = Path(sess["cif_path"]).read_text()

    bonds_info = [
        {"name": name}
        for name in model.bonds
    ]

    return {
        "cif_text": cif_text,
        "bonds": bonds_info,
        "num_bonds": len(bonds_info),
    }


# ── Configuration optimization ────────────────────────────────────────────────

@app.post("/configs/optimize")
def optimize_configs(req: OptimizeConfigsRequest):
    """
    Generate optimal spin configurations (optconfs step).
    Returns updated .spin text to paste into the editor.
    """
    sess = _get_session(req.session_id)
    model = sess["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    if not model.bonds:
        raise HTTPException(status_code=400, detail="No bonds defined. Run Add bonds first.")

    energies, confs, labels = sess["configurations"]
    known, start = [], []
    for i, c in enumerate(confs):
        if not np.isnan(energies[i]):
            known.append(c)
        else:
            start.append(c)

    update_size = max(req.bunch_size, req.num_configs)
    try:
        newconfs, cn = model.find_optimal_configurations(
            num_new_confs=req.num_configs,
            start=start,
            known=known,
            its=req.iterations,
            update_size=update_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")

    new_labels = [str(confindex(c)) for c in newconfs]
    lines = [f"\n# New configurations. |ΔJ|/|ΔE| < {cn:.4g}:"]
    for i, nc in enumerate(newconfs):
        lines.append(f"nan\t{_fmt_conf(nc)}\t\t# {new_labels[i]}")

    return {
        "condition_number": float(cn),
        "new_configs_text": "\n".join(lines),
        "num_new_configs": len(newconfs),
    }


@app.post("/configs/optimal_independent_set")
def optimal_independent_set(req: OptimalIndepSetRequest):
    """Find an optimal independent subset of configurations."""
    sess = _get_session(req.session_id)
    model = sess["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    if not model.bonds:
        raise HTTPException(status_code=400, detail="No bonds defined.")

    _, confs, labels = sess["configurations"]
    if not confs:
        raise HTTPException(status_code=400, detail="No configurations loaded.")

    try:
        newconfs, cn = model.optimize_independent_set(confs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")

    new_labels = [str(confindex(c)) for c in newconfs]
    lines = [f"\n# Optimal independent subset. sqrt(l)/||A^-1|| = {cn:.4g}:"]
    for i, nc in enumerate(newconfs):
        lines.append(f"# {_fmt_conf(nc)}\t\t# {new_labels[i]}")

    return {
        "condition_number": float(cn),
        "subset_text": "\n".join(lines),
        "num_configs": len(newconfs),
    }


# ── Equations ─────────────────────────────────────────────────────────────────

@app.post("/configs/equations")
def get_equations(req: EquationsRequest):
    """
    Parse configurations text and return the formatted equations.
    """
    sess = _get_session(req.session_id)
    model = sess["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    if not model.bonds:
        raise HTTPException(status_code=400, detail="No bonds defined.")

    try:
        energies, confs, labels = _parse_configs(req.configs_text, model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    sess["configurations"] = (energies, confs, labels)

    if not confs:
        return {"equations": "# No configurations loaded.\n", "condition_number": None}

    cm = model.coefficient_matrix(confs, False)
    equations = model.formatted_equations(
        cm, ensname=None, comments=labels, eq_format=req.output_format
    )
    cost = model.cost(confs)
    equations += f"\n\n|ΔJ|/|ΔE| < {cost:.4g}"

    return {"equations": equations, "condition_number": float(cost)}


# ── Evaluate couplings ────────────────────────────────────────────────────────

@app.post("/evaluate")
def evaluate_couplings(req: EvaluateRequest):
    """
    Full evaluate_cc step.
    Parses configs+energies, computes coupling constants J_i,
    returns parameters, errors, chi values and inequalities.
    """
    sess = _get_session(req.session_id)
    model = sess["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")
    if not model.bonds:
        raise HTTPException(status_code=400, detail="No bonds defined.")

    try:
        energies, confs, labels = _parse_configs(req.configs_text, model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    sess["configurations"] = (energies, confs, labels)

    # Filter: only rows with a real energy (not nan)
    valid_confs, valid_energs, valid_labels = [], [], []
    for i, en in enumerate(energies):
        if not np.isnan(en):
            valid_confs.append(confs[i])
            valid_energs.append(en)
            valid_labels.append(labels[i])

    num_bonds = len(model.bonds)
    if len(valid_confs) < num_bonds + 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Need at least {num_bonds + 1} configurations with known energies "
                f"to determine {num_bonds} coupling constants. "
                f"Currently have {len(valid_confs)}."
            ),
        )

    try:
        js, jerr, chis, ar = model.compute_couplings(
            valid_confs,
            valid_energs,
            err_energs=req.energy_tolerance,
            montecarlo=req.use_montecarlo,
            mcsteps=req.mc_steps if req.use_montecarlo else None,
            mcsizefactor=req.mc_size_factor,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Coupling computation failed: {exc}")

    bond_names = list(model.bonds.keys())
    offset_energy = float(js[-1])
    js_vals = js[:-1]                     # remove E0
    jmax = float(np.max(np.abs(js_vals))) if len(js_vals) else 1.0
    fmt = req.output_format
    incompatible = bool(np.any(jerr < 0))

    # Build parameter list
    parameters = [{"name": "E0", "value": offset_energy, "error": None, "incompatible": False}]
    for i, name in enumerate(bond_names):
        parameters.append({
            "name": name,
            "value": float(js_vals[i]),
            "value_normalized": float(js_vals[i] / jmax) if jmax else 0,
            "error": float(jerr[i]),
            "error_normalized": float(jerr[i] / jmax) if jmax else 0,
            "incompatible": bool(jerr[i] < 0),
        })

    # Equations for the configurations that have energies
    cm = model.coefficient_matrix(valid_confs, False)
    equations = model.formatted_equations(
        cm, ensname=None, comments=valid_labels, eq_format=fmt
    )

    # Bound inequalities
    ineqs_raw = model.bound_inequalities(valid_confs, valid_energs, err_energs=req.energy_tolerance)
    inequalities = [
        {"coefficients": list(map(float, iq[0])), "lower": float(iq[1]), "upper": float(iq[2])}
        for iq in ineqs_raw
    ]

    return {
        "jmax": jmax,
        "parameters": parameters,
        "chis": [float(c) for c in chis],
        "chi_labels": valid_labels,
        "acceptance_rate": float(ar) if req.use_montecarlo else None,
        "incompatible": incompatible,
        "equations": equations,
        "inequalities": inequalities,
        "num_configs_used": len(valid_confs),
    }


# ── Config file I/O ───────────────────────────────────────────────────────────

@app.post("/configs/parse")
def parse_configs(req: EquationsRequest):
    """
    Validate and parse a .spin text block.
    Returns structured list of (energy, config, label).
    Useful for front-end validation before submission.
    """
    sess = _get_session(req.session_id)
    model = sess["model"]
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    energies, confs, labels = _parse_configs(req.configs_text, model)
    sess["configurations"] = (energies, confs, labels)

    known = sum(1 for e in energies if not np.isnan(e))
    return {
        "total": len(confs),
        "with_energy": known,
        "without_energy": len(confs) - known,
        "configs": [
            {"energy": e, "config": c, "label": l}
            for e, c, l in zip(energies, confs, labels)
        ],
    }


@app.get("/configs/{session_id}/download")
def download_configs(session_id: str):
    """Download the current configurations as a .spin file."""
    sess = _get_session(session_id)
    energies, confs, labels = sess["configurations"]
    if not confs:
        raise HTTPException(status_code=404, detail="No configurations in session.")

    text = _confs_to_text(energies, confs, labels)
    tmp = TMPDIR / f"{session_id}_configs.spin"
    tmp.write_text(text)
    return FileResponse(str(tmp), filename="configurations.spin", media_type="text/plain")


@app.get("/model/{session_id}/download")
def download_model(session_id: str):
    """Download the current CIF file."""
    sess = _get_session(session_id)
    if not sess["cif_path"] or not Path(sess["cif_path"]).exists():
        raise HTTPException(status_code=404, detail="No model in session.")
    return FileResponse(sess["cif_path"], filename="model.cif", media_type="text/plain")
