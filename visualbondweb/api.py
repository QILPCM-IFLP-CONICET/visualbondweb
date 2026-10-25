"""
Visualbond — FastAPI backend for spectrojotometer
"""
import logging
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np

# ── NumPy 2.0 compatibility patch ────────────────────────────────────────────
# spectrojotometer uses np.Infinity which was removed in NumPy 2.0
if not hasattr(np, 'Infinity'):
    np.Infinity = np.inf
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spectrojotometer.magnetic_model import (
    compatible_configurations,
    equivalent_model,
    map_configurations,
    site_maps,
)
from spectrojotometer.monte_carlo import detect_frustration
from spectrojotometer.monte_carlo.adapter import magnetic_model_to_ising
from spectrojotometer.model_io import (
    SpinConfigurationError,
    confindex,
    magnetic_model_from_file,
    parse_spin_configurations,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualbond")


def _error_bound_fields(cost):
    """
    The error bound of a set of configurations (MagneticModel.cost: the
    largest error of a coupling per unit error in the energies) for a
    response: a number, or None if some coupling is not determined (inf is
    not valid JSON). condition_number is kept, as text, for older clients.
    """
    cost = float(cost)
    return {
        "error_bound": cost if np.isfinite(cost) else None,
        "condition_number": str(cost),
    }


def _error_bound_label(cost):
    cost = float(cost)
    if not np.isfinite(cost):
        return "|ΔJ|/|ΔE| = inf (some couplings are not determined)"
    return f"|ΔJ|/|ΔE| < {cost:.4g}"


class _WarningCollector(logging.Handler):
    """Collect the warnings emitted from one thread."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.thread = threading.get_ident()
        self.messages = []

    def emit(self, record):
        if record.thread == self.thread:
            self.messages.append(record.getMessage())


@contextmanager
def library_warnings():
    """
    Collect the warnings that spectrojotometer emits inside the block, to
    return them to the client (they also go to the server log, as usual).

    Requests are served concurrently and share the library's logger, so
    only the messages emitted from the current thread are collected. The
    block must not await: in async endpoints, other requests could run in
    the same thread meanwhile.
    """
    collector = _WarningCollector()
    library_logger = logging.getLogger("spectrojotometer")
    library_logger.addHandler(collector)
    try:
        yield collector.messages
    finally:
        library_logger.removeHandler(collector)

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
# The lattices for the Monte Carlo simulation can take a few megabytes.
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── In-memory session store ───────────────────────────────────────────────────
# Maps session_id -> { model, cif_path, configs }
sessions: dict = {}

TMPDIR = Path(tempfile.gettempdir()) / "visualbond"
TMPDIR.mkdir(exist_ok=True)

# ── Static files (frontend) ──────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class AddBondsRequest(BaseModel):
    session_id: str
    rmin: float = 0.0
    rmax: float = 4.9
    discretization: float = 0.02
    # Ignored: generate_bonds searches every cell within the range (it
    # only limited that search; it never built a supercell).
    supercell_size: Optional[list[int]] = None


class OptimizeConfigsRequest(BaseModel):
    session_id: str
    # Number of new configurations. None (or 0): chosen automatically, the
    # smallest set that determines every coupling, enlarged while each
    # configuration reduces the error bound by at least min_improvement.
    num_configs: Optional[int] = None
    min_improvement: float = Field(0.05, ge=0.0, le=1.0)
    bunch_size: int = 10
    iterations: int = 100


class OptimalIndepSetRequest(BaseModel):
    session_id: str
    # As in OptimizeConfigsRequest.
    num_configs: Optional[int] = None
    min_improvement: float = Field(0.05, ge=0.0, le=1.0)


class EvaluateRequest(BaseModel):
    session_id: str
    configs_text: str          # raw .spin file content with energies
    energy_tolerance: float = 0.001
    # Error-bound method: "exact" (default), "montecarlo" or "ellipse".
    method: Optional[Literal["exact", "montecarlo", "ellipse"]] = None
    # Deprecated: kept for older clients. Used only if method is None.
    use_montecarlo: Optional[bool] = None
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
    Parse configurations text into (energies, configs, labels), with the
    reader of spectrojotometer (parse_spin_configurations): one
    configuration per line, `<energy|nan> [0, 1, 0, ...] # label`, padded
    to the number of sites of the model. Configurations without a label
    are labelled by their index (confindex). An invalid line is an error
    (422) with its number.
    """
    try:
        energies, confs, labels = parse_spin_configurations(
            configs_text, model.lattice_properties["cell_size"]
        )
    except SpinConfigurationError as exc:
        raise HTTPException(
            status_code=422, detail=f"Parse error at line {exc.line}: {exc.reason}"
        )
    labels = [label or str(confindex(conf)) for label, conf in zip(labels, confs)]
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
    sessions[sid] = {
        "model": None,
        "cif_path": None,
        "configurations": ([], [], []),
        "primitive_cell": False,
    }
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
async def upload_model(
    file: UploadFile = File(...),
    primitive_cell: bool = Form(False),
):
    """
    Upload a .cif or .struct file.
    Returns session_id + the CIF text of the parsed model.

    `primitive_cell`: if True (only meaningful for CIFs whose symmetry
    loop lists pure-translation centering operators, e.g. F-centered
    lattices), atoms are kept exactly as declared in the CIF's
    asymmetric unit instead of being expanded to fill the conventional
    cell. This setting is remembered for the session: every later call
    that reloads this model (`/model/{id}/cif`, `/model/{id}/validate`)
    reuses it automatically, so the compact representation is never
    silently lost on a round-trip.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".cif", ".struct"):
        raise HTTPException(status_code=400, detail="Only .cif and .struct files are supported.")

    # Save upload to a temp file
    tmp = TMPDIR / f"{uuid.uuid4()}{suffix}"
    content = await file.read()
    tmp.write_bytes(content)

    try:
        with library_warnings() as warnings:
            model = magnetic_model_from_file(filename=str(tmp), primitive_cell=primitive_cell)
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
        "primitive_cell": primitive_cell,
    }
    tmp.unlink(missing_ok=True)

    bonds_info = [
        {"name": name, "distance": float(model.bonds[name].get("distance", 0))}
        for name in model.bonds
    ] if model.bonds else []

    num_atoms = len(model.site_properties.get("coord_atomos", []))

    return {
        "warnings": warnings,
        "session_id": sid,
        "filename": file.filename,
        "cif_text": cif_text,
        "num_atoms": num_atoms,
        "cell_size": model.lattice_properties.get("cell_size", 0),
        "primitive_cell": primitive_cell,
        "space_group_symbol": model.lattice_properties.get("space_group_symbol"),
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
        with library_warnings() as warnings:
            model = magnetic_model_from_file(
                filename=str(tmp), primitive_cell=sess.get("primitive_cell", False)
            )
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Cannot parse updated CIF: {exc}")
    if sess["cif_path"]:
        Path(sess["cif_path"]).unlink(missing_ok=True)
    model.save_cif(str(tmp))
    sess["model"] = model
    sess["cif_path"] = str(tmp)
    return {
        "warnings": warnings,
        "status": "updated",
        "cell_size": model.lattice_properties.get("cell_size"),
        "primitive_cell": sess.get("primitive_cell", False),
    }


class ValidateCifRequest(BaseModel):
    cif_text: str


@app.post("/model/{session_id}/validate")
def validate_and_update_cif(session_id: str, req: ValidateCifRequest):
    """
    Validate the CIF text from the editor and update the session model.
    Called when the user leaves tab 1. Returns model info on success,
    or raises 422 with a human-readable detail on parse failure.

    Reuses the session's `primitive_cell` setting (set at upload time,
    or via `/model/{session_id}/primitive_cell`) so hand-edits made in
    the text editor don't silently fall back to the expanded/conventional
    representation.
    """
    sess = _get_session(session_id)
    tmp = TMPDIR / f"{uuid.uuid4()}.cif"
    tmp.write_text(req.cif_text)
    try:
        with library_warnings() as warnings:
            model = magnetic_model_from_file(
                filename=str(tmp), primitive_cell=sess.get("primitive_cell", False)
            )
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
        "warnings": warnings,
        "status": "ok",
        "num_atoms": num_atoms,
        "cell_size": model.lattice_properties.get("cell_size", 0),
        "primitive_cell": sess.get("primitive_cell", False),
        "space_group_symbol": model.lattice_properties.get("space_group_symbol"),
        "bonds": bonds_info,
    }


class PrimitiveCellRequest(BaseModel):
    primitive_cell: bool


@app.post("/model/{session_id}/primitive_cell")
def set_primitive_cell(session_id: str, req: PrimitiveCellRequest):
    """
    Change the session's primitive_cell setting and immediately re-parse
    the current CIF with it. Lets the frontend toggle between the
    expanded (conventional cell) and compact (primitive cell) views
    without losing track of which one is active.

    Note: this can only recover the compact form if the CIF currently
    held by the session still declares the centering symmetry operators
    explicitly (e.g. it hasn't already been through a save/reload cycle
    with primitive_cell=False, which bakes in the fully expanded, P1
    form and discards the information needed to reduce it again).
    """
    sess = _get_session(session_id)
    if not sess["cif_path"] or not Path(sess["cif_path"]).exists():
        raise HTTPException(status_code=404, detail="No CIF file for this session.")
    cif_text = Path(sess["cif_path"]).read_text()
    tmp = TMPDIR / f"{uuid.uuid4()}.cif"
    tmp.write_text(cif_text)
    try:
        with library_warnings() as warnings:
            model = magnetic_model_from_file(
                filename=str(tmp), primitive_cell=req.primitive_cell
            )
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    Path(sess["cif_path"]).unlink(missing_ok=True)
    model.save_cif(str(tmp))
    sess["model"] = model
    sess["cif_path"] = str(tmp)
    sess["primitive_cell"] = req.primitive_cell
    return {
        "warnings": warnings,
        "status": "ok",
        "primitive_cell": req.primitive_cell,
        "num_atoms": len(model.site_properties.get("coord_atomos", [])),
        "cell_size": model.lattice_properties.get("cell_size", 0),
        "space_group_symbol": model.lattice_properties.get("space_group_symbol"),
        "cif_text": Path(sess["cif_path"]).read_text(),
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
    except ValueError as exc:  # e.g. a range too long for the cell
        raise HTTPException(status_code=422, detail=f"Bond generation failed: {exc}")
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

    num_configs = req.num_configs or None
    update_size = max(req.bunch_size, num_configs or 1)
    try:
        newconfs, cn = model.find_optimal_configurations(
            num_new_confs=num_configs,
            start=start,
            known=known,
            its=req.iterations,
            update_size=update_size,
            min_improvement=req.min_improvement,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")

    new_labels = [str(confindex(c)) for c in newconfs]
    lines = [f"\n# New configurations. {_error_bound_label(cn)}:"]
    for i, nc in enumerate(newconfs):
        lines.append(f"nan\t{_fmt_conf(nc)}\t\t# {new_labels[i]}")

    return {
        **_error_bound_fields(cn),
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
        newconfs, cn = model.optimize_independent_set(
            confs,
            length=req.num_configs or None,
            min_improvement=req.min_improvement,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")

    new_labels = [str(confindex(c)) for c in newconfs]
    lines = [f"\n# Optimal independent subset. {_error_bound_label(cn)}:"]
    for i, nc in enumerate(newconfs):
        lines.append(f"# {_fmt_conf(nc)}\t\t# {new_labels[i]}")

    return {
        **_error_bound_fields(cn),
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
        return {
            "equations": "# No configurations loaded.\n",
            "error_bound": None,
            "condition_number": None,
        }

    cm = model.coefficient_matrix(confs, False)
    equations = model.formatted_equations(
        cm, ensname=None, comments=labels, eq_format=req.output_format
    )
    cost = model.cost(confs)
    equations += "\n\n" + _error_bound_label(cost)

    return {"equations": equations, **_error_bound_fields(cost)}


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

    method = req.method
    if method is None:
        method = (
            "exact" if req.use_montecarlo is None
            else ("montecarlo" if req.use_montecarlo else "ellipse")
        )
    logger.info(
        "evaluate: %d configurations with energy, method=%s", len(valid_confs), method
    )
    try:
        with library_warnings() as warnings:
            js, jerr, chis, ar = model.compute_couplings(
                valid_confs,
                valid_energs,
                err_energs=req.energy_tolerance,
                method=method,
                mcsteps=req.mc_steps,
                mcsizefactor=req.mc_size_factor,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Coupling computation failed: {exc}")

    bond_names = list(model.bonds.keys())
    offset_energy = float(js[-1])
    js_vals = np.asarray(js[:-1], dtype=float)  # remove E0
    jerr = np.asarray(jerr, dtype=float)
    # Undetermined couplings come back with a nan error bar. nan is not
    # valid JSON, so they are reported as null with an explicit flag.
    undetermined = np.isnan(jerr[:-1])
    determined_vals = np.abs(js_vals[~undetermined])
    jmax = float(np.max(determined_vals)) if len(determined_vals) else 1.0
    if jmax == 0:
        jmax = 1.0
    fmt = req.output_format
    incompatible = bool(np.any(jerr < 0))

    # Build parameter list
    parameters = [{"name": "E0", "value": offset_energy, "error": None, "incompatible": False}]
    for i, name in enumerate(bond_names):
        if undetermined[i]:
            parameters.append({
                "name": name,
                "value": None,
                "value_normalized": None,
                "error": None,
                "error_normalized": None,
                "incompatible": False,
                "undetermined": True,
            })
            continue
        parameters.append({
            "name": name,
            "value": float(js_vals[i]),
            "value_normalized": float(js_vals[i] / jmax),
            "error": float(jerr[i]),
            "error_normalized": float(jerr[i] / jmax),
            "incompatible": bool(jerr[i] < 0),
            "undetermined": False,
        })

    # Equations for the configurations that have energies
    cm = model.coefficient_matrix(valid_confs, False)
    equations = model.formatted_equations(
        cm, ensname=None, comments=valid_labels, eq_format=fmt
    )

    # Bound inequalities
    with library_warnings() as more_warnings:
        ineqs_raw = model.bound_inequalities(
            valid_confs, valid_energs, err_energs=req.energy_tolerance
        )
    warnings.extend(more_warnings)
    # bound_inequalities returns (v, center, half_width), meaning
    # |v . J - center| <= half_width.
    inequalities = [
        {
            "coefficients": [float(c) for c in v],
            "center": float(center),
            "half_width": float(half_width),
            "lower": float(center - half_width),
            "upper": float(center + half_width),
        }
        for v, center, half_width in ineqs_raw
    ]

    return {
        "warnings": warnings,
        "jmax": jmax,
        "parameters": parameters,
        "chis": [float(c) for c in chis],
        "chi_labels": valid_labels,
        "method": method,
        "acceptance_rate": float(ar) if method == "montecarlo" else None,
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
            {"energy": _json_number(e), "config": c, "label": l}
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


# ── Import configurations from another model ─────────────────────────────────
#
# The other model is uploaded on its own (with its own primitive_cell
# option); its configurations are translated to the model of the session
# with spectrojotometer.magnetic_model (site_maps, compatible_configurations,
# map_configurations, equivalent_model).


def _import_model(sess: dict):
    other = sess.get("import_model")
    if other is None:
        raise HTTPException(status_code=400, detail="Upload the other model first.")
    return other


@app.post("/import/model")
async def import_model(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    primitive_cell: bool = Form(False),
    tol: float = Form(0.1),
):
    """
    Upload the model the configurations come from. Returns the site map
    (for each site of the session's model, its site in the other model;
    -1 if none), how it was found ("position": same frame; "structure":
    aligned with pymatgen's StructureMatcher), the energy scale (ratio of
    the numbers of sites) and the warnings of the library.
    """
    sess = _get_session(session_id)
    model = sess["model"]
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".cif", ".struct"):
        raise HTTPException(status_code=400, detail="Only .cif and .struct files are supported.")
    tmp = TMPDIR / f"{uuid.uuid4()}{suffix}"
    tmp.write_bytes(await file.read())
    try:
        with library_warnings() as warnings:
            other = magnetic_model_from_file(filename=str(tmp), primitive_cell=primitive_cell)
            site_map, _, method = site_maps(other, model, tol)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot use this model: {exc}")
    finally:
        tmp.unlink(missing_ok=True)
    sess["import_model"] = other
    sess["import_name"] = file.filename
    n_other = len(other.site_properties["coord_atomos"])
    n_model = len(model.site_properties["coord_atomos"])
    return {
        "warnings": warnings,
        "filename": file.filename,
        "num_sites": n_other,
        "site_map": site_map,
        "method": method,
        "missing_sites": [i for i, j in enumerate(site_map) if j < 0],
        "energy_scale": n_model / n_other,
    }


class ImportMapRequest(BaseModel):
    session_id: str
    configs_text: str
    # The site map to use (it can be edited); by default the automatic one.
    site_map: Optional[list[int]] = None
    tol: float = 0.1


@app.post("/import/map")
def import_map(req: ImportMapRequest):
    """
    Translate configurations of the other model to the session's model.
    Each site takes the spin of its site in the other model, and the
    energies are scaled by the ratio of the numbers of sites.
    Configurations the session's cell cannot represent (not periodic with
    its lattice) are not translated: they are kept as comments, with the
    reason, as the lines that are not configurations.
    """
    sess = _get_session(req.session_id)
    model, other = sess["model"], _import_model(sess)
    n_other = len(other.site_properties["coord_atomos"])
    scale = len(model.site_properties["coord_atomos"]) / n_other
    try:
        with library_warnings() as warnings:
            # Checks the site map once (sites without counterpart, size).
            map_configurations(other, [], model, req.tol, mapping=req.site_map)
            lines, imported, skipped = [], 0, 0
            for line in req.configs_text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    lines.append(line)
                    continue
                try:
                    energies, confs, comments = parse_spin_configurations(stripped)
                except ValueError:
                    lines.append("# not a configuration: " + stripped)
                    skipped += 1
                    continue
                conf, comment = confs[0], comments[0].strip()
                if len(conf) != n_other:
                    lines.append(f"# skipped, {len(conf)} values for {n_other} sites: {stripped}")
                    skipped += 1
                    continue
                if not compatible_configurations(other, [conf], model, req.tol)[0]:
                    lines.append("# skipped, not periodic with the current cell: " + stripped)
                    skipped += 1
                    continue
                (mapped,) = map_configurations(other, [conf], model, req.tol, mapping=req.site_map)
                energy = energies[0] * scale
                lines.append(f"{energy}\t{mapped}\t# {comment}".rstrip())
                imported += 1
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{exc}. Edit the site map, or increase the length tolerance.",
        )
    return {
        "warnings": warnings,
        "mapped_text": "\n".join(lines).strip("\n") + "\n",
        "imported": imported,
        "skipped": skipped,
        "energy_scale": scale,
    }


@app.get("/import/{session_id}/transformed_model", response_class=PlainTextResponse)
def import_transformed_model(
    session_id: str, tol: float = 0.1, site_map: Optional[str] = None, align: bool = False
):
    """
    The other model rewritten in the cell and order of sites of the
    session's model (its own positions and couplings), as a CIF file
    (spectrojotometer.magnetic_model.equivalent_model).

    `site_map`: the site map edited by hand, as comma-separated indices
    (one site of the other model per site of the session's model), as in
    /import/map: the sites of the result follow it. By default, the
    automatic map.

    `align`: if true, the result is moved rigidly onto the session's
    model (rotated and translated), each site at the image closest to its
    site of the session's model, instead of in the frame of the other
    model and wrapped into the cell (equivalent_model(..., align=True)):
    each bond (i, j, cell shift) of the session's model is then the same
    bond in the result. The file is named `<model>_aligned.cif`.
    """
    sess = _get_session(session_id)
    mapping = None
    if site_map is not None and site_map.strip():
        try:
            mapping = [int(x) for x in site_map.replace(",", " ").split()]
        except ValueError:
            raise HTTPException(
                status_code=422, detail="The site map must be a list of site indices."
            )
    try:
        transformed = equivalent_model(
            _import_model(sess), sess["model"], tol, mapping=mapping, align=align
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    path = TMPDIR / f"{uuid.uuid4()}.cif"
    transformed.save_cif(str(path))
    text = path.read_text()
    path.unlink(missing_ok=True)
    stem = Path(sess.get("import_name") or "model").stem
    return PlainTextResponse(
        text,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{stem}_{"aligned" if align else "transformed"}.cif"'
            )
        },
    )


# ── Monte Carlo simulation ───────────────────────────────────────────────────
#
# The simulations run in the browser (static/mc_engine.js, in a Web
# Worker): the server only builds the lattice of the supercell, which is
# cheap. Its size is limited before building it, by the number of bonds.

MC_MAX_BONDS = int(float(os.environ.get("VISUALBOND_MC_MAX_BONDS", 500_000)))


def _json_number(x):
    x = float(x)
    return x if np.isfinite(x) else None


@app.get("/montecarlo/{session_id}/couplings")
def montecarlo_couplings(session_id: str):
    """The couplings of the model, with their values as estimated by
    /evaluate (null if not estimated, or undetermined)."""
    model = _get_session(session_id)["model"]
    return {
        "couplings": [
            {
                "name": name,
                "distance": float(info.get("distance", 0)),
                "num_bonds": len(info.get("bonds", [])),
                "value": _json_number(info.get("value", float("nan"))),
            }
            for name, info in model.bonds.items()
        ],
        "num_sites": len(model.site_properties["coord_atomos"]),
    }


class LatticeRequest(BaseModel):
    session_id: str
    supercell: list[int] = Field(default=[2, 2, 2], min_length=3, max_length=3)
    # Values of the couplings; by default, the ones estimated by /evaluate.
    couplings: Optional[dict[str, float]] = None


@app.post("/montecarlo/lattice")
def montecarlo_lattice(req: LatticeRequest):
    """
    The Ising model of the supercell of the session, for the Monte Carlo
    simulation that runs in the browser (static/mc_engine.js): its sites,
    bonds (pairs of sites) and their couplings J, the base energy E0
    (H = E0 - sum_b J_b s_i s_j), and whether it is frustrated, with an
    odd cycle of frustrated bonds or the gauge that maps it onto a
    ferromagnet (see spectrojotometer.monte_carlo).
    """
    model = _get_session(req.session_id)["model"]
    if any(s < 1 or s > 8 for s in req.supercell):
        raise HTTPException(status_code=422, detail="Each supercell size must be between 1 and 8.")
    num_bonds = sum(len(info.get("bonds", [])) for info in model.bonds.values())
    num_bonds *= int(np.prod(req.supercell))
    if num_bonds > MC_MAX_BONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"This supercell has {num_bonds} bonds; the limit of this server is "
                f"{MC_MAX_BONDS}. Use a smaller supercell."
            ),
        )
    try:
        with library_warnings() as warnings:
            spins, bonds, j_vals, e0 = magnetic_model_to_ising(
                model, tuple(req.supercell), J_overrides=req.couplings or None
            )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc).strip("'\""))
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    j_vals = np.asarray(j_vals, dtype=np.float64)
    frustrated, gauge, cycle = detect_frustration(bonds, j_vals, len(spins))
    logger.info("montecarlo: lattice of %d sites, %d bonds", len(spins), len(bonds))
    return {
        "num_sites": len(spins),
        "bonds": bonds.tolist(),
        "J": j_vals.tolist(),
        "E0": float(e0),
        "frustrated": bool(frustrated),
        "frustration_cycle": [int(i) for i in cycle] if frustrated and cycle is not None else None,
        "gauge": None if gauge is None else [int(g) for g in gauge],
        "warnings": warnings,
    }
