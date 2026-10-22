# visualbondweb

Web interface (FastAPI backend + static frontend) for the
[spectrojotometer](https://github.com/mmatera/spectrojotometer) package.

Note: the backend is `visualbondweb/api.py` (the module installed by
`pip install .` and started by `render.yaml`). `api_main.py`, in the repo
root, is an older copy that no deployment uses; it is out of date
(it lacks, e.g., `supercell_size` and the error-bound `method`) and
should not be used as a reference.

## Installation and tests

```bash
pip install ".[test]"
pytest
```

The backend depends on
[spectrojotometer](https://github.com/QILPCM-IFLP-CONICET/spectrojotometer),
installed from git. The `/evaluate` endpoint uses
`compute_couplings(method=...)`, so it needs a version of the library
that includes it. Pinning the dependency to a specific commit
(`spectrojotometer @ git+https://...spectrojotometer@<sha>`) avoids
deploying an untested combination when the library changes.

## Frontend

The web UI (`visualbondweb/static/index.html`) has a **"Use primitive
cell (centering symmetries)"** checkbox next to the file upload zone on
Tab 1, and in-app documentation for it under **Help → Loading a model
file → Primitive cell** (`visualbondweb/static/help.html`). It mirrors
the `primitive_cell` API field/endpoint described below.

## API overview

Interactive docs (generated from the endpoint docstrings) are available
at `/docs` once the server is running.

### Sessions

* `POST /session/new` — create an empty session, returns `session_id`.
* `DELETE /session/{session_id}` — remove a session and its temp files.

### Loading a model

* `POST /model/upload` — multipart upload of a `.cif` or `.struct` file.
  Form fields:
  * `file` (required)
  * `primitive_cell` (optional, bool, default `false`) — see below.

  Returns `session_id`, `cif_text`, `num_atoms`, `cell_size`,
  `primitive_cell`, `space_group_symbol`, `bonds`.

* `GET /model/{session_id}/cif` — current CIF text for the session.
* `POST /model/{session_id}/cif` — replace the model from an edited CIF
  upload. Reuses the session's `primitive_cell` setting.
* `POST /model/{session_id}/validate` — validate/apply CIF text edited
  in the frontend's text tab. Reuses the session's `primitive_cell`
  setting. Body: `{"cif_text": "..."}`.
* `POST /model/{session_id}/primitive_cell` — change the session's
  `primitive_cell` setting and re-parse the current CIF with it. Body:
  `{"primitive_cell": true|false}`.

#### `primitive_cell`

Mirrors `spectrojotometer.model_io.magnetic_model_from_file`'s
`primitive_cell` parameter (see that package's README for the full
explanation):

* `false` (default): CIFs whose symmetry loop declares extra operators
  (e.g. the centering translations of an F, I, C, A or B centered
  lattice) get their asymmetric-unit atoms expanded to fill the
  conventional cell.
* `true`: atoms are kept exactly as declared in the CIF (never
  expanded); the model's Bravais vectors switch to a primitive basis.
  Only works if the CIF's symmetry operators are pure translations
  (a centered lattice, not a general point-group symmetry) — otherwise
  the request fails with a 422 explaining why.

**This setting is sticky per session, on purpose.** Once a session is
created with `primitive_cell=true`, every subsequent reload
(`/model/{id}/cif`, `/model/{id}/validate`) automatically reuses it —
the frontend does not need to (and should not) resend it on every call.
This avoids a real failure mode we hit during development: a single
reload that silently omits the flag bakes the model back into its fully
expanded, plain-P1 form via `save_cif`, and from that point on there is
no information left to recover the compact form automatically. Use
`POST /model/{session_id}/primitive_cell` to change the setting
mid-session instead of trying to smuggle it into another endpoint.

### Bonds and configurations

See the endpoint docstrings under `/docs` for `add_bonds`,
`optimize_configs`, `optimal_independent_set`, `equations`,
and the configuration import/export/download endpoints.

### Estimating the couplings: `POST /evaluate`

Fits the coupling constants to the configurations with known energies,
and bounds their uncertainty. The couplings compatible with the data
are those that reproduce every energy within `energy_tolerance`; they
form a convex polytope, and each error bar is the half width of a box
around it. See spectrojotometer's README ("Estimating the couplings and
their uncertainties") for the details.

Request body:

| field | default | meaning |
|---|---|---|
| `session_id` | | |
| `configs_text` | | `.spin` text: `energy [config] # label` per line; `nan` energies are ignored |
| `energy_tolerance` | `0.001` | maximum error of each energy, in the same units |
| `method` | `"exact"` | `"exact"`: exact box of the polytope (linear programming). `"montecarlo"`: random sampling, an inner estimate. `"ellipse"`: box of an ellipsoid containing the polytope, an outer estimate |
| `mc_steps` | `1000` | Monte Carlo samples (only for `"montecarlo"`) |
| `mc_size_factor` | `1.0` | scale of the Monte Carlo sampling ellipsoid; keep at 1 |
| `output_format` | `"plain"` | format of `equations`: `plain`, `latex` or `wolfram` |
| `use_montecarlo` | | deprecated; if `method` is absent, `true` means `"montecarlo"` and `false` means `"ellipse"` |

Response:

* `method`: the method used.
* `parameters`: one entry per parameter. The first one, `E0`, is the
  average energy of the configurations (the coupling columns are
  centered when solving). For each coupling: `value`, `error` (half
  width of the box), `value_normalized` and `error_normalized`
  (divided by `jmax`), `incompatible`, and `undetermined`. An
  undetermined coupling (one that the configurations do not fix) has
  `null` value and error.
* `jmax`: the largest determined coupling, in absolute value.
* `incompatible`: `true` if no set of couplings reproduces all the
  energies within the tolerance. The errors are then negative.
* `chis`, `chi_labels`: residual of the least-squares fit for each
  configuration, relative to the tolerance. Values beyond ±1 point to
  configurations the model cannot reproduce.
* `acceptance_rate`: fraction of accepted Monte Carlo samples (`null`
  for the other methods).
* `inequalities`: constraints on combinations of couplings,
  `lower ≤ coefficients · J ≤ upper`. Also given as `center` and
  `half_width` (`|coefficients · J − center| ≤ half_width`).
* `equations`: the linear system, formatted.
* `num_configs_used`.
