# visualbondweb

Web interface (FastAPI backend + static frontend) for the
[spectrojotometer](https://github.com/mmatera/spectrojotometer) package.

Note: there are currently two near-identical copies of the backend —
`api_main.py` (repo root) and `visualbondweb/api.py` (the module actually
installed by `pip install .`, per `pyproject.toml`). Keep changes to both
in sync until they're unified into a single module.

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
`optimize_configs`, `optimal_independent_set`, `equations`, `evaluate`,
and the configuration import/export/download endpoints.
