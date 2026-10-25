# visualbondweb

Web interface (FastAPI backend + static frontend) for the
[spectrojotometer](https://github.com/QILPCM-IFLP-CONICET/spectrojotometer) package.

The backend is `visualbondweb/api.py` (the module installed by
`pip install .` and started by `render.yaml` and the `Dockerfile`).

## Installation and tests

Python 3.11 or later.

```bash
pip install ".[test]"
pytest
node --test tests/js/*.test.js   # the Monte Carlo engine of the browser
```

Some tests of `pytest` also run that engine with Node.js (`node`), and are
skipped without it.

The backend depends on
[spectrojotometer](https://github.com/QILPCM-IFLP-CONICET/spectrojotometer)
0.3.0 or later (the error-bound methods of `/evaluate` and the error
bound of a set of configurations), installed from git and pinned to a
specific commit in `requirements.txt` and `pyproject.toml`
(`spectrojotometer @ git+https://...spectrojotometer@<sha>`), so that a
change in the library reaches the server only through an explicit
update, tested by the CI. The current pin, `6448ef4`, is needed for the
bonds across cells in the CIF text of a session (see *Bonds and
configurations*) and for "Add bonds" pressed again; with an earlier
commit, `tests/test_cif_bonds.py` fails.

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
  `primitive_cell`, `space_group_symbol`, `bonds`, `warnings`.

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

### Warnings

The endpoints that load a model (`/model/upload`, `/model/{id}/cif`,
`/model/{id}/validate`, `/model/{id}/primitive_cell`) and `/evaluate`
return a `warnings` field: the list of warnings that spectrojotometer
emitted while serving that request (e.g. bonds of the CIF file that
could not be used, or a model incompatible with the energies). The
frontend shows them in its log panel. They also go to the server log,
like every other message of the library and of the API, through the
standard `logging` module.

Requests are served concurrently and share the library's logger; each
response only gets the warnings emitted while serving that request
(see `library_warnings` in `visualbondweb/api.py`).

### Bonds and configurations

`POST /model/add_bonds` generates the couplings of the model (`rmin`,
`rmax`, `discretization`): every pair of sites within the range, in any
cell, once. Its `supercell_size` is accepted and ignored: it only
limited the cells searched for neighbours (and bonds were missed); it
never built a supercell. A range that needs cell offsets beyond -5..4
is an error (422). Calling it again adds only the bonds the couplings do
not have yet (with spectrojotometer before 6448ef4, pressing "Add bonds"
twice added every bond a second time, doubling its coefficients).
`POST /configs/parse` and the download endpoints handle the
configuration files; see their docstrings under `/docs`.

The model of a session is kept as a CIF file (`cif_text`, the download,
the transformed model of the import), and read back whenever its text is
edited (`/model/{session_id}/validate`, `/model/{session_id}/cif`). A bond
to a site in another cell is written with the standard CIF code
`_geom_bond_site_symmetry_2` = `n_klm`: the identity operator `n` and the
cell shift + 5 along each axis (`1_655` for one cell along a, `1_545` for
one cell back along b). CIF files written by earlier versions (by
spectrojotometer before 6448ef4, or downloaded from this server before
this update), with the shift itself in each digit (`1_100`), are still
read, with a warning in `warnings`; the text of the session is then in
the standard convention. With those versions, editing the CIF text moved
such bonds 4 or 5 cells away (69.7 Å instead of 2.94 Å in a spinel):
the fits were right, since they only depend on the sites of each bond,
but not the bonds of the model, the Monte Carlo lattice built from them,
or the CIF downloaded afterwards.

Three endpoints report the **error bound** of a set of configurations,
`error_bound`: the largest error of a coupling per unit error in the
energies (`MagneticModel.cost` of spectrojotometer). If every energy is
known within δE, no coupling has an error larger than δE ×
`error_bound`. It is `null` when the configurations do not determine
every coupling. `condition_number` is the same value as text (`"inf"`
for undetermined couplings), kept for older clients; despite its name, it
is not the condition number of the coefficient matrix.

* `POST /configs/equations` (`session_id`, `configs_text`,
  `output_format`): parses the configurations, stores them in the
  session and returns the formatted linear system, `equations`, with
  its `error_bound`.
* `POST /configs/optimize` (`session_id`, `num_configs`, `bunch_size` =
  10, `iterations` = 100, `min_improvement` = 0.05): proposes new
  configurations that, together with the ones of the session with known
  energies, minimize the error bound. Returns `new_configs_text`
  (energies `nan`), `num_new_configs` and `error_bound`.
* `POST /configs/optimal_independent_set` (`session_id`, `num_configs`,
  `min_improvement` = 0.05): the subset of the configurations of the
  session with the smallest error bound. Returns `subset_text`,
  `num_configs` and `error_bound`.

When `num_configs` is omitted (or 0), the number of configurations is
chosen automatically: the smallest set that determines every coupling,
enlarged while each extra configuration reduces the error bound by at
least `min_improvement` (5% by default).

### Importing configurations from another model

The configurations computed for another model (e.g. a distorted phase,
or a different cell) can be translated to the model of the session. In
the frontend: tab 2, "Load configs from other model".

* `POST /import/model` (form: `session_id`, `file`, `primitive_cell`,
  `tol` = 0.1): uploads the other model. Returns `site_map` (for each
  site of the session's model, its site in the other model; -1 if
  none), `method` (`position`: same frame; `structure`: aligned with
  pymatgen's StructureMatcher, for a different origin, orientation or
  cell), `missing_sites`, `energy_scale` (ratio of the numbers of sites)
  and `warnings`.
* `POST /import/map` (`session_id`, `configs_text`, `site_map`, `tol`):
  translates the configurations; the energies are scaled by
  `energy_scale`. Configurations the session's cell cannot represent
  (not periodic with its lattice) are kept as comments with the reason.
  Returns `mapped_text`, `imported`, `skipped`. A site without
  counterpart is an error (422): edit `site_map`, or increase `tol`.
* `GET /import/{session_id}/transformed_model` (`tol`, `site_map`): the
  other model rewritten in the cell and order of sites of the session's
  model (its own positions and couplings), as a CIF file. With
  `site_map` (comma-separated indices, as edited for `/import/map`), its
  sites follow that map: the same supercell of the other model, in
  another order, so that the configurations imported with it keep their
  coefficients. Each site of the other model must appear as many times
  as the supercell holds it (422 otherwise). With `align=true`, the
  model is moved rigidly onto the session's model, each site at the
  image closest to its site of the session's model, instead of in the
  frame of the other model and wrapped into the cell
  (`equivalent_model(..., align=True)`): each bond (i, j, cell shift) of
  the session's model is then the same bond in it, e.g. to compare the
  bond lengths of a distorted phase with those of the session's model.
  The file is then `<model>_aligned.cif` instead of
  `<model>_transformed.cif`. In the frontend, "Align onto the current
  model", next to "Export transformed model…".

### Monte Carlo: `POST /montecarlo/lattice`

Tab 4 simulates the Ising model of the session at finite temperature. The
simulation runs in the browser, in a Web Worker
(`visualbondweb/static/mc_engine.js`, a port of
`spectrojotometer.monte_carlo`: Metropolis with restarts, Wolff and
parallel tempering, with the same observables, choice of algorithm and
warnings); the server only builds the lattice of the supercell.

* `GET /montecarlo/{session_id}/couplings`: the couplings, with the
  values estimated by `/evaluate` (`null` if not estimated).
* `POST /montecarlo/lattice` (`session_id`, `supercell` = [2, 2, 2],
  `couplings`: values to use instead of the estimated ones): the Ising
  model H = E0 − Σ J s_i s_j of the supercell: `num_sites`, `bonds`
  (pairs of sites), `J`, `E0`, `frustrated`, `frustration_cycle` (an odd
  cycle of frustrated bonds) and `gauge` (the signs that map an
  unfrustrated model onto a ferromagnet, for Wolff).

The engine returns, per temperature, `T`, `E`, `M`, `M2`, `M4`, `chi`,
`chi_connected`, `C`, `U4`; and `algorithm`, `N`, `frustrated`,
`frustration_cycle`, `Tc_peak`, `frustration_factor` (and
`swap_acceptance` with parallel tempering, `mean_cluster_size` with
Wolff). Its random numbers differ from the library's, so the same seed
gives reproducible results in the browser, but not the same as
spectrojotometer.

The size of the lattice is limited by the environment variable
`VISUALBOND_MC_MAX_BONDS` (bonds of the supercell, default 500000),
checked before building it.

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
* `warnings`: warnings of spectrojotometer while computing (see below).
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
