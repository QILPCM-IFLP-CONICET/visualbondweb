// Tests of static/mc_engine.js, run with `node --test tests/js/*.test.js`: the three
// algorithms against the exact enumeration of small lattices.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const mc = require(path.join(__dirname, "..", "..", "visualbondweb", "static", "mc_engine.js"));

/** Periodic square lattice L x L with coupling J (H = -sum J s_i s_j). */
function square(L, J) {
  const idx = (x, y) => ((x + L) % L) * L + ((y + L) % L);
  const bonds = [];
  for (let x = 0; x < L; x++) for (let y = 0; y < L; y++) bonds.push([idx(x, y), idx(x + 1, y)], [idx(x, y), idx(x, y + 1)]);
  const gauge = [];
  for (let x = 0; x < L; x++) for (let y = 0; y < L; y++) gauge.push(J < 0 && (x + y) % 2 ? -1 : 1);
  return { num_sites: L * L, bonds, J: bonds.map(() => J), E0: 0, frustrated: false, frustration_cycle: null, gauge };
}

/** Periodic triangular lattice L x L, antiferromagnetic: frustrated. */
function triangular(L) {
  const idx = (x, y) => ((x + L) % L) * L + ((y + L) % L);
  const bonds = [];
  for (let x = 0; x < L; x++) for (let y = 0; y < L; y++) {
    bonds.push([idx(x, y), idx(x + 1, y)], [idx(x, y), idx(x, y + 1)], [idx(x, y), idx(x + 1, y + 1)]);
  }
  return { num_sites: L * L, bonds, J: bonds.map(() => -1), E0: 0.5, frustrated: true, frustration_cycle: [0, 1, 4, 0], gauge: null };
}

/** Exact E per site, C per site and <m^2> (m = sum w_i s_i / N). */
function exact(lat, T, weights) {
  const N = lat.num_sites;
  let Z = 0, E1 = 0, E2 = 0, M2 = 0, Emin = Infinity;
  const states = [];
  for (let c = 0; c < 2 ** N; c++) {
    let e = lat.E0, m = 0;
    for (let k = 0; k < lat.bonds.length; k++) {
      const [i, j] = lat.bonds[k];
      e -= lat.J[k] * (((c >> i) & 1) * 2 - 1) * (((c >> j) & 1) * 2 - 1);
    }
    for (let i = 0; i < N; i++) m += weights[i] * (((c >> i) & 1) * 2 - 1);
    states.push([e, m / N]);
    Emin = Math.min(Emin, e);
  }
  for (const [e, m] of states) {
    const w = Math.exp(-(e - Emin) / T);
    Z += w; E1 += w * e; E2 += w * e * e; M2 += w * m * m;
  }
  E1 /= Z; E2 /= Z; M2 /= Z;
  return { E: E1 / N, C: (E2 - E1 * E1) / (T * T * N), M2 };
}

const close = (x, y, tol, what) => assert.ok(Math.abs(x - y) <= tol, `${what}: ${x} vs exact ${y} (tolerance ${tol})`);

for (const algorithm of ["wolff", "metropolis", "pt"]) {
  test(`${algorithm} samples the Boltzmann distribution of a 4x4 antiferromagnet`, () => {
    const lat = square(4, -1);
    // Metropolis decorrelates slowly at low temperature: more sweeps.
    const n_sweeps = algorithm === "wolff" ? 40000 : 200000;
    const params = { T_min: 1.5, T_max: 3.5, n_temps: 3, n_equil: 500, n_sweeps, algorithm, seed: 7 };
    const { results, warnings } = mc.runIsingMC(lat, params);
    assert.equal(results.algorithm, algorithm);
    assert.deepEqual(warnings, []);
    // Wolff measures M in the gauge basis (staggered); the others, uniform.
    const weights = algorithm === "wolff" ? lat.gauge : lat.gauge.map(() => 1);
    results.T.forEach((T, t) => {
      const ex = exact(lat, T, weights);
      close(results.E[t], ex.E, 0.01, `E(T=${T})`);
      close(results.C[t], ex.C, Math.max(0.08 * ex.C, 0.01), `C(T=${T})`);
      close(results.M2[t], ex.M2, 0.01, `M2(T=${T})`);
    });
  });
}

test("Wolff: auto on an unfrustrated model; a sweep flips about N spins", () => {
  const lat = square(8, -1);
  const { results } = mc.runIsingMC(lat, { T_min: 0.2, T_max: 100, n_temps: 3, n_equil: 50, n_sweeps: 200, seed: 3 });
  assert.equal(results.algorithm, "wolff");
  assert.ok(results.mean_cluster_size > 1 && results.mean_cluster_size < 64, String(results.mean_cluster_size));
  assert.equal(results.N, 64);
  assert.equal(results.frustrated, false);
});

test("a frustrated model: auto uses Metropolis and warns; Wolff is refused; PT is exact", () => {
  const lat = triangular(3);
  const auto = mc.runIsingMC(lat, { T_min: 1, T_max: 2, n_temps: 2, n_equil: 10, n_sweeps: 10 });
  assert.equal(auto.results.algorithm, "metropolis");
  assert.match(auto.warnings[0], /^Frustrated model detected \(odd cycle: \[0, 1, 4, 0\]\)\. Using Metropolis\./);
  assert.deepEqual(auto.results.frustration_cycle, [0, 1, 4, 0]);
  assert.throws(() => mc.runIsingMC(lat, { algorithm: "wolff" }), /not ergodic/);

  const { results } = mc.runIsingMC(lat, { T_min: 0.3, T_max: 3, n_temps: 6, n_equil: 1000, n_sweeps: 40000, algorithm: "pt", seed: 2, swap_every: 2 });
  assert.ok(results.swap_acceptance > 0.3, String(results.swap_acceptance));
  results.T.forEach((T, t) => close(results.E[t], exact(lat, T, lat.J.map(() => 1)).E, 0.01, `E(T=${T})`));
});

test("Metropolis restarts warn when the replicas disagree", () => {
  // At very low temperature, single-spin flips cannot leave the domains
  // of a random start: the replicas end with different energies.
  const lat = square(8, 1);
  const { warnings } = mc.runIsingMC(lat, { T_min: 0.05, T_max: 0.05, n_temps: 1, n_equil: 1, n_sweeps: 2, algorithm: "metropolis", n_restarts: 4, seed: 1 });
  assert.match(warnings.join(" "), /The energies of the replicas differ by/);
});

test("estimateTc interpolates the peak; the temperatures are geometric", () => {
  const T = [1, 2, 3, 4];
  const chi = T.map((t) => 5 - (t - 2.3) ** 2);
  close(mc.estimateTc(T, chi), 2.3, 1e-12, "Tc");
  assert.equal(mc.estimateTc([1, 2, 3], [3, 2, 1]), 1);
  const g = mc.geomspace(0.5, 4, 4);
  close(g[1] / g[0], 2, 1e-12, "ratio");
  assert.deepEqual([g[0], g[3]], [0.5, 4]);
  assert.deepEqual(mc.geomspace(0.5, 4, 1), [0.5]);
  assert.equal(mc.frustrationFactor(2, [1, -1], 4), 4);
});

test("invalid parameters are rejected", () => {
  const lat = square(2, 1);
  assert.throws(() => mc.runIsingMC(lat, { T_min: 3, T_max: 1 }), /T_max must be at least T_min/);
  assert.throws(() => mc.runIsingMC(lat, { n_temps: 0 }), /n_temps/);
  assert.throws(() => mc.runIsingMC(lat, { algorithm: "glauber" }), /Unknown algorithm/);
  // The page validates before asking the server for the lattice.
  assert.throws(() => mc.validateParams({ T_min: 3, T_max: 1 }), /T_max must be at least T_min/);
  assert.throws(() => mc.validateParams({ n_sweeps: 0 }), /n_sweeps/);
  mc.validateParams({});
  assert.throws(() => mc.runIsingMC({ num_sites: 2, bonds: [[0, 5]], J: [1], E0: 0 }, {}), /out of range/);
});
