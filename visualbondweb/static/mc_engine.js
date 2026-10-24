/*
 * Monte Carlo simulation of the Ising model H = E0 - sum_b J_b s_i s_j
 * (s = +-1), in the browser: a port of spectrojotometer.monte_carlo
 * (Metropolis with restarts, Wolff with a gauge transformation, parallel
 * tempering, and the estimate of Tc), with the same definition of a
 * sweep, the same observables, the same choice of algorithm and the same
 * warnings. The lattice (sites, bonds, couplings, frustration and gauge)
 * is built by the server: POST /montecarlo/lattice.
 *
 * The random numbers come from a xoshiro128** generator, so a seed gives
 * reproducible results here, but not the same as the library.
 *
 * Loaded by mc_worker.js (a Web Worker) in the browser, and by Node in
 * the tests (module.exports).
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------------
  // Random numbers: xoshiro128**, seeded with splitmix32.
  // ---------------------------------------------------------------------

  function makeRng(seed) {
    let x = (seed >>> 0) ^ 0x9e3779b9;
    function splitmix32() {
      x = (x + 0x9e3779b9) | 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    }
    let s0 = splitmix32(), s1 = splitmix32(), s2 = splitmix32(), s3 = splitmix32();
    if ((s0 | s1 | s2 | s3) === 0) s0 = 1;
    function next() {
      const r = Math.imul(rotl(Math.imul(s1, 5), 7), 9) >>> 0;
      const t = s1 << 9;
      s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 11);
      return r;
    }
    return {
      /** Uniform in [0, 1). */
      random: () => next() / 4294967296,
      /** Uniform integer in [0, n). */
      int: (n) => Math.floor((next() / 4294967296) * n),
      /** A seed for another generator. */
      seed: () => next(),
    };
  }

  function rotl(v, k) {
    return (v << k) | (v >>> (32 - k));
  }

  // ---------------------------------------------------------------------
  // The lattice: bonds (pairs of sites) and neighbour lists (CSR).
  // ---------------------------------------------------------------------

  function makeLattice(N, bonds, J) {
    const nb = bonds.length;
    const bi = new Int32Array(nb), bj = new Int32Array(nb), bJ = new Float64Array(nb);
    const deg = new Int32Array(N);
    for (let k = 0; k < nb; k++) {
      const i = bonds[k][0], j = bonds[k][1];
      if (!(i >= 0 && i < N && j >= 0 && j < N)) throw new Error(`bond ${k} refers to a site out of range`);
      bi[k] = i; bj[k] = j; bJ[k] = J[k];
      deg[i]++; deg[j]++;
    }
    const start = new Int32Array(N + 1);
    for (let i = 0; i < N; i++) start[i + 1] = start[i] + deg[i];
    const idx = new Int32Array(start[N]), nJ = new Float64Array(start[N]);
    const fill = start.slice(0, N);
    for (let k = 0; k < nb; k++) {
      const i = bi[k], j = bj[k];
      idx[fill[i]] = j; nJ[fill[i]++] = bJ[k];
      idx[fill[j]] = i; nJ[fill[j]++] = bJ[k];
    }
    return { N, bi, bj, bJ, start, idx, nJ };
  }

  /** The same lattice with every coupling replaced by |J| (gauge basis). */
  function absLattice(lat) {
    return Object.assign({}, lat, { bJ: lat.bJ.map(Math.abs), nJ: lat.nJ.map(Math.abs) });
  }

  function energy(lat, spins, E0) {
    let e = 0;
    const { bi, bj, bJ } = lat;
    for (let k = 0; k < bJ.length; k++) e += bJ[k] * spins[bi[k]] * spins[bj[k]];
    return E0 - e;
  }

  function randomSpins(N, rng) {
    const s = new Int8Array(N);
    for (let i = 0; i < N; i++) s[i] = rng.random() < 0.5 ? -1 : 1;
    return s;
  }

  // ---------------------------------------------------------------------
  // Metropolis
  // ---------------------------------------------------------------------

  function metropolisSweep(lat, spins, beta, rng) {
    const { N, start, idx, nJ } = lat;
    for (let n = 0; n < N; n++) {
      const i = rng.int(N);
      let h = 0;
      for (let k = start[i]; k < start[i + 1]; k++) h += nJ[k] * spins[idx[k]];
      const dE = 2 * spins[i] * h;
      if (dE <= 0 || rng.random() < Math.exp(-beta * dE)) spins[i] = -spins[i];
    }
  }

  // ---------------------------------------------------------------------
  // Wolff (on the gauge basis, where every coupling is |J| >= 0)
  // ---------------------------------------------------------------------

  /**
   * Flip Wolff clusters until minFlipped spins have been flipped in total
   * or maxClusters clusters have been flipped. A neighbour is added with
   * probability pAdd (1 - exp(-2 beta |J|)) only if it still has the sign
   * of the cluster; each site is flipped as soon as it is added.
   * Returns [clusters, flipped].
   */
  function wolffClusters(lat, spins, pAdd, rng, stack, minFlipped, maxClusters) {
    const { N, start, idx } = lat;
    let clusters = 0, flipped = 0;
    while (flipped < minFlipped && clusters < maxClusters) {
      const seed = rng.int(N);
      const s = spins[seed];
      spins[seed] = -s;
      stack[0] = seed;
      let top = 1, size = 1;
      while (top > 0) {
        const u = stack[--top];
        for (let k = start[u]; k < start[u + 1]; k++) {
          const v = idx[k];
          if (spins[v] === s && rng.random() < pAdd[k]) {
            spins[v] = -s;
            stack[top++] = v;
            size++;
          }
        }
      }
      clusters++;
      flipped += size;
    }
    return [clusters, flipped];
  }

  // ---------------------------------------------------------------------
  // Observables
  // ---------------------------------------------------------------------

  function newAccumulator() {
    return { M: 0, Mabs: 0, M2: 0, M4: 0, E: 0, E2: 0, n: 0 };
  }

  function accumulate(a, M, E) {
    a.M += M; a.Mabs += Math.abs(M); a.M2 += M * M; a.M4 += M * M * M * M;
    a.E += E; a.E2 += E * E; a.n++;
  }

  /** The averages of spectrojotometer (IsingMonteCarlo.sample). */
  function observables(a, N, beta) {
    const n = a.n;
    const M = a.M / n, Mabs = a.Mabs / n, M2 = a.M2 / n, M4 = a.M4 / n;
    const E = a.E / n, E2 = a.E2 / n;
    return {
      M: Mabs / N,
      M2: M2 / (N * N),
      M4: M4 / (N * N * N * N),
      E: E / N,
      chi: (beta / N) * (M2 - M * M),
      chi_connected: (beta / N) * (M2 - Mabs * Mabs),
      C: ((beta * beta) / N) * (E2 - E * E),
      U4: M2 > 0 ? 1 - M4 / (3 * M2 * M2) : 0,
    };
  }

  function sumSpins(spins) {
    let m = 0;
    for (let i = 0; i < spins.length; i++) m += spins[i];
    return m;
  }

  const KEYS = ["M", "M2", "M4", "E", "chi", "chi_connected", "C", "U4"];

  function emptyResults() {
    const r = { T: [] };
    for (const k of KEYS) r[k] = [];
    return r;
  }

  function pushObservables(results, T, obs) {
    results.T.push(T);
    for (const k of KEYS) results[k].push(obs[k]);
  }

  // ---------------------------------------------------------------------
  // The three algorithms, as in spectrojotometer.monte_carlo.runner
  // ---------------------------------------------------------------------

  function runWolff(lat, E0, gauge, Ts, p, rng, notify) {
    const glat = absLattice(lat);
    const N = lat.N;
    const spins = new Int8Array(N).fill(1); // ordered state in the gauge basis
    const stack = new Int32Array(N);
    const pAdd = new Float64Array(glat.nJ.length);
    const results = emptyResults();
    let totalClusters = 0, totalFlipped = 0;
    Ts.forEach((T, t) => {
      const beta = 1 / T;
      for (let k = 0; k < pAdd.length; k++) pAdd[k] = -Math.expm1(-2 * beta * glat.nJ[k]);
      // Equilibration, which estimates the clusters of a sweep at this T.
      // (Stopping when N spins have been flipped would bias measurements.)
      let estClusters = 0, estFlipped = 0;
      const equil = p.n_equil > 0 ? p.n_equil : 10;
      for (let s = 0; s < equil; s++) {
        const [c, f] = wolffClusters(glat, spins, pAdd, rng, stack, N, N);
        estClusters += c; estFlipped += f;
      }
      const perSweep = Math.max(1, Math.round((N * estClusters) / estFlipped));
      const acc = newAccumulator();
      for (let s = 0; s < p.n_sweeps; s++) {
        const [c, f] = wolffClusters(glat, spins, pAdd, rng, stack, Infinity, perSweep);
        totalClusters += c; totalFlipped += f;
        // M in the gauge basis (the staggered magnetization of an
        // antiferromagnet); E does not depend on the basis.
        accumulate(acc, sumSpins(spins), energy(glat, spins, E0));
      }
      pushObservables(results, T, observables(acc, N, beta));
      notify(t + 1, Ts.length, `T = ${T.toFixed(3)}`);
    });
    results.mean_cluster_size = totalClusters ? totalFlipped / totalClusters : null;
    return results;
  }

  function runMetropolis(lat, E0, Ts, p, rng, notify, warnings) {
    const N = lat.N;
    const R = p.n_restarts;
    const sums = {};
    for (const k of KEYS) sums[k] = new Float64Array(Ts.length);
    const energies = Ts.map(() => new Float64Array(R));
    let step = 0;
    for (let r = 0; r < R; r++) {
      const spins = randomSpins(N, rng);
      const rrng = makeRng(rng.seed());
      Ts.forEach((T, t) => {
        const beta = 1 / T;
        for (let s = 0; s < p.n_equil; s++) metropolisSweep(lat, spins, beta, rrng);
        const acc = newAccumulator();
        for (let s = 0; s < p.n_sweeps; s++) {
          metropolisSweep(lat, spins, beta, rrng);
          accumulate(acc, sumSpins(spins), energy(lat, spins, E0));
        }
        const obs = observables(acc, N, beta);
        for (const k of KEYS) sums[k][t] += obs[k];
        energies[t][r] = obs.E;
        notify(++step, R * Ts.length, `T = ${T.toFixed(3)}`);
      });
    }
    const results = { T: Ts.slice() };
    for (const k of KEYS) results[k] = Array.from(sums[k], (v) => v / R);
    if (R > 1) {
      // Spread of the energies of the replicas at the same temperature,
      // relative to the energy scale of the model, sum |J| / N.
      let sumAbsJ = 0;
      for (const j of lat.bJ) sumAbsJ += Math.abs(j);
      const scale = Math.max(sumAbsJ / N, 1e-12);
      let spread = 0;
      for (const e of energies) spread = Math.max(spread, (Math.max(...e) - Math.min(...e)) / scale);
      if (spread > 0.05) {
        warnings.push(
          `The energies of the replicas differ by ${(100 * spread).toFixed(1)}% of the ` +
          "energy scale of the model: the sampling may not be ergodic. " +
          "Increase n_equil / n_sweeps, or use algorithm='pt'."
        );
      }
    }
    return results;
  }

  function runParallelTempering(lat, E0, Ts, p, rng, notify) {
    const N = lat.N, R = Ts.length;
    let reps = Ts.map(() => randomSpins(N, rng));
    const rngs = Ts.map(() => makeRng(rng.seed()));
    const E = new Float64Array(R);
    let attempts = 0, accepted = 0;
    const total = p.n_equil + p.n_sweeps;
    const every = Math.max(1, Math.floor(total / 100));
    const acc = Ts.map(() => newAccumulator());
    for (let s = 0; s < total; s++) {
      for (let r = 0; r < R; r++) metropolisSweep(lat, reps[r], 1 / Ts[r], rngs[r]);
      const measuring = s >= p.n_equil;
      const swapping = rng.random() < 1 / p.swap_every;
      if (measuring || swapping) {
        for (let r = 0; r < R; r++) E[r] = energy(lat, reps[r], E0);
      }
      if (swapping) {
        for (let i = 0; i < R - 1; i++) {
          const arg = (1 / Ts[i] - 1 / Ts[i + 1]) * (E[i] - E[i + 1]);
          attempts++;
          if (arg >= 0 || rng.random() < Math.exp(arg)) {
            [reps[i], reps[i + 1]] = [reps[i + 1], reps[i]];
            [E[i], E[i + 1]] = [E[i + 1], E[i]];
            accepted++;
          }
        }
      }
      if (measuring) {
        for (let r = 0; r < R; r++) accumulate(acc[r], sumSpins(reps[r]), E[r]);
      }
      if ((s + 1) % every === 0 || s + 1 === total) {
        notify(s + 1, total, s < p.n_equil ? "Equilibrating replicas" : "Measuring replicas");
      }
    }
    const results = emptyResults();
    Ts.forEach((T, r) => pushObservables(results, T, observables(acc[r], N, 1 / T)));
    results.swap_acceptance = attempts ? accepted / attempts : 0;
    return results;
  }

  // ---------------------------------------------------------------------
  // Analysis
  // ---------------------------------------------------------------------

  /** Tc as the maximum of chi, with a parabolic interpolation. */
  function estimateTc(T, chi) {
    let k = 0;
    for (let i = 1; i < chi.length; i++) if (chi[i] > chi[k]) k = i;
    if (k > 0 && k < T.length - 1) {
      const [x0, x1, x2] = [T[k - 1], T[k], T[k + 1]];
      const [y0, y1, y2] = [chi[k - 1], chi[k], chi[k + 1]];
      const d01 = (y1 - y0) / (x1 - x0), d12 = (y2 - y1) / (x2 - x1);
      const a = (d12 - d01) / (x2 - x0);
      const b = d01 - a * (x0 + x1);
      if (a < 0) return -b / (2 * a);
    }
    return T[k];
  }

  /** f = Tc / (sum_bonds |J| / N). Low if there is frustration. */
  function frustrationFactor(Tc, J, N) {
    if (!J.length || !N) return null;
    let s = 0;
    for (const j of J) s += Math.abs(j);
    const scale = s / N;
    return scale > 0 ? Tc / scale : null;
  }

  function geomspace(a, b, n) {
    if (n === 1) return [a];
    const r = Math.log(b / a) / (n - 1);
    return Array.from({ length: n }, (_, i) => (i === n - 1 ? b : a * Math.exp(r * i)));
  }

  function finite(x) {
    return Number.isFinite(x) ? x : null;
  }

  // ---------------------------------------------------------------------
  // Entry point
  // ---------------------------------------------------------------------

  const DEFAULTS = {
    T_min: 0.5, T_max: 4.0, n_temps: 20, n_equil: 500, n_sweeps: 2000,
    algorithm: "auto", n_restarts: 1, seed: 42, swap_every: 10,
  };

  function checkParams(p) {
    const int = (k, lo, hi) => {
      if (!Number.isInteger(p[k]) || p[k] < lo || (hi !== undefined && p[k] > hi)) {
        throw new Error(`${k} must be an integer` + (hi !== undefined ? ` between ${lo} and ${hi}.` : ` of at least ${lo}.`));
      }
    };
    if (!(p.T_min > 0) || !(p.T_max > 0)) throw new Error("The temperatures must be positive.");
    if (p.T_max < p.T_min) throw new Error("T_max must be at least T_min.");
    int("n_temps", 1, 200); int("n_equil", 0); int("n_sweeps", 1); int("n_restarts", 1, 20); int("swap_every", 1);
    if (!["auto", "metropolis", "wolff", "pt"].includes(p.algorithm)) throw new Error(`Unknown algorithm: ${p.algorithm}`);
  }

  /**
   * lattice: as returned by POST /montecarlo/lattice (num_sites, bonds,
   * J, E0, frustrated, frustration_cycle, gauge). params: T_min, T_max,
   * n_temps, n_equil, n_sweeps, algorithm, n_restarts, seed, swap_every.
   * progress(done, total, message) is called as the simulation advances.
   *
   * Returns {results, warnings}; results has the keys of run_ising_mc
   * (without curie_weiss), and mean_cluster_size with Wolff.
   */
  function runIsingMC(lattice, params, progress) {
    const p = Object.assign({}, DEFAULTS, params);
    checkParams(p);
    const N = lattice.num_sites;
    const lat = makeLattice(N, lattice.bonds, lattice.J);
    const warnings = [];
    const cycle = lattice.frustration_cycle || [];
    let algorithm = p.algorithm;
    if (algorithm === "auto") {
      algorithm = lattice.frustrated ? "metropolis" : "wolff";
      if (lattice.frustrated) {
        warnings.push(
          `Frustrated model detected (odd cycle: [${cycle.slice(0, 6).join(", ")}]` +
          `${cycle.length > 6 ? "..." : ""}). Using Metropolis. ` +
          "For an accurate Tc, consider algorithm='pt'."
        );
      }
    } else if (algorithm === "wolff" && lattice.frustrated) {
      throw new Error("Wolff is not ergodic for frustrated models. Use 'metropolis' or 'pt'.");
    }
    const notify = progress || (() => {});
    const Ts = geomspace(p.T_min, p.T_max, p.n_temps);
    const rng = makeRng(p.seed === null || p.seed === undefined ? (Math.random() * 2 ** 32) >>> 0 : p.seed);
    notify(0, algorithm === "pt" ? p.n_equil + p.n_sweeps : Ts.length * (algorithm === "metropolis" ? p.n_restarts : 1),
           `Algorithm: ${algorithm}`);

    let results;
    if (algorithm === "wolff") results = runWolff(lat, lattice.E0, lattice.gauge, Ts, p, rng, notify);
    else if (algorithm === "metropolis") results = runMetropolis(lat, lattice.E0, Ts, p, rng, notify, warnings);
    else results = runParallelTempering(lat, lattice.E0, Ts, p, rng, notify);

    for (const k of ["T"].concat(KEYS)) results[k] = results[k].map(finite);
    results.frustrated = !!lattice.frustrated;
    results.frustration_cycle = lattice.frustrated ? cycle : null;
    results.algorithm = algorithm;
    results.N = N;
    if (results.T.length >= 3) {
      results.Tc_peak = finite(estimateTc(results.T, results.chi_connected));
      results.frustration_factor =
        results.Tc_peak === null ? null : finite(frustrationFactor(results.Tc_peak, lattice.J, N));
    } else {
      results.Tc_peak = null;
      results.frustration_factor = null;
    }
    return { results, warnings };
  }

  /** Throw an Error if the parameters are not valid (with the defaults). */
  function validateParams(params) {
    checkParams(Object.assign({}, DEFAULTS, params));
  }

  const api = { runIsingMC, validateParams, estimateTc, frustrationFactor, geomspace, makeRng };
  if (root) root.MCEngine = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : this);
