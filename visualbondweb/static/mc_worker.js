/*
 * Web Worker of the Monte Carlo tab: runs mc_engine.js on the lattice
 * built by the server (POST /montecarlo/lattice), off the main thread.
 *
 * Receives {lattice, params}; posts {type: "progress", done, total,
 * message} (at most every 100 ms), then {type: "done", results,
 * warnings} or {type: "error", error}. To cancel, the page terminates
 * the worker.
 */
importScripts("mc_engine.js");

self.onmessage = (event) => {
  const { lattice, params } = event.data;
  let last = 0;
  const progress = (done, total, message) => {
    const now = Date.now();
    if (now - last >= 100 || done === total) {
      last = now;
      self.postMessage({ type: "progress", done, total, message });
    }
  };
  try {
    const { results, warnings } = self.MCEngine.runIsingMC(lattice, params, progress);
    self.postMessage({ type: "done", results, warnings });
  } catch (err) {
    self.postMessage({ type: "error", error: err && err.message ? err.message : String(err) });
  }
};
