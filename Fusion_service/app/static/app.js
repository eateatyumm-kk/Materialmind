(() => {
  "use strict";

  const MAX_FILE_BYTES = 2 * 1024 * 1024;
  const REQUEST_TIMEOUT_MS = 60_000;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("predict-form"), formula: $("formula"), cifInput: $("cif-input"), dropzone: $("dropzone"),
    fileChip: $("file-chip"), fileName: $("file-name"), fileSize: $("file-size"), fileClear: $("file-clear"),
    exampleList: $("example-list"), hint: $("form-hint"), submit: $("submit"), status: $("status"),
    empty: $("empty-state"), loading: $("loading"), error: $("error"), errorTitle: $("error-title"),
    errorDetail: $("error-detail"), errorHint: $("error-hint"), result: $("result"),
    copyJson: $("copy-json"), copyStatus: $("copy-status"),
  };

  const state = { file: null, busy: false, lastResponse: null, referenceRmse: null, examples: [], activeExample: null };

  // ---------- formatting ----------
  const fmtGpa = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
  const fmtLog = (v) => v.toFixed(3);
  const fmtBytes = (n) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(2)} MB`);

  // ---------- panel state ----------
  function showPanel(name) {
    el.empty.hidden = name !== "empty";
    el.loading.hidden = name !== "loading";
    el.error.hidden = name !== "error";
    el.result.hidden = name !== "result";
  }

  function refreshSubmit() {
    el.submit.disabled = state.busy || !el.formula.value.trim() || !state.file;
  }

  function setHint(message, isError = false) {
    el.hint.textContent = message;
    el.hint.style.color = isError ? "var(--danger)" : "";
  }

  const DEFAULT_HINT = "The formula should describe the same material as the CIF file.";

  // ---------- file handling ----------
  function setFile(file, exampleId = null) {
    if (!file) {
      state.file = null;
      state.activeExample = null;
      el.cifInput.value = "";
      el.fileChip.hidden = true;
    } else {
      if (!/\.cif$/i.test(file.name)) return setHint("Please choose a file with a .cif extension.", true);
      if (file.size === 0) return setHint("That file is empty.", true);
      if (file.size > MAX_FILE_BYTES) return setHint(`That file is ${fmtBytes(file.size)}; the limit is 2 MB.`, true);
      state.file = file;
      state.activeExample = exampleId;
      el.fileName.textContent = file.name;
      el.fileSize.textContent = fmtBytes(file.size);
      el.fileChip.hidden = false;
    }
    setHint(DEFAULT_HINT);
    syncExampleChips();
    refreshSubmit();
  }

  el.cifInput.addEventListener("change", () => setFile(el.cifInput.files[0] || null));
  el.fileClear.addEventListener("click", () => setFile(null));
  el.formula.addEventListener("input", () => {
    // Editing the formula after picking an example means the user is now customising it.
    if (state.activeExample) { state.activeExample = null; syncExampleChips(); }
    refreshSubmit();
  });

  ["dragenter", "dragover"].forEach((type) =>
    el.dropzone.addEventListener(type, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((type) =>
    el.dropzone.addEventListener(type, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); }));
  el.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0] || null));
  // A file dropped slightly outside the zone would otherwise navigate the tab to the file.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // ---------- examples ----------
  function syncExampleChips() {
    el.exampleList.querySelectorAll(".chip").forEach((chip) =>
      chip.setAttribute("aria-pressed", String(chip.dataset.id === state.activeExample)));
  }

  function useExample(example) {
    el.formula.value = example.formula;
    setFile(new File([example.cif], example.filename, { type: "chemical/x-cif" }), example.id);
    el.submit.focus();
  }

  async function loadExamples() {
    try {
      const res = await fetch("/static/examples.json");
      if (!res.ok) throw new Error(String(res.status));
      state.examples = await res.json();
    } catch {
      el.exampleList.textContent = "Examples unavailable.";
      return;
    }
    for (const example of state.examples) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.dataset.id = example.id;
      chip.setAttribute("aria-pressed", "false");
      chip.textContent = example.formula;
      chip.title = example.label;
      chip.addEventListener("click", () => useExample(example));
      el.exampleList.appendChild(chip);
    }
  }

  // ---------- service status ----------
  async function loadStatus() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error(String(res.status));
      const info = await res.json();
      state.referenceRmse = info.reference_test_rmse_log10;
      el.status.textContent = `${info.ensemble_size} models · ${info.device}`;
      el.status.dataset.state = "ok";
    } catch {
      el.status.textContent = "Service unreachable";
      el.status.dataset.state = "error";
    }
  }

  // ---------- request ----------
  function errorMessage(status, body) {
    if (body && typeof body.detail === "string") return body.detail;
    if (body && Array.isArray(body.detail)) {
      return body.detail.map((d) => `${(d.loc || []).slice(1).join(".") || "input"}: ${d.msg}`).join("\n");
    }
    return `The server returned HTTP ${status}.`;
  }

  function showError(title, detail, hint) {
    el.errorTitle.textContent = title;
    el.errorDetail.textContent = detail;
    el.errorHint.textContent = hint;
    showPanel("error");
  }

  async function predict(event) {
    event.preventDefault();
    if (state.busy || el.submit.disabled) return;

    state.busy = true;
    el.submit.textContent = "Predicting…";
    refreshSubmit();
    showPanel("loading");

    const body = new FormData();
    body.append("formula", el.formula.value.trim());
    body.append("cif_file", state.file, state.file.name);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch("/predict", { method: "POST", body, signal: controller.signal });
      const payload = await res.json().catch(() => null);
      if (!res.ok) {
        const hint = res.status >= 500
          ? "This looks like a problem on the server, not with your input."
          : "Check that the formula and CIF file are valid and describe the same material.";
        return showError("Couldn't run the prediction", errorMessage(res.status, payload), hint);
      }
      renderResult(payload);
    } catch (err) {
      const timedOut = err && err.name === "AbortError";
      showError(
        timedOut ? "The request timed out" : "Couldn't reach the service",
        timedOut ? "No response within 60 seconds." : String(err && err.message ? err.message : err),
        "Check that the service is running and try again.",
      );
    } finally {
      clearTimeout(timer);
      state.busy = false;
      el.submit.textContent = "Predict bulk modulus";
      refreshSubmit();
    }
  }
  el.form.addEventListener("submit", predict);

  // ---------- rendering ----------
  function renderResult(data) {
    state.lastResponse = data;
    const members = Object.entries(data.per_seed_log10_bulk_modulus)
      .map(([seed, log10]) => ({ seed, log10 }))
      .sort((a, b) => Number(a.seed) - Number(b.seed));

    $("r-formula").textContent = data.formula;
    $("r-k").textContent = fmtGpa(data.bulk_modulus_gpa_mean);
    $("r-lower").textContent = fmtGpa(data.bulk_modulus_gpa_lower);
    $("r-upper").textContent = fmtGpa(data.bulk_modulus_gpa_upper);
    $("r-log-mean").textContent = fmtLog(data.log10_bulk_modulus_mean);
    $("r-log-std").textContent = `±${fmtLog(data.log10_bulk_modulus_std)}`;
    $("r-size").textContent = String(data.ensemble_size);
    $("r-scope").textContent = data.uncertainty_scope;

    const context = $("r-context");
    if (typeof state.referenceRmse === "number") {
      const factor = 10 ** state.referenceRmse;
      context.textContent =
        `On held-out test materials this model's typical error (RMSE) is about ${fmtLog(state.referenceRmse)} in log₁₀ K, ` +
        `roughly a factor of ${factor.toFixed(2)} (about ${Math.round((factor - 1) * 100)}%) in K. ` +
        "That is usually larger than the ensemble spread shown here, so the range above understates the real error.";
    } else {
      context.textContent = "";
    }

    const tbody = $("r-members");
    tbody.replaceChildren(...members.map((m) => {
      const row = document.createElement("tr");
      [m.seed, fmtLog(m.log10), fmtGpa(10 ** m.log10)].forEach((text) => {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.appendChild(cell);
      });
      return row;
    }));

    el.copyStatus.textContent = "";
    showPanel("result"); // must be visible before drawing so the plot can measure its container
    drawPlot();
    $("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function drawPlot() {
    const data = state.lastResponse;
    if (!data || el.result.hidden) return;
    const members = Object.entries(data.per_seed_log10_bulk_modulus).map(([seed, log10]) => ({ seed, log10 }));
    renderPlot($("r-plot"), members, data.log10_bulk_modulus_mean, data.log10_bulk_modulus_std);
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawPlot, 150);
  });

  function svgEl(name, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderPlot(svg, members, mean, std) {
    // Size the drawing to the container so text stays at a constant readable size on narrow screens.
    const W = Math.max(280, Math.round(svg.parentElement.clientWidth) || 640), H = 150, padX = 28, axisY = 112;
    const tickCount = W < 420 ? 3 : 5;
    const values = members.map((m) => m.log10);
    const lo0 = Math.min(...values, mean - std);
    const hi0 = Math.max(...values, mean + std);
    const pad = Math.max(hi0 - lo0, 0.1) * 0.3;
    const lo = lo0 - pad, hi = hi0 + pad;
    const x = (v) => padX + ((v - lo) / (hi - lo)) * (W - 2 * padX);

    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute(
      "aria-label",
      `Ensemble predictions from ${fmtGpa(10 ** Math.min(...values))} to ${fmtGpa(10 ** Math.max(...values))} GPa, mean ${fmtGpa(10 ** mean)} GPa`,
    );

    const nodes = [];
    const bandLeft = x(mean - std), bandRight = x(mean + std);
    nodes.push(svgEl("rect", { class: "band", x: bandLeft, y: 34, width: Math.max(bandRight - bandLeft, 2), height: 68, rx: 4 }));
    nodes.push(svgEl("line", { class: "axis", x1: padX, x2: W - padX, y1: axisY, y2: axisY }));

    for (let i = 0; i < tickCount; i++) {
      const t = lo + ((hi - lo) * i) / (tickCount - 1);
      nodes.push(svgEl("line", { class: "tick", x1: x(t), x2: x(t), y1: axisY, y2: axisY + 5 }));
      nodes.push(svgEl("text", { class: "tick-label", x: x(t), y: axisY + 22 }, fmtGpa(10 ** t)));
    }

    nodes.push(svgEl("line", { class: "mean", x1: x(mean), x2: x(mean), y1: 30, y2: 106 }));
    nodes.push(svgEl("text", { class: "mean-label", x: x(mean), y: 20 }, `mean ${fmtGpa(10 ** mean)}`));

    // Stagger neighbouring dots vertically so near-identical predictions stay individually visible.
    [...members].sort((a, b) => a.log10 - b.log10).forEach((m, i) => {
      const dot = svgEl("circle", { class: "dot", cx: x(m.log10), cy: 68 + ((i % 3) - 1) * 16, r: 6 });
      dot.appendChild(svgEl("title", {}, `seed ${m.seed}: ${fmtGpa(10 ** m.log10)} GPa (log₁₀ ${fmtLog(m.log10)})`));
      nodes.push(dot);
    });

    svg.replaceChildren(...nodes);
  }

  // ---------- copy JSON ----------
  el.copyJson.addEventListener("click", async () => {
    const text = JSON.stringify(state.lastResponse, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      el.copyStatus.textContent = "Copied to clipboard.";
    } catch {
      el.copyStatus.textContent = "Couldn't access the clipboard in this browser.";
    }
  });

  // ---------- init ----------
  showPanel("empty");
  refreshSubmit();
  loadStatus();
  loadExamples();
})();
