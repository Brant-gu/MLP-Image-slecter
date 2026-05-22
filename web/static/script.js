/* ═══════════════════════════════════════════════════════════
   MLP Dashboard – Frontend Logic  (i18n-aware)
   ═══════════════════════════════════════════════════════════ */

// ── state ────────────────────────────────────────────────────
let currentBatch  = null;
let pollTimer     = null;
let previewCls    = "sphere";
let archMode      = "auto";   // "auto" | "custom"
let evalListsLoaded = false;

const STEP_ORDER = ["generate", "normalize", "prepare", "surrogate", "train_main"];
const HIDDEN_OPTIONS = [32, 64, 128, 256, 512];
const DROPOUT_STEP   = 0.05;

// ── DOM refs ─────────────────────────────────────────────────
const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ── API helpers ──────────────────────────────────────────────
async function api(url, opts = {}) {
    let res;
    try {
        res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
    } catch (err) {
        console.error("API network error:", url, err);
        return { ok: false, error: "Network request failed" };
    }
    if (!res.ok) {
        console.error("API HTTP error:", url, res.status, res.statusText);
        try { return await res.json(); }
        catch (_) { return { ok: false, error: `HTTP ${res.status}` }; }
    }
    try { return await res.json(); }
    catch (err) {
        console.error("API JSON parse error:", url, err);
        return { ok: false, error: "Invalid response" };
    }
}

// ═══════════════════════════════════════════════════════════════
//  BATCH LIST
// ═══════════════════════════════════════════════════════════════

async function refreshBatches(silent) {
    const list = await api("/api/batches");
    const container = $("#batch-list");
    if (!list.length) {
        container.innerHTML = `<div class="empty-hint">${t("sidebar.empty")}</div>`;
        return;
    }
    container.innerHTML = list.map(b => {
        let badge = "";
        if (b.running) {
            badge = `<span class="batch-badge running">${t("batch.running")}</span>`;
        } else if (b.failed) {
            badge = `<span class="batch-badge failed">${t("batch.failed")}</span>`;
        } else if (b.done === b.total) {
            badge = `<span class="batch-badge done">${t("batch.done")}</span>`;
        } else {
            badge = `<span class="batch-badge">${b.done}/${b.total}</span>`;
        }

        const active = currentBatch === b.name ? " active" : "";
        return `<div class="batch-item${active}" data-batch="${b.name}">
                  <span class="batch-name">${b.name}</span>${badge}
                </div>`;
    }).join("");

    container.querySelectorAll(".batch-item").forEach(el => {
        el.addEventListener("click", () => selectBatch(el.dataset.batch));
    });
}

async function selectBatch(name) {
    currentBatch = name;
    $("#current-batch-title").textContent = name;
    $("#batch-meta").textContent = "";

    $("#btn-download-model").disabled = false;
    $("#btn-delete-batch").disabled = false;
    $("#btn-run-all").disabled = false;
    $$(".btn-step").forEach(b => b.disabled = false);
    $("#tab-bar").style.display = "flex";

    await pollStatus(true);
    startPolling();
    switchTab("tab-pipeline");
    refreshBatches(true);
}

async function newBatch() {
    stopPolling();
    const ts = new Date().toISOString().replace(/[-:]/g, "").replace("T", "_").slice(0, 15);
    const name = `batch_${ts}`;
    currentBatch = name;
    $("#current-batch-title").textContent = name + " (" + t("topbar.new_hint") + ")";
    $("#batch-meta").textContent = "";
    $("#btn-download-model").disabled = true;
    $("#btn-delete-batch").disabled = false;
    $("#btn-run-all").disabled = false;
    $$(".btn-step").forEach(b => b.disabled = false);
    $("#tab-bar").style.display = "flex";
    $("#results-container").style.display = "none";
    $("#results-empty").style.display = "block";
    $("#epoch-chart-container").style.display = "none";
    $("#results-table").style.display = "none";

    // reset progress labels
    resetProgress();
    switchTab("tab-pipeline");
    refreshBatches(true);
}

// ═══════════════════════════════════════════════════════════════
//  PIPELINE EXECUTION
// ═══════════════════════════════════════════════════════════════

function collectParams() {
    return {
        n_per_class:       parseInt($("#param-n").value) || 1000,
        img_size:          parseInt($("#param-size").value) || 32,
        seed:              parseInt($("#param-seed").value) || 42,
        supersample:       parseInt($("#param-ss").value) || 3,
        epochs:            parseInt($("#param-epochs").value) || 30,
        surrogate_epochs:  parseInt($("#param-sur-epochs").value) || 10,
        // PGD params
        sur_epsilon:       parseFloat($("#param-sur-eps").value)   || 0.03,
        sur_alpha:         parseFloat($("#param-sur-alpha").value) || 0.01,
        sur_pgd_steps:     parseInt($("#param-sur-steps").value)   || 5,
        main_epsilon:      parseFloat($("#param-main-eps").value)  || 0.06,
        main_alpha:        parseFloat($("#param-main-alpha").value)|| 0.01,
        main_pgd_steps:    parseInt($("#param-main-steps").value)  || 10,
        lambda_adv:        parseFloat($("#param-lambda").value)    || 0.6,
        num_surrogates:    parseInt($("#param-num-sur").value)    || 5,
        arch_mode:         archMode,
        arch_config:       collectArchConfig(),
        noise_props:       collectNoiseProps(),
        shift_magnitude:   parseFloat($("#param-shift-mag").value) || 0,
        shift_prob:        (parseFloat($("#param-shift-prob").value) || 0) / 100,
    };
}

async function runPipeline(step = "all") {
    const params = collectParams();
    const batchName = currentBatch || `batch_${Date.now()}`;

    if (!currentBatch) {
        currentBatch = batchName;
        $("#current-batch-title").textContent = batchName;
        $("#tab-bar").style.display = "flex";
    }

    setButtonsEnabled(false);
    $("#btn-cancel").style.display = "inline-flex";
    $("#btn-run-all").disabled = true;
    resetProgress();

    const resp = await api("/api/run", {
        method: "POST",
        body: JSON.stringify({ batch_name: batchName, step, params }),
    });

    if (!resp.ok) {
        alert(t("launch_failed") + (resp.error || t("error_unknown")));
        setButtonsEnabled(true);
        $("#btn-cancel").style.display = "none";
        return;
    }

    currentBatch = resp.batch_name;
    $("#current-batch-title").textContent = currentBatch;
    startPolling();
    refreshBatches(true);
}

async function cancelRun() {
    if (!currentBatch || !confirm(t("cancel_confirm"))) return;
    await api("/api/cancel", {
        method: "POST",
        body: JSON.stringify({ batch_name: currentBatch }),
    });
    setButtonsEnabled(true);
    $("#btn-cancel").style.display = "none";
    stopPolling();
    await pollStatus(true);
    refreshBatches(true);
}

function setButtonsEnabled(enabled) {
    $$(".btn-step").forEach(b => b.disabled = !enabled);
    if (enabled) $("#btn-run-all").disabled = false;
}

function resetProgress() {
    $("#batch-meta").textContent = "";
    STEP_ORDER.forEach(step => {
        const item = $(`.progress-item[data-step="${step}"]`);
        if (!item) return;
        item.querySelector(".progress-fill").style.width = "0%";
        item.querySelector(".progress-fill").className = "progress-fill";
        const stateEl = item.querySelector(".progress-state");
        stateEl.textContent = t("progress.pending");
        stateEl.className = "progress-state pending";
    });
}

// ═══════════════════════════════════════════════════════════════
//  POLLING
// ═══════════════════════════════════════════════════════════════

function startPolling() {
    stopPolling();
    pollTimer = setInterval(() => pollStatus(false), 1500);
}

function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollStatus(silent) {
    if (!currentBatch) return;
    const s = await api(`/api/batches/${currentBatch}/status`);
    if (!s || !s.steps) return;

    let anyRunning = false;
    let anyFailed  = false;
    let allDone    = true;

    STEP_ORDER.forEach(step => {
        const info = s.steps[step] || { state: "pending", progress: 0, message: "" };
        const item = $(`.progress-item[data-step="${step}"]`);
        if (!item) return;

        const fill    = item.querySelector(".progress-fill");
        const stateEl = item.querySelector(".progress-state");

        fill.style.width = info.progress + "%";
        stateEl.textContent = t("progress." + info.state);

        fill.className = "progress-fill";
        if (info.state === "completed") fill.classList.add("completed");
        if (info.state === "failed")    fill.classList.add("failed");

        stateEl.className = "progress-state " + info.state;

        if (info.state === "running") anyRunning = true;
        if (info.state === "failed")  anyFailed  = true;
        if (info.state !== "completed" && info.state !== "failed") allDone = false;
    });

    // batch meta
    const done = STEP_ORDER.filter(st => s.steps[st] && s.steps[st].state === "completed").length;
    $("#batch-meta").textContent = `${done}/5 ${t("steps_complete")}`;

    if (!s.running && !anyRunning) {
        stopPolling();
        setButtonsEnabled(true);
        $("#btn-cancel").style.display = "none";
        $("#btn-run-all").disabled = false;

        if (allDone) {
            $("#btn-download-model").disabled = false;
            loadResults();
            loadPreviews();
        }
    }

    if (!silent) refreshBatches(true);
}

// ═══════════════════════════════════════════════════════════════
//  RESULTS
// ═══════════════════════════════════════════════════════════════

async function loadResults() {
    if (!currentBatch) return;
    const r = await api(`/api/batches/${currentBatch}/results`);

    const hasResults = r.accuracy !== null && r.accuracy !== undefined;

    if (hasResults) {
        $("#results-container").style.display = "";
        $("#results-empty").style.display = "none";
        $("#metric-accuracy").textContent = (r.accuracy * 100).toFixed(2) + "%";

        if (r.classification_report) {
            const lines = r.classification_report.trim().split("\n");
            if (lines.length > 1) {
                const tbody = $("#results-table tbody");
                tbody.innerHTML = "";
                const clsMap = { 0: t("class.sphere"), 1: t("class.cube"), 2: t("class.tetrahedron") };
                for (let i = 1; i < lines.length; i++) {
                    const cols = lines[i].trim().split(/\s+/);
                    if (cols.length >= 5) {
                        const cls = clsMap[cols[0]] || cols[0];
                        tbody.innerHTML += `<tr>
                            <td>${cols[0]} (${cls})</td>
                            <td>${cols[1]}</td><td>${cols[2]}</td>
                            <td>${cols[3]}</td><td>${cols[4]}</td>
                        </tr>`;
                    }
                }
                $("#results-table").style.display = "";
            }
        }

        if (r.epochs && r.epochs.length > 0) {
            $("#epoch-chart-container").style.display = "";
            drawEpochChart(r.epochs);
        }
    } else {
        $("#results-container").style.display = "none";
        $("#results-empty").style.display = "";
    }
}

function drawEpochChart(data) {
    const canvas = $("#epoch-chart");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 30, right: 30, bottom: 50, left: 60 };
    const pw = W - pad.left - pad.right;
    const ph = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const xs = data.map(d => d.epoch);
    const cleanVals = data.map(d => d.clean_acc);
    const advVals = data.map(d => d.adv_acc);
    const allVals = [...cleanVals, ...advVals];
    const yMin = Math.floor(Math.min(...allVals) * 100) / 100 - 0.02;
    const yMax = Math.ceil(Math.max(...allVals) * 100) / 100 + 0.02;

    function xPos(e) { return pad.left + ((e - xs[0]) / (xs[xs.length - 1] - xs[0])) * pw; }
    function yPos(v) { return pad.top + ph - ((v - yMin) / (yMax - yMin)) * ph; }

    // grid
    ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
        const y = pad.top + (ph / ySteps) * i;
        const val = yMax - ((yMax - yMin) / ySteps) * i;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif";
        ctx.textAlign = "right"; ctx.fillText(val.toFixed(3), pad.left - 8, y + 4);
    }
    ctx.textAlign = "center";
    for (let i = 0; i < xs.length; i += Math.ceil(xs.length / 8)) {
        const x = xPos(xs[i]);
        ctx.fillText(xs[i], x, H - pad.bottom + 16);
    }

    // lines
    function drawLine(vals, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
        ctx.moveTo(xPos(xs[0]), yPos(vals[0]));
        for (let i = 1; i < vals.length; i++) ctx.lineTo(xPos(xs[i]), yPos(vals[i]));
        ctx.stroke();
        for (let i = 0; i < vals.length; i++) {
            ctx.fillStyle = color; ctx.beginPath();
            ctx.arc(xPos(xs[i]), yPos(vals[i]), 2.5, 0, Math.PI * 2); ctx.fill();
        }
    }
    drawLine(cleanVals, "#3a5afc");
    drawLine(advVals, "#ef4444");

    // legend (i18n)
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#3a5afc";
    ctx.fillText(t("results.clean_acc"), W - pad.right - 150, pad.top - 10);
    ctx.fillStyle = "#ef4444";
    ctx.fillText(t("results.adv_acc"), W - pad.right - 50, pad.top - 10);

    // axes
    ctx.strokeStyle = "#1a1a2e"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, H - pad.bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, H - pad.bottom); ctx.lineTo(W - pad.right, H - pad.bottom); ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════
//  IMAGE PREVIEW
// ═══════════════════════════════════════════════════════════════

async function loadPreviews(cls) {
    if (cls) previewCls = cls;
    if (!currentBatch) return;
    const data = await api(`/api/batches/${currentBatch}/preview/${previewCls}`);
    const grid = $("#preview-grid");
    if (data.error || !data.images || !data.images.length) {
        grid.innerHTML = `<div class="empty-hint">${t("preview.empty")}</div>`;
        return;
    }

    grid.innerHTML = "";
    const size = 32;
    const display = 4;
    data.images.forEach(pixels => {
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        canvas.style.width = (size * display) + "px";
        canvas.style.height = (size * display) + "px";
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(size, size);
        for (let i = 0; i < pixels.length; i++) {
            const v = pixels[i];
            imgData.data[i * 4]     = v;
            imgData.data[i * 4 + 1] = v;
            imgData.data[i * 4 + 2] = v;
            imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        grid.appendChild(canvas);
    });
}

// ═══════════════════════════════════════════════════════════════
//  LOG VIEWER
// ═══════════════════════════════════════════════════════════════

async function loadLog(step) {
    if (!currentBatch) return;
    const data = await api(`/api/batches/${currentBatch}/log/${step}`);
    $("#log-viewer").textContent = data.log || t("logs.empty");
}

function updateLogSelector() {
    const opts = {
        generate:   t("logs.step.generate"),
        normalize:  t("logs.step.normalize"),
        prepare:    t("logs.step.prepare"),
        surrogate:  t("logs.step.surrogate"),
        train_main: t("logs.step.train_main"),
    };
    const sel = $("#log-step-selector");
    Array.from(sel.options).forEach(opt => {
        if (opts[opt.value]) opt.textContent = opts[opt.value];
    });
}

// ═══════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ═══════════════════════════════════════════════════════════════

function switchTab(tabId) {
    $$(".tab-content").forEach(el => el.classList.remove("active"));
    $$(".tab").forEach(el => el.classList.remove("active"));
    const tabContent = document.getElementById(tabId);
    if (tabContent) tabContent.classList.add("active");
    const tabBtn = document.querySelector(`.tab[data-tab="${tabId}"]`);
    if (tabBtn) tabBtn.classList.add("active");

    if (tabId === "tab-preview") loadPreviews();
    if (tabId === "tab-results") loadResults();
    if (tabId === "tab-logs")    loadLog($("#log-step-selector").value);
    if (tabId === "tab-eval") {
        if (!evalListsLoaded) {
            loadModelList();
            loadDatasetList();
            evalListsLoaded = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  ARCHITECTURE EDITOR
// ═══════════════════════════════════════════════════════════════

function countParams(h1, h2) {
    // 1024*h1 + h1 + h1*h2 + h2 + h2*3 + 3
    return 1024 * h1 + h1 + h1 * h2 + h2 + h2 * 3 + 3;
}

function buildArchEditor() {
    const n = parseInt($("#param-num-sur").value) || 5;
    const list = $("#arch-editor-list");
    list.innerHTML = "";

    for (let i = 0; i < n; i++) {
        const row = document.createElement("div");
        row.className = "arch-row";
        row.dataset.index = i;

        // index
        const idx = document.createElement("span");
        idx.className = "arch-row-index";
        idx.textContent = i + 1;

        // hidden 1
        const sel1 = document.createElement("select");
        HIDDEN_OPTIONS.forEach(v => {
            const o = document.createElement("option");
            o.value = v;
            o.textContent = v;
            sel1.appendChild(o);
        });
        sel1.value = (i % 2 === 0) ? 128 : 256;

        // hidden 2
        const sel2 = document.createElement("select");
        HIDDEN_OPTIONS.filter(v => v <= 256).forEach(v => {
            const o = document.createElement("option");
            o.value = v;
            o.textContent = v;
            sel2.appendChild(o);
        });
        sel2.value = 64;

        // dropout
        const dp = document.createElement("input");
        dp.type = "number";
        dp.min = "0"; dp.max = "0.5"; dp.step = String(DROPOUT_STEP);
        dp.value = (0.05 * (i % 5)).toFixed(2);
        dp.style.width = "70px";

        // param count
        const params = document.createElement("span");
        params.className = "arch-row-params";
        params.textContent = (countParams(parseInt(sel1.value), parseInt(sel2.value)) / 1000).toFixed(0) + "K";

        function updateParams() {
            params.textContent = (countParams(parseInt(sel1.value), parseInt(sel2.value)) / 1000).toFixed(0) + "K";
        }
        sel1.addEventListener("change", updateParams);
        sel2.addEventListener("change", updateParams);

        // delete button
        const del = document.createElement("button");
        del.className = "arch-row-del";
        del.innerHTML = "×";
        del.title = t("arch.row_remove");
        del.addEventListener("click", () => {
            row.remove();
            reindexArchRows();
            // update num_surrogates to match
            const count = $$("#arch-editor-list .arch-row").length;
            if (count > 0) {
                $("#param-num-sur").value = count;
            }
        });

        row.appendChild(idx);
        row.appendChild(sel1);
        row.appendChild(sel2);
        row.appendChild(dp);
        row.appendChild(params);
        row.appendChild(del);
        list.appendChild(row);
    }
}

function reindexArchRows() {
    $$("#arch-editor-list .arch-row").forEach((row, i) => {
        row.dataset.index = i;
        row.querySelector(".arch-row-index").textContent = i + 1;
    });
}

function collectArchConfig() {
    if (archMode !== "custom") return null;
    const rows = $$("#arch-editor-list .arch-row");
    if (rows.length === 0) return null;
    const configs = [];
    rows.forEach(row => {
        const sel1 = row.querySelectorAll("select")[0];
        const sel2 = row.querySelectorAll("select")[1];
        const dp   = row.querySelector("input[type='number']");
        configs.push({
            hidden: [parseInt(sel1.value), parseInt(sel2.value)],
            dropout: parseFloat(dp.value),
        });
    });
    return JSON.stringify(configs);
}

function switchArchMode(mode) {
    archMode = mode;
    const autoBtn   = $("#arch-mode-auto");
    const customBtn = $("#arch-mode-custom");
    const editor    = $("#arch-custom-editor");
    const hintAuto  = $("#arch-hint-auto");
    const hintCustom = $("#arch-hint-custom");

    if (mode === "auto") {
        autoBtn.classList.add("active");
        customBtn.classList.remove("active");
        editor.style.display = "none";
        hintAuto.style.display = "";
        hintCustom.style.display = "none";
    } else {
        customBtn.classList.add("active");
        autoBtn.classList.remove("active");
        editor.style.display = "";
        hintCustom.style.display = "";
        hintAuto.style.display = "none";
        buildArchEditor();
    }
}

// ═══════════════════════════════════════════════════════════════
//  NOISE CONFIGURATION
// ═══════════════════════════════════════════════════════════════

function buildNoiseGrid() {
    const grid = $("#noise-grid");
    grid.innerHTML = "";
    for (let level = 0; level <= 10; level++) {
        const row = document.createElement("div");
        row.className = "noise-row";

        const label = document.createElement("span");
        label.className = "noise-label";
        label.textContent = level * 10 + "%";

        const input = document.createElement("input");
        input.type = "number";
        input.className = "noise-input";
        input.dataset.level = level;
        input.value = level === 0 ? 100 : 0;
        input.min = 0;
        input.max = 100;
        input.step = 1;
        input.addEventListener("input", updateNoiseTotal);

        const pct = document.createElement("span");
        pct.className = "noise-pct";
        pct.textContent = "%";

        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(pct);
        grid.appendChild(row);
    }
}

function updateNoiseTotal() {
    let total = 0;
    $$("#noise-grid .noise-input").forEach(inp => {
        total += parseInt(inp.value) || 0;
    });
    $("#noise-total-val").textContent = total + "%";
    const hint = $("#noise-total-hint");
    if (total === 100) {
        hint.textContent = "";
        hint.style.color = "";
    } else if (total < 100) {
        hint.textContent = " (" + (100 - total) + "% 未分配)";
        hint.style.color = "#f59e0b";
    } else {
        hint.textContent = " (超出 " + (total - 100) + "%)";
        hint.style.color = "#ef4444";
    }
}

function collectNoiseProps() {
    const vals = [];
    $$("#noise-grid .noise-input").forEach(inp => {
        vals.push((parseInt(inp.value) || 0) / 100);  // convert % to fraction
    });
    return vals;
}

function applyNoisePreset(preset) {
    const inputs = $$("#noise-grid .noise-input");
    if (preset === "clean") {
        inputs.forEach((inp, i) => { inp.value = i === 0 ? 100 : 0; });
    } else if (preset === "uniform") {
        const v = Math.floor(100 / 11);
        const rem = 100 - v * 11;
        inputs.forEach((inp, i) => { inp.value = v + (i < rem ? 1 : 0); });
    }
    updateNoiseTotal();
}

// ═══════════════════════════════════════════════════════════════
//  EVENT BINDINGS
// ═══════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    refreshBatches(true);
    updateLogSelector();

    // language toggle
    $("#lang-toggle").addEventListener("click", () => {
        setLang(LANG === "zh" ? "en" : "zh");
        updateLogSelector();
    });

    $("#btn-new-batch").addEventListener("click", newBatch);
    $("#btn-refresh-batches").addEventListener("click", () => refreshBatches(false));

    $("#btn-run-all").addEventListener("click", () => runPipeline("all"));

    $$(".btn-step[data-step]").forEach(btn => {
        btn.addEventListener("click", () => {
            const step = btn.dataset.step;
            if (step) runPipeline(step);
        });
    });

    $("#btn-cancel").addEventListener("click", cancelRun);

    $("#btn-download-model").addEventListener("click", () => {
        if (currentBatch) window.open(`/api/download/${currentBatch}/1.npz`);
    });

    $("#btn-delete-batch").addEventListener("click", async () => {
        if (!currentBatch) return;
        if (!confirm(t("delete_confirm"))) return;
        const resp = await api("/api/delete-batch", {
            method: "POST",
            body: JSON.stringify({ batch_name: currentBatch }),
        });
        if (resp.ok) {
            currentBatch = null;
            $("#current-batch-title").textContent = t("topbar.placeholder");
            $("#batch-meta").textContent = "";
            $("#btn-download-model").disabled = true;
            $("#btn-delete-batch").disabled = true;
            $("#btn-run-all").disabled = true;
            $$(".btn-step").forEach(b => b.disabled = true);
            $("#tab-bar").style.display = "none";
            resetProgress();
            $("#results-empty").style.display = "block";
            $("#results-table").style.display = "none";
            $("#epoch-chart-container").style.display = "none";
            $("#preview-grid").innerHTML = `<div class="empty-hint">${t("preview.empty")}</div>`;
            refreshBatches(true);
        } else {
            alert(t("delete_failed") + (resp.error || t("error_unknown")));
        }
    });

    // tabs
    $$(".tab").forEach(tab => {
        tab.addEventListener("click", () => switchTab(tab.dataset.tab));
    });

    // collapsible cards
    $$(".card-header.collapsible").forEach(header => {
        header.addEventListener("click", () => {
            const body = document.getElementById(header.dataset.target);
            if (body) {
                body.classList.toggle("collapsed");
                header.classList.toggle("collapsed");
            }
        });
    });

    // log step selector
    $("#log-step-selector").addEventListener("change", (e) => {
        loadLog(e.target.value);
    });

    // architecture mode toggle
    $("#arch-mode-auto").addEventListener("click", () => switchArchMode("auto"));
    $("#arch-mode-custom").addEventListener("click", () => switchArchMode("custom"));

    // rebuild arch editor when num_surrogates changes
    $("#param-num-sur").addEventListener("change", () => {
        if (archMode === "custom") buildArchEditor();
    });
    $("#param-num-sur").addEventListener("input", () => {
        if (archMode === "custom") buildArchEditor();
    });

    // noise presets
    $$(".noise-preset-bar .btn-step").forEach(btn => {
        btn.addEventListener("click", () => {
            $$(".noise-preset-bar .btn-step").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            applyNoisePreset(btn.dataset.preset);
        });
    });

    // build noise grid on load
    buildNoiseGrid();
    updateNoiseTotal();

    // preview class buttons
    $$(".preview-controls .btn-step").forEach(btn => {
        btn.addEventListener("click", () => {
            $$(".preview-controls .btn-step").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            previewCls = btn.dataset.cls;
            loadPreviews(previewCls);
        });
    });

    // Ctrl+Enter shortcut
    document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            e.preventDefault();
            runPipeline("all");
        }
    });

    $("#btn-run-eval").addEventListener("click", runEvaluation);

    $("#eval-model-select").addEventListener("change", checkEvalReady);
    $("#eval-data-select").addEventListener("change", checkEvalReady);

    // periodic server health check
    checkServerStatus();
    setInterval(checkServerStatus, 30000);
});

// ═══════════════════════════════════════════════════════════════
//  ADVERSARIAL EVALUATION MODULE
// ═══════════════════════════════════════════════════════════════

async function loadModelList() {
    const models = await api("/api/models");
    const sel = $("#eval-model-select");
    sel.innerHTML = `<option value="">${t("eval.model_placeholder")}</option>`;
    models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.path;
        opt.textContent = `${m.batch_name}/${m.filename} (${m.size_kb} KB)`;
        opt.dataset.batch = m.batch_name;
        opt.dataset.filename = m.filename;
        sel.appendChild(opt);
    });
    if (!models.length) {
        sel.innerHTML += `<option disabled>-- ${t("sidebar.empty")} --</option>`;
    }
}

async function loadDatasetList() {
    const batches = await api("/api/batches");
    const sel = $("#eval-data-select");
    sel.innerHTML = `<option value="">${t("eval.data_placeholder")}</option>`;
    batches.forEach(b => {
        if (b.done >= 2) {  // at least normalized data available
            const opt = document.createElement("option");
            opt.value = b.name;
            opt.textContent = `${b.name} (${b.done}/${b.total} ${t("steps_complete")})`;
            sel.appendChild(opt);
        }
    });
}

function checkEvalReady() {
    const model = $("#eval-model-select").value;
    const ds = $("#eval-data-select").value;
    $("#btn-run-eval").disabled = !model || !ds;
}

async function runEvaluation() {
    const modelPath = $("#eval-model-select").value;
    const dataBatch = $("#eval-data-select").value;
    if (!modelPath || !dataBatch) return;

    const btn = $("#btn-run-eval");
    btn.disabled = true;
    btn.textContent = "⏳ " + t("eval.running");

    const dataDir = `batches/${dataBatch}`;
    const resp = await api("/api/evaluate", {
        method: "POST",
        body: JSON.stringify({ model_path: modelPath, data_dir: dataDir }),
    });

    btn.disabled = false;
    btn.textContent = t("eval.run");
    checkEvalReady();

    if (!resp.ok) {
        alert(t("eval.error") + (resp.error || t("error_unknown")));
        return;
    }

    renderEvalResults(resp);
}

function renderEvalResults(data) {
    $("#eval-results").style.display = "";

    // overview
    $("#eval-overall-acc").textContent = (data.overall_accuracy * 100).toFixed(2) + "%";
    $("#eval-avg-conf").textContent = (data.avg_confidence * 100).toFixed(1) + "%";
    $("#eval-total-samples").textContent = data.total_samples;
    $("#eval-model-info").textContent = t("eval.model_info")
        .replace("{model}", data.model)
        .replace("{data}", data.data_dir);

    // per-class table
    const classNames = { 0: t("class.sphere"), 1: t("class.cube"), 2: t("class.tetrahedron") };
    const pcTbody = $("#eval-per-class-table tbody");
    pcTbody.innerHTML = "";
    data.per_class.forEach(c => {
        const pct = (c.accuracy * 100).toFixed(1);
        pcTbody.innerHTML += `<tr>
            <td>${c.class} (${classNames[c.class] || c.name})</td>
            <td><strong>${pct}%</strong></td>
            <td>${c.count}</td>
            <td><div class="eval-bar-wrap"><div class="eval-bar" style="width:${pct}%"></div></div></td>
        </tr>`;
    });

    // deciles table
    const dcTbody = $("#eval-deciles-table tbody");
    dcTbody.innerHTML = "";
    data.deciles.forEach(d => {
        const pct = (d.accuracy * 100).toFixed(1);
        const confPct = (d.avg_confidence * 100).toFixed(1);
        dcTbody.innerHTML += `<tr>
            <td>${d.group}/10</td>
            <td>${d.min_confidence.toFixed(2)} – ${d.max_confidence.toFixed(2)}</td>
            <td><strong>${pct}%</strong></td>
            <td>${confPct}%</td>
            <td>${d.count}</td>
            <td><div class="eval-bar-wrap"><div class="eval-bar" style="width:${pct}%"></div></div></td>
        </tr>`;
    });

    // chart
    drawEvalChart(data.deciles);
}

function drawEvalChart(deciles) {
    const canvas = $("#eval-decile-chart");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 25, right: 30, bottom: 45, left: 55 };
    const pw = W - pad.left - pad.right;
    const ph = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const accVals = deciles.map(d => d.accuracy);
    const confVals = deciles.map(d => d.avg_confidence);
    const allVals = [...accVals, ...confVals];
    const yMin = Math.max(0, Math.floor(Math.min(...allVals) * 100) / 100 - 0.05);
    const yMax = Math.min(1, Math.ceil(Math.max(...allVals) * 100) / 100 + 0.05);

    function xPos(i) { return pad.left + (i / (deciles.length - 1)) * pw; }
    function yPos(v) { return pad.top + ph - ((v - yMin) / (yMax - yMin)) * ph; }

    // grid
    ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
        const y = pad.top + (ph / ySteps) * i;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
        ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        const val = yMax - ((yMax - yMin) / ySteps) * i;
        ctx.fillText((val * 100).toFixed(0) + "%", pad.left - 8, y + 4);
    }

    // x-axis labels
    ctx.textAlign = "center";
    ctx.fillStyle = "#6b7280";
    deciles.forEach((d, i) => {
        ctx.fillText((i + 1), xPos(i), H - pad.bottom + 16);
    });
    ctx.fillText(t("eval.decile_group"), W / 2, H - 4);

    // accuracy line
    function drawLine(vals, color, dash) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.5;
        if (dash) ctx.setLineDash([6, 3]);
        else ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(xPos(0), yPos(vals[0]));
        for (let i = 1; i < vals.length; i++) ctx.lineTo(xPos(i), yPos(vals[i]));
        ctx.stroke();
        ctx.setLineDash([]);
        for (let i = 0; i < vals.length; i++) {
            ctx.fillStyle = color; ctx.beginPath();
            ctx.arc(xPos(i), yPos(vals[i]), 3.5, 0, Math.PI * 2); ctx.fill();
        }
    }
    drawLine(accVals, "#3a5afc", false);
    drawLine(confVals, "#10b981", true);

    // legend
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#3a5afc";
    ctx.fillText(t("eval.decile_acc") + " (%)", W - pad.right - 160, pad.top - 8);
    ctx.fillStyle = "#10b981";
    ctx.fillText(t("eval.avg_conf") + " (%)", W - pad.right - 50, pad.top - 8);

    // axes
    ctx.strokeStyle = "#1a1a2e"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, H - pad.bottom); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.left, H - pad.bottom); ctx.lineTo(W - pad.right, H - pad.bottom); ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════
//  SERVER HEALTH CHECK
// ═══════════════════════════════════════════════════════════════

async function checkServerStatus() {
    const dot = $("#server-status");
    if (!dot) return;
    try {
        const res = await fetch("/api/batches");
        if (res.ok) {
            dot.className = "status-dot ok";
        } else {
            dot.className = "status-dot error";
        }
    } catch (_) {
        dot.className = "status-dot error";
    }
}
