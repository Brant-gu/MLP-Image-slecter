"""
Flask backend for MLP Adversarial Training Dashboard.
Provides REST API for pipeline control, status tracking, and data preview.
"""

import os
import sys
import json
import glob
import time
import shutil
import subprocess
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, Response

# ── config ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCHES_DIR  = os.path.join(PROJECT_ROOT, "batches")
LOCK          = threading.Lock()
active_runs   = {}          # batch_name → {"pid": ..., "step": ..., "thread": ...}

os.makedirs(BATCHES_DIR, exist_ok=True)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static"))

# ── helpers ──────────────────────────────────────────────────

def _status_path(batch_name):
    return os.path.join(BATCHES_DIR, batch_name, "status.json")

def read_status(batch_name):
    path = _status_path(batch_name)
    if not os.path.exists(path):
        return {"steps": {}, "running": False, "batch_name": batch_name}
    with open(path, "r") as f:
        return json.load(f)

def write_status(batch_name, data):
    os.makedirs(os.path.dirname(_status_path(batch_name)), exist_ok=True)
    with open(_status_path(batch_name), "w") as f:
        json.dump(data, f, indent=2)

def log_path(batch_name, step_name):
    return os.path.join(BATCHES_DIR, batch_name, f"{step_name}.log")

def _update_step(batch_name, step, state, progress=0, message=""):
    with LOCK:
        s = read_status(batch_name)
        s["steps"][step] = {
            "state": state,         # pending | running | completed | failed
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        write_status(batch_name, s)

def _run_subprocess(batch_name, step, cmd, env=None):
    """Run a command in a thread, streaming output to a log file."""
    _update_step(batch_name, step, "running", 10, "Starting…")

    lpath = log_path(batch_name, step)
    os.makedirs(os.path.dirname(lpath), exist_ok=True)

    try:
        with open(lpath, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                cwd=PROJECT_ROOT,
                env=env,
                bufsize=1,
            )
            active_runs[batch_name] = {"pid": proc.pid, "step": step}

            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()

            proc.wait()

        if proc.returncode == 0:
            _update_step(batch_name, step, "completed", 100, "Done")
        else:
            _update_step(batch_name, step, "failed", 100, f"Exit code {proc.returncode}")
    except Exception as e:
        _update_step(batch_name, step, "failed", 100, str(e))
    finally:
        active_runs.pop(batch_name, None)


def _launch_step(batch_name, step, cmd, env=None):
    """Start a single step in a background thread. Manages running flag."""
    with LOCK:
        s = read_status(batch_name)
        if s.get("running"):
            return False, "Another step is already running for this batch."
        s["running"] = True
        write_status(batch_name, s)

    def _wrapper():
        _run_subprocess(batch_name, step, cmd, env)
        s = read_status(batch_name)
        s["running"] = False
        write_status(batch_name, s)

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    return True, f"Step '{step}' started."


# ── API: batches ─────────────────────────────────────────────

@app.route("/api/batches")
def api_batches():
    """List all batch directories with summary."""
    items = []
    for name in sorted(os.listdir(BATCHES_DIR), reverse=True):
        d = os.path.join(BATCHES_DIR, name)
        if not os.path.isdir(d):
            continue
        s = read_status(name)
        steps = s.get("steps", {})

        # overall progress
        total_steps = 5
        done = sum(1 for v in steps.values() if v["state"] == "completed")
        failed = any(v["state"] == "failed" for v in steps.values())
        running = s.get("running", False)

        items.append({
            "name": name,
            "done": done,
            "total": total_steps,
            "running": running,
            "failed": failed,
            "steps": steps,
        })
    return jsonify(items)


@app.route("/api/batches/<batch_name>/status")
def api_batch_status(batch_name):
    return jsonify(read_status(batch_name))


@app.route("/api/batches/<batch_name>/log/<step>")
def api_batch_log(batch_name, step):
    lpath = log_path(batch_name, step)
    if not os.path.exists(lpath):
        return jsonify({"log": ""})
    with open(lpath, "r", encoding="utf-8") as f:
        return jsonify({"log": f.read()})


@app.route("/api/batches/<batch_name>/results")
def api_batch_results(batch_name):
    """Parse and return training results from log output."""
    s = read_status(batch_name)
    train_log = log_path(batch_name, "train_main")
    results = {"accuracy": None, "classification_report": "", "epochs": []}

    if not os.path.exists(train_log):
        return jsonify(results)

    with open(train_log, "r", encoding="utf-8") as f:
        text = f.read()

    # parse epoch lines
    for line in text.split("\n"):
        if "loss=" in line and "clean_acc=" in line:
            parts = line.split()
            try:
                epoch_str = parts[1].split("/")[0].strip()
                epoch = int(epoch_str.split()[0] if "Epoch" in line else
                           parts[0].replace("Epoch", "").strip().split("/")[0])
                loss = float([p for p in parts if "loss=" in p][0].split("=")[1])
                clean_acc = float([p for p in parts if "clean_acc=" in p][0].split("=")[1])
                adv_acc = float([p for p in parts if "adv_acc=" in p][0].split("=")[1])
                results["epochs"].append({
                    "epoch": epoch,
                    "loss": round(loss, 4),
                    "clean_acc": round(clean_acc, 4),
                    "adv_acc": round(adv_acc, 4),
                })
            except (ValueError, IndexError):
                pass

    # parse final accuracy
    for line in text.split("\n"):
        if line.startswith("Accuracy:"):
            try:
                results["accuracy"] = float(line.split(":")[1].strip())
            except ValueError:
                pass

    # parse classification report
    report_started = False
    report_lines = []
    for line in text.split("\n"):
        if "precision" in line and "recall" in line and "f1" in line:
            report_started = True
            report_lines.append(line)
            continue
        if report_started and line.strip() and line.strip()[0].isdigit():
            report_lines.append(line)
        elif report_started and not line.strip():
            break
    results["classification_report"] = "\n".join(report_lines)

    return jsonify(results)


# ── API: preview images ──────────────────────────────────────

@app.route("/api/batches/<batch_name>/preview/<cls>")
def api_preview(batch_name, cls):
    """Return first few images from a class txt file as pixel arrays."""
    cls_map = {"sphere": "sphere_mix.txt", "cube": "cube_mix.txt",
               "tetrahedron": "tetrahedron_mix.txt"}
    fname = cls_map.get(cls)
    if not fname:
        return jsonify({"error": "invalid class"}), 400

    path = os.path.join(BATCHES_DIR, batch_name, fname)
    if not os.path.exists(path):
        return jsonify({"error": "file not found"}), 404

    images = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            pixels = [int(float(x)) for x in line.strip().split(",")]
            images.append(pixels)

    return jsonify({"images": images, "count": len(images)})


# ── API: pipeline actions ────────────────────────────────────

def _build_env(batch_name):
    """Minimal env – no extra vars needed; scripts use CLI args."""
    return os.environ.copy()


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run the full pipeline or a single step."""
    data = request.json
    batch_name = data.get("batch_name") or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    step       = data.get("step", "all")     # "all" | "generate" | "normalize" | "prepare" | "surrogate" | "train_main"
    params     = data.get("params", {})

    n_per_class = params.get("n_per_class", 1000)
    seed        = params.get("seed", 42)
    epochs      = params.get("epochs", 30)
    sur_epochs  = params.get("surrogate_epochs", 10)
    img_size    = params.get("img_size", 32)
    supersample = params.get("supersample", 3)
    # PGD params
    sur_eps     = params.get("sur_epsilon", 0.03)
    sur_alpha   = params.get("sur_alpha", 0.01)
    sur_steps   = params.get("sur_pgd_steps", 5)
    main_eps    = params.get("main_epsilon", 0.06)
    main_alpha  = params.get("main_alpha", 0.01)
    main_steps  = params.get("main_pgd_steps", 10)
    lambda_adv     = params.get("lambda_adv", 0.6)
    num_surrogates = params.get("num_surrogates", 5)
    arch_mode      = params.get("arch_mode", "auto")   # "auto" | "custom"
    arch_config    = params.get("arch_config", None)    # JSON string for custom mode
    noise_props      = params.get("noise_props", None)      # list of 11 floats or None/empty
    shift_magnitude  = params.get("shift_magnitude", 0.0)   # 0-255
    shift_prob       = params.get("shift_prob", 0.0)        # 0-1

    batch_dir = os.path.join(BATCHES_DIR, batch_name)
    os.makedirs(batch_dir, exist_ok=True)

    # build generate command
    gen_cmd = [sys.executable, "generate_data.py",
               "-n", str(n_per_class), "--size", str(img_size),
               "--seed", str(seed), "--supersample", str(supersample),
               "--output-dir", batch_dir]
    if noise_props and any(p > 0 for p in noise_props):
        gen_cmd += ["--noise-props", ",".join(str(p) for p in noise_props)]
    if shift_magnitude > 0 and shift_prob > 0:
        gen_cmd += ["--shift-magnitude", str(shift_magnitude),
                    "--shift-prob", str(shift_prob)]

    # initialize status — running flag set later when execution actually starts
    status_template = {
        "running": False,
        "batch_name": batch_name,
        "params": params,
        "steps": {
            "generate":      {"state": "pending", "progress": 0, "message": "", "timestamp": ""},
            "normalize":     {"state": "pending", "progress": 0, "message": "", "timestamp": ""},
            "prepare":       {"state": "pending", "progress": 0, "message": "", "timestamp": ""},
            "surrogate":     {"state": "pending", "progress": 0, "message": "", "timestamp": ""},
            "train_main":    {"state": "pending", "progress": 0, "message": "", "timestamp": ""},
        },
    }
    write_status(batch_name, status_template)

    env = _build_env(batch_name)

    # build surrogate / train_main commands respecting arch_mode
    sur_cmd = [sys.executable, "surrogate.py",
               "--data-dir", batch_dir, "--output-dir", batch_dir,
               "--epochs", str(sur_epochs),
               "--epsilon", str(sur_eps), "--alpha", str(sur_alpha),
               "--pgd-steps", str(sur_steps)]
    main_cmd = [sys.executable, "TRAIN-5.py",
                "--data-dir", batch_dir, "--surrogate-dir", batch_dir,
                "--output", os.path.join(batch_dir, "1.npz"),
                "--epochs", str(epochs),
                "--epsilon", str(main_eps), "--alpha", str(main_alpha),
                "--pgd-steps", str(main_steps), "--lambda-adv", str(lambda_adv)]

    if arch_mode == "custom" and arch_config:
        sur_cmd  += ["--arch-config", arch_config]
        main_cmd += ["--arch-config", arch_config]
    else:
        sur_cmd  += ["--num-surrogates", str(num_surrogates)]
        main_cmd += ["--num-surrogates", str(num_surrogates)]

    if step == "all":
        def run_all():
            steps = [
                ("generate", gen_cmd),
                ("normalize", [
                    sys.executable, "normalize.py",
                    "--input-dir", batch_dir, "--output-dir", batch_dir,
                ]),
                ("prepare", [
                    sys.executable, "prepare_surrogate_data.py",
                    "--input-dir", batch_dir, "--output-dir", batch_dir,
                ]),
                ("surrogate", sur_cmd),
                ("train_main", main_cmd),
            ]
            for step_name, cmd in steps:
                # check if previous step failed
                st = read_status(batch_name)
                prev_failed = any(
                    v["state"] == "failed" for v in st["steps"].values()
                )
                if prev_failed:
                    _update_step(batch_name, step_name, "pending", 0,
                                 "Skipped – previous step failed")
                    continue

                _run_subprocess(batch_name, step_name, cmd, env)

            st = read_status(batch_name)
            st["running"] = False
            write_status(batch_name, st)

        with LOCK:
            s = read_status(batch_name)
            if s.get("running"):
                return jsonify({"ok": False, "error": "Batch already running"}), 409
            s["running"] = True
            write_status(batch_name, s)

        t = threading.Thread(target=run_all, daemon=True)
        t.start()
        active_runs[batch_name] = {"pid": None, "step": "all", "thread": t}
        return jsonify({"ok": True, "batch_name": batch_name, "step": "all"})

    # single step
    step_cmds = {
        "generate":  gen_cmd,
        "normalize": [sys.executable, "normalize.py",
                      "--input-dir", batch_dir, "--output-dir", batch_dir],
        "prepare":   [sys.executable, "prepare_surrogate_data.py",
                      "--input-dir", batch_dir, "--output-dir", batch_dir],
        "surrogate": sur_cmd,
        "train_main": main_cmd,
    }

    cmd = step_cmds.get(step)
    if not cmd:
        return jsonify({"ok": False, "error": f"Unknown step: {step}"}), 400

    ok, msg = _launch_step(batch_name, step, cmd, env)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 409
    return jsonify({"ok": True, "batch_name": batch_name, "step": step})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    """Cancel a running batch."""
    data = request.json
    batch_name = data.get("batch_name")
    if not batch_name:
        return jsonify({"ok": False, "error": "batch_name required"}), 400

    run_info = active_runs.get(batch_name)
    if not run_info:
        return jsonify({"ok": False, "error": "No active run for this batch"}), 404

    pid = run_info.get("pid")
    if pid:
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    active_runs.pop(batch_name, None)
    s = read_status(batch_name)
    s["running"] = False
    # mark current step as failed
    current_step = run_info.get("step")
    if current_step and current_step in s.get("steps", {}):
        s["steps"][current_step]["state"] = "failed"
        s["steps"][current_step]["message"] = "Cancelled by user"
    write_status(batch_name, s)
    return jsonify({"ok": True})


@app.route("/api/download/<batch_name>/<filename>")
def api_download(batch_name, filename):
    """Download a file from a batch folder."""
    path = os.path.join(BATCHES_DIR, batch_name, filename)
    if not os.path.exists(path):
        return jsonify({"error": "file not found"}), 404
    return send_file(os.path.abspath(path), as_attachment=True)


@app.route("/api/delete-batch", methods=["POST"])
def api_delete_batch():
    data = request.json
    batch_name = data.get("batch_name")
    if not batch_name:
        return jsonify({"ok": False, "error": "batch_name required"}), 400

    if batch_name in active_runs:
        return jsonify({"ok": False, "error": "Cannot delete a running batch"}), 409

    batch_dir = os.path.join(BATCHES_DIR, batch_name)
    if os.path.isdir(batch_dir):
        shutil.rmtree(batch_dir)
    return jsonify({"ok": True})


# ── API: evaluation / adversarial module ──────────────────────

@app.route("/api/models")
def api_models():
    """List all .npz model files across all batches."""
    models = []
    for name in sorted(os.listdir(BATCHES_DIR), reverse=True):
        d = os.path.join(BATCHES_DIR, name)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".npz"):
                fpath = os.path.join(d, fname)
                models.append({
                    "batch_name": name,
                    "filename": fname,
                    "path": fpath,
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                })
    return jsonify(models)


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Run model evaluation against a dataset."""
    data = request.json
    model_path = data.get("model_path", "")
    data_dir = data.get("data_dir", "")

    if not model_path or not os.path.exists(model_path):
        return jsonify({"ok": False, "error": "Model file not found"}), 400
    if not data_dir or not os.path.isdir(data_dir):
        return jsonify({"ok": False, "error": "Data directory not found"}), 400

    result_path = os.path.join(os.path.dirname(model_path), "_eval_result.json")
    cmd = [
        sys.executable, "eval_model.py",
        "--model", model_path,
        "--data-dir", data_dir,
        "--output-json", result_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT, timeout=120)
        if proc.returncode != 0:
            return jsonify({"ok": False, "error": proc.stderr or proc.stdout}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Evaluation timed out"}), 500

    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)
        os.remove(result_path)
        return jsonify({"ok": True, **result})
    return jsonify({"ok": False, "error": "No output produced"})


# ── main page ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Batches dir:  {BATCHES_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=True)
