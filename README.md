# MLP Adversarial Training Dashboard

MLP-based 3-class image classifier (sphere / cube / tetrahedron) with ensemble PGD adversarial training, plus a web dashboard for pipeline control and evaluation.

## Overview

- **Data**: 32×32 grayscale renders of 3D shapes at random rotations with directional lighting
- **Training**: Ensemble PGD adversarial training with multiple surrogate models
- **Attack**: Configurable noise injection (11 levels) and brightness shift attack
- **Evaluation**: Confidence-sorted decile accuracy, per-class metrics
- **Dashboard**: Web UI for pipeline control, parameter tuning, real-time progress, and result visualization

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web dashboard
python web/app.py

# 3. Open browser → http://127.0.0.1:5000
```

Use the dashboard to configure parameters and run training, or use the CLI pipeline directly:

```bash
# CLI: run full pipeline (1000 images/class, 30 epochs)
python run_pipeline.py

# Custom settings
python run_pipeline.py -n 2000 --epochs 50 --surrogate-epochs 15
```

## Project Structure

```
├── web/
│   ├── app.py                    # Flask backend (REST API)
│   ├── templates/index.html      # Dashboard HTML
│   └── static/
│       ├── script.js             # Frontend logic
│       ├── style.css             # Styles
│       └── i18n.js               # Chinese/English translations
├── generate_data.py              # Step 1: Render 3D shapes with noise + brightness shift
├── normalize.py                  # Step 2: Normalize pixels 0-255 → 0-1
├── prepare_surrogate_data.py     # Step 3: Stack into train_X.npy / train_y.npy
├── surrogate.py                  # Step 4: Train surrogate models + generate adv examples
├── TRAIN-5.py                    # Step 5: Main adversarial training (ensemble PGD)
├── eval_model.py                 # Model evaluation with decile confidence analysis
├── run_pipeline.py               # One-click pipeline runner (steps 1-5)
├── predict_test.py               # Prediction utility
├── transfer.py                   # adv_X.npy / adv_y.npy → txt conversion
└── requirements.txt
```

## Pipeline Steps

1. **Generate Images** — Renders 3D shapes (sphere, cube, tetrahedron) with random rotations and directional lighting. Supports configurable noise levels (0%-100%) and brightness shift attack.
2. **Normalize** — Scales pixel values from [0, 255] to [0, 1].
3. **Prepare Data** — Stacks normalized txt files into `.npy` arrays with labels (0=sphere, 1=cube, 2=tetrahedron).
4. **Train Surrogates** — Trains 5 diverse MLP surrogate models (varying hidden layers and dropout) with ensemble PGD.
5. **Main Training** — Adversarial training against ensemble PGD attacks from the surrogate models.

## Dashboard Features

- **Pipeline tab**: Configure all parameters (image count, architecture, PGD settings, noise/shift attack) and run the pipeline step-by-step or one-click
- **Results tab**: Overall accuracy, per-class precision/recall/F1, training curves
- **Preview tab**: Visual preview of generated images before/after training
- **Logs tab**: Per-step log output in real time
- **Eval tab**: Adversarial evaluation of trained models against any dataset, with confidence decile analysis

## Language

Dashboard supports **Chinese** and **English**. Toggle via the language button in the sidebar footer. Language preference is persisted in localStorage.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Flask 3.0+
- NumPy
