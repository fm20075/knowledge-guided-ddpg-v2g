# Knowledge-Guided DDPG for Coordinated V2G Protection: A Real-World Case Study in Hangzhou, China

This repository contains the code for the paper accepted to **IEEE Transactions on Smart Grid**.

The code implements:
- GMM-JCC joint risk probability estimation module
- PPIPAM electric vehicle aggregation strategy
- V2G simulation environment with 11‑dim state and 4‑dim action
- DDPG algorithm for optimal V2G scheduling

## Requirements
- Python 3.9 – 3.10
- Dependencies listed in `requirements.txt`

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate simulation data (based on Hangzhou Xihu District scenario):
   ```bash
   python data/generate_simulation_data.py
   ```

3. Train the DDPG agent and evaluate performance:
   ```bash
   python run/train.py
   ```

All results (metrics, logs, and figures) are saved in the `output/` folder.

## Results

The final test metrics (average joint risk probability, maximum risk, average load rate, total reward) are printed at the end of training and also saved in `output/final_test_metrics.txt`.

## Repository Structure

```plain
.
├── core/               # Core modules
│   ├── config.py       # Global parameters
│   ├── ddpg.py         # DDPG agent implementation
│   ├── ev_aggregation.py  # EV aggregator (PPIPAM)
│   ├── gmm_jcc_risk.py # GMM-JCC risk estimator
│   └── v2g_env.py      # V2G simulation environment
├── data/               # Data generation and dataset
│   ├── generate_simulation_data.py
│   ├── simulation_data.npy
│   └── data_readme.md
├── run/                # Execution scripts
│   ├── train.py
│   ├── plot_figures_final.py
│   ├── run.bat
│   ├── run.sh
│   └── run_plot.bat
├── model/              # Saved models (generated during training; not included in repo)
├── output/             # Results and figures (auto‑generated)
├── requirements.txt
├── LICENSE
└── README.md