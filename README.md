# 📢 开源声明 / Open Source Declaration

本仓库包含以下学术论文的官方完整源代码：
This repository contains the official and complete source code for the following academic research paper:

* **Paper Title:** Strain-based Monitoring Methodology and Numerical Validation for the Evaluation of Transverse Connection Condition in Precast Multi-Girder Bridges
* **Authors:** Wenhao Zheng, Han Wei, etc.
* **Affiliation:** Research Institute of Highway, Ministry of Transport, Beijing, China

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Academic-Grade](https://img.shields.io/badge/Academic--Grade-Verified-success.svg)]()

This repository hosts the official open-source Python framework for the real-time structural health monitoring (SHM) and early-stage degradation evaluation of hinge joints in multi-girder/precast hollow slab bridges (PHSBs). 

By implementing an edge-computing streaming pipeline, the framework orchestrates **orthogonal signal decoupling** via zero-phase filters and evaluates structural integrity using the **Pearson Correlation Coefficient (PCC) of high-frequency dynamic strain increments**. Compared to traditional total strain metrics, this methodology eliminates low-frequency environmental drifts and quasi-static baselines, expanding the early-stage damage sensitivity margin by over **300%**.

---

## 🌟 Key Innovations & Methodology

1. **Orthogonal Signal Decoupling:** Employs symmetric, zero-phase cascaded Time-Domain Ormsby filters to rigidly isolate high-frequency dynamic increments from dominant low-frequency quasi-static baselines (e.g., thermal drifts, pavement cooling, low-frequency vehicle deflections).
2. **Edge-Triggered Real-Time Finite State Machine (FSM):** Simulates an on-site edge gateway via an $O(1)$ time-complexity circular buffer state machine. It adaptively captures discrete single-vehicle bridge-crossing events under stochastic traffic flows without boundary distortion.
3. **Physical-Prior Evaluation Boundary:** Identifies a profound mechanical transition point at $k = 1 \times 10^6 \text{ N/m}^2$, charting the evolution from structural synergy into hazardous "single-girder bearing behavior".

---

## 📂 Repository Architecture

The repository is structured into two standalone modules corresponding to the theoretical and numerical verification chapters of the companion paper:

```text
├── .gitignore               # Standard cache and graphic exclusion filter
├── LICENSE                  # MIT permissive open-source license
├── README.md                # Project homepage and user manual
├── requirements.txt         # Minimal environmental dependency list
│
├── Theoretical_Analysis/     # Analytical Dual-Beam Forced-Vibration Model
│   ├── response_history.py  # Analytical closed-form solver for dual-beam dynamics
│   ├── PCC_stiffness.py     # Maps indicator evolution against continuous stiffness bounds
│   ├── plot_PCC_velocity.py # Evaluates sensitivity margins under varying traffic velocities
│   └── plot_PCC_weight.py   # Proves normalization independence under scaled axle loads
│
└── Finite_Element_Simulation/ # High-Fidelity Stochastic Multi-Girder Field Emulation
    ├── utils.py             # Feature extraction and single-vehicle peak tracking routines
    ├── ormsby_filter.py     # Base operator for time-domain zero-phase signal filtering
    ├── simulation_strain.py # High-efficiency strain modal superposition solver
    ├── streaming_pipeline.py# Real-time FSM edge engine processing live sensor streams
    ├── global_degradation.py# Scenario A: Global uniform joint deterioration tracks
    ├── local_degradation.py# Scenario B: Localized damage localization & immunity checks
    ├── sudden_damage.py     # Scenario C: Time-domain stitched sudden brittle snapping alerts
    └── noise_robustness.py  # Scenario D: Performance convergence under multi-tier sensor noise
