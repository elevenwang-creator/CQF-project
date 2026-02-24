# Numerical Methods & Deep-Learning & Quantum Finance

This repository contains my final projects completed during the Certificate in Quantitative Finance (CQF) program. The work spans classical numerical methods for derivative pricing and exploratory research at the intersection of quantum computing and finance.

## 📌 Project Highlights

### 1. Fokker-Planck Equation Solver (Apr 2025 – Oct 2025)
A numerical solver for the Heston model's Fokker-Planck equation using variational methods and B-splines, applied to barrier option pricing.
- [Code](FokkerPlanckSolver/) | [Documentation](FokkerPlanckSolver/README.md)

### 2. NAIS-Net for High-Dimensional FBSDEs (Jan 2024 – Nov 2024)
A TensorFlow implementation of NAIS-Net (Non-Autonomous Input-Stable Network) for solving 100-dimensional FBSDEs, with applications to rough Bergomi and Black-Scholes-Barenblatt equations.
- [Code](NAISNet_FBSDE/) | [Documentation](NAISNet_FBSDE/README.md)

### 3. Ongoing Research: Quantum Neural Networks for Non-Autonomous Architectures (Nov 2025 – Present)
Analyzing the fundamental encoding conflict between NAIS-Net's "non-autonomous" structure (extra input at each layer) and standard QNN architectures (QAOA-based). Two circuit-level solutions are proposed and compared in terms of resource trade-offs.
- [Research Notes](NAISNet_FBSDE/QuantumAcceleration/quantum_exploration.md) (in progress)

## 🔧 Requirements
- Python 3.8+
- TensorFlow 2.x / PyTorch
- NumPy, SciPy, Matplotlib, Qiskit (for quantum exploration)

## 📄 Final Reports
- [FPE Solver Report](FokkerPlanckSolver/'Barrier Call Option Pricing.pdf')
- [NAIS-Net Framework Report](NAISNet_FBSDE/'NAIS-Net Framework for Solving High Dimensional FBSDEs in the Pricing of European Options.pdf')

## 📬 Contact
For questions or collaboration, feel free to open an issue or reach out via [GitHub](https://github.com/elevenwang-creator/CQF-project).
