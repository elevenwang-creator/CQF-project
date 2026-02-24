# NAIS-Net Framework for High-Dimensional Pricing

Implemented a **NAIS-Net**(Non-autonomous Input-output Stable) architecture utilizing TensorFlow to solve PDEs by **reformulating** them as **FBSDEs**.Benchmarked scalability against 1D Black-Scholes and **100D Black-Scholes-Barenblatt (BSB)** equations; validated robustness using the **rough Bergomi(rBergomi)** model.

## Computational Optimization:
1. Optimized variance processes in the rBM using **Hybrid simulation schemas**.
2. Accelerated computation by mapping **Toeplitz matrix** multiplication to **Discrete Convolution** via **FFT**.
3. Controlled convergence and stabilized training by incorporating specific penalty terms into the objective function.
