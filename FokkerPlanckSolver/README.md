# Title: Barrier Call Option Pricing



The current algorithm frameworks was proposed by S.Stoykov in the paper - **Numerical Solution of the Fokker-Planck Equation by Variational Approach: An application to Pricing Barrier Options**.

Applying the variational approach leads to the weak formulation of the Fokker-Planck equation (FPE). By replacing the probability density function (PDF) with an approximation based on a combination of B-spline basis functions in the weak formulation, we obtain a system of ordinary differential equations (ODEs). This system is solved numerically using a fifth-order implicit Runge-Kutta method. Finally, the
numerical solution is applied to price barrier options under the Heston stochastic volatility model.

Paper Source: 
Stoykov, S. (2024). [Numerical Solution of Fokker-Planck Equation by Variational Approach – an Application to Pricing Barrier Options. *Wilmott*, 2024(133).](https://doi.org/10.54946/wilm.12077)
