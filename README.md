# KFEstimate.jl
KFEstimate is a [Julia](https://julialang.org/) package for parameter estimation in linear and non linear state space models, using [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). This is achieved by using stochastic gradient descent on a standard energy log-likelihood to compute a maximum a-priori (MAP) estimate of the model parameters

It is developped by Jean-Guillaume Brasier at [Inria Paris](https://www.inria.fr/en/centre-inria-de-paris) in the [DYOGENE](https://www.di.ens.fr/dyogene/) team.

### Parametrised State Space Models (PSSM)

If you are unfamiliar with Kalman Filtering see: [Kalman Filters](https://en.wikipedia.org/wiki/Kalman_filter)

Let us consider a simple parametrised state space model:

- $x_{t+1} = A(θ)*x_t + w_t$
- $y_t = H(θ)*x_t + r_t$

where:
- $Cov(w_t)↝𝒩(0, Q(θ))$, and $Cov(r_t)↝𝒩(0, R(θ))$
- $A(θ)$ and $H(θ)$ are the process and measurement matrices respectively.


### Parameter Estimation
