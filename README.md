# JointPosteriors

[![Build Status](https://travis-ci.org/chriselrod/JointPosteriors.jl.svg?branch=master)](https://travis-ci.org/chriselrod/JointPosteriors.jl)

[![Coverage Status](https://coveralls.io/repos/chriselrod/JointPosteriors.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chriselrod/JointPosteriors.jl?branch=master)

[![codecov.io](http://codecov.io/github/chriselrod/JointPosteriors.jl/coverage.svg?branch=master)](http://codecov.io/github/chriselrod/JointPosteriors.jl?branch=master)

## Introduction

This package implements sparse grid quadrature. I'm working on documentation, organization, tests, and adding more features. And then on getting this package registered. Until then, you can still install via
```
julia> Pkg.clone("https://github.com/chriselrod/SparseQuadratureGrids.jl")
julia> Pkg.clone("https://github.com/chriselrod/LogDensities.jl")
julia> Pkg.clone("https://github.com/chriselrod/JointPosteriors.jl")
```

#### Example 1: Binary Classification

As an example, consider the model from:
> Beavers, D. P., Stamey, J. D., & Bekele, B. N. (2011). A Bayesian model to assess a binary measurement system when no gold standard system is available. Journal of Quality Technology, 43(1), 16.

In the model, there was some number (N) of parts that were defective with probability 1-tau. We have a test that is imperfect, with a false-effective rate of theta+, and a false-defective rate of theta-. For example, if a part is actually defective, the test will rate it as effective with probability theta+. We can test each part (n) times, and we treat each trial as independent.

So, our parameters are a total of three different probabilities. We can specify this via building a parameter struct, and specifying that it has a probability vector of length 3. We'll let the first index denote tau, the second theta_minus, and the last theta_plus.
```
julia> using JointPosteriors

julia> struct BinaryClassification{T} <: parameters{T}
         x::Vector{T}
         p::ProbabilityVector{3,T}
       end
```
Currently, when defining parameter structures, the first field must always be `x::Vector{T}`. This may change eventually (depending in part by how much people hate that).

That is all we need to initialize our model:
```
julia> bc_model = Model(BinaryClassification);
```

But before we really do anything with it, we need data nad a log likelihood function. We have a lot of freedom to write these parts however we'd like.
```
julia> struct BinaryClassificationData <: Data
         X::Array{Int64,1}
         freq::Array{Int64,1}
         NmX::Array{Int64, 1}
         α_m_m1::Float64
         β_m_m1::Float64
         α_p_m1::Float64
         β_p_m1::Float64
         α_τ_m1::Float64
         β_τ_m1::Float64
       end
```
In this case, `X` is how many of the (n) trials were succesful, and `freq` is the number of parts with the corresponding `X` successes. That is, if `X[3] = 2` and `freq[3] =  5`, we had five parts were the test came up positive on two of the (n) trials. `NmX` is defined conveniently to mean the number of trial failures; that is, `NmX[3] = 7` would mean those 5 failed 7 times, and thus that (n) must equal 9.

The six `Float64`s at the end are parameters for beta priors minus one. I subtract one off of each because of beta pdf.
Now, all we must do is define the log density function of our model. Putting our priors together, and summing out the unknown true status gives us the following log density:

```
julia> function log_density(Θ::BinaryClassification, data::Data)

         log_π = data.α_m_m1 * log(Θ.p[2]) + data.β_m_m1 * log(1 - Θ.p[2]) + data.α_p_m1*log(Θ.p[3]) + data.β_p_m1 * log(1 - Θ.p[3]) + data.α_τ_m1 * log(Θ.p[1]) + data.β_τ_m1 * log(1 - Θ.p[1])

         for i ∈ eachindex(data.X)
           log_π += data.freq[i] * log( Θ.p[1] * (1 - Θ.p[2])^data.X[i] * Θ.p[2]^(data.NmX[i]) + (1-Θ.p[1]) * Θ.p[3]^data.X[i] * (1-Θ.p[3])^data.NmX[i] )
         end

         log_π

       end
```

We can define a convenience function for creaturing our data structure, which defaults our Beta parameters to one.
```
julia> function BinaryClassificationData(X::Array{Int, 1}, freq::Array{Int,1}, n::Int; αm::Real = 1, βm::Real = 1, αp::Real = 1, βp::Real = 1, ατ::Real = 1, βτ::Real = 1)
         BinaryClassificationData(X, freq, n .- X, αm - 1, βm - 1, αp - 1, βp - 1, ατ - 1, βτ - 1)
       end
BinaryClassificationData

julia> X = [0, 1, 2, 3, 4, 7, 8, 9];

julia> freq = [10, 2, 2, 1, 2, 3, 2, 16];

julia> data = BinaryClassificationData(X, freq, 9, βm = 2, βp = 2);
```
For our test error rates, we use Beta(1,2) priors, to add a little information that suggests the error rates are less than 1/2. The tests are binary, so if they weren't, we'd just reverse the results!

Now that we have data, we can calculate the joint posterior:

```
julia> jp = JointPosterior(bc_model, data);
```

But really, we're more interested in the marginals distributions of our three parameters. So, simply define functions finding the marginals we're interested in:
```
julia> τ(Θ::BinaryClassification) = Θ.p[1]
τ (generic function with 1 method)

julia> θ_minus(Θ::BinaryClassification) = Θ.p[2]
θ_minus (generic function with 1 method)

julia> θ_plus(Θ::BinaryClassification) = Θ.p[3]
θ_plus (generic function with 1 method)

```
And compute the marginals!
```
julia> marginal_τ = marginal(jp, τ)
Marginal parameter
μ: 0.5503617376062098
σ: 0.07786503681942877
Quantiles: [0.386014 0.48146 0.550112 0.550112 0.703985]


julia> marginal_θ_minus = marginal(jp, θ_minus)
Marginal parameter
μ: 0.04728258409326777
σ: 0.015587761958469353
Quantiles: [0.0220781 0.0353606 0.0451179 0.0573919 0.0809962]


julia> marginal_θ_plus = marginal(jp, θ_plus)
Marginal parameter
μ: 0.11511160848670976
σ: 0.025674446894039842
Quantiles: [0.0713712 0.0951648 0.112278 0.13201 0.170533]
```
###### Comparison with Stan

You can compare these results with that from MCMC. For example, using Stan
```
julia> using Stan, Mamba
julia> Stan_data = Dict( "n" => 9, "N" => 38, "X" => vcat(zeros(Int64, 10), ones(Int64, 2), fill(2,2), 3, fill(4, 2), fill(7, 3), fill(8, 2), fill(9, 16)));
julia> const binary = "
       data {
         int n;
         int N;
         int X[N];
       }
       transformed data{
         int NmX[N];
         for (i in 1:N){
           NmX[i] = n - X[i];
         }
       }
       parameters {
         real<lower = 0, upper = 1> tau;
         real<lower = 0, upper = 0.5> theta_minus;
         real<lower = 0, upper = 0.5> theta_plus;
       }
       transformed parameters {
         real OmTau;
         real OmTm;
         real OmTp;
         OmTau = 1 - tau;
         OmTm = 1 - theta_minus;
         OmTp = 1 - theta_plus;
       }
       model {
         vector[N] cache;
         OmTm ~ beta(1, 2);
         OmTp ~ beta(1, 2);
         for (i in 1:N){
           cache[i] = tau * OmTm^X[i] * theta_minus^NmX[i] + OmTau * theta_plus^X[i] * OmTp^NmX[i];
         }
         target += sum(log(cache));
       }
       ";

julia> binary_class_stan = Stanmodel(Sample(), name = "Binary", model = binary, monitors = ["tau", "theta_minus", "theta_plus"]);
julia> stan_res = stan(binary_class_stan, [Stan_data])
...snip...
Warmup took (0.11, 0.13, 0.12, 0.11) seconds, 0.48 seconds total
Sampling took (0.11, 0.11, 0.092, 0.099) seconds, 0.41 seconds total

                 Mean     MCSE  StdDev     5%    50%    95%  N_Eff  N_Eff/s    R_hat
lp__             -126  2.9e-02     1.2   -129   -126   -125   1835     4515  1.0e+00
accept_stat__    0.86  2.3e-03    0.14   0.56   0.91    1.0   4000     9845  1.0e+00
stepsize__        1.1  6.0e-02   0.085    1.0    1.2    1.3    2.0      4.9  4.3e+13
treedepth__       1.9  5.8e-03    0.34    1.0    2.0    2.0   3541     8716  1.0e+00
n_leapfrog__      2.9  9.5e-03    0.58    1.0    3.0    3.0   3796     9342  1.0e+00
divergent__      0.00  0.0e+00    0.00   0.00   0.00   0.00   4000     9845     -nan
energy__          128  4.0e-02     1.7    126    127    131   1810     4455  1.0e+00
tau              0.55  1.2e-03   0.077   0.42   0.55   0.68   4000     9845  1.0e+00
theta_minus     0.052  2.6e-04   0.016  0.028  0.050  0.082   4000     9845  1.0e+00
theta_plus       0.12  4.1e-04   0.026  0.082   0.12   0.17   4000     9845  1.0e+00
OmTau            0.45  1.2e-03   0.077   0.32   0.45   0.58   4000     9845  1.0e+00
OmTm             0.95  2.6e-04   0.016   0.92   0.95   0.97   4000     9845  1.0e+00
OmTp             0.88  4.1e-04   0.026   0.83   0.88   0.92   4000     9845  1.0e+00


julia> describe(stan_res[2])
Iterations = 1:1000
Thinning interval = 1
Chains = 1,2,3,4
Samples per chain = 1000

Empirical Posterior Estimates:
                Mean         SD        Naive SE        MCSE      ESS
        tau 0.550815186 0.077343408 0.00122290665 0.00110432852 4000
theta_minus 0.052320052 0.016412226 0.00025950008 0.00020854788 4000
 theta_plus 0.121547492 0.025631407 0.00040526812 0.00034733095 4000

Quantiles:
                2.5%       25.0%     50.0%      75.0%      97.5%  
        tau 0.398738025 0.49840800 0.5518635 0.60470475 0.69873750
theta_minus 0.025102625 0.04061495 0.0504063 0.06183745 0.08850898
 theta_plus 0.075059647 0.10359050 0.1199970 0.13848575 0.17445715
```

Note that total CPU time was just under a second. For comparison,
```
julia> function run_bc_model()
         jp = JointPosterior(bc_model, data)
         marginal_τ = marginal(jp, τ)
         marginal_θ_plus = marginal(jp, θ_plus)
         marginal_θ_minus = marginal(jp, θ_minus)
       end
run_bc_model (generic function with 1 method)

julia> using BenchmarkTools

julia> @benchmark run_bc_model()
BenchmarkTools.Trial: 
  memory estimate:  2.17 MiB
  allocs estimate:  39446
  --------------
  minimum time:     14.038 ms (0.00% GC)
  median time:      14.403 ms (0.00% GC)
  mean time:        14.683 ms (1.65% GC)
  maximum time:     17.794 ms (14.05% GC)
  --------------
  samples:          341
  evals/sample:     1

```
That is about 0.89 seconds for MCMC vs 15 milliseconds, about 60-fold faster using the default number of iterations (1,000 posterior samples for each of 4 chains).

#### Example 2: Hello World, Linear Regression!

Bob Carpenter [compared how](http://andrewgelman.com/2017/05/31/compare-stan-pymc3-edward-hello-world/) you implement linear regression in Stan, PyMC3, and Edward. 
To specify the model here, we just need:
```
julia> struct HiWorld{T} <: parameters{T,1}
         x::Vector{T}
         β::RealVector{3,T}
         σ::PositiveVector{1,T}
       end

julia> struct HiWorldData <: Data
         X::Array{Float64,2}
         y::Vector{Float64}
       end

julia> function log_density(Θ::HiWorld, data::Data)
         lpdf_normal(Θ.β, 0, 10) + lpdf_normal(Θ.σ[1], 0, 1) + lpdf_normal(data.y, data.X * Θ.β, Θ.σ[1])
       end
```

We can create our own dataset like that from the PyMC3 example.
```
julia> # True parameter values
       sigma = 1;

julia> beta = [1, 1, 2.5];

julia> # Size of dataset
       size = 100;

julia> # Predictor variable
       X = hcat(ones(size), randn(size,2));

julia> X[:,3] .*= 0.2;

julia> # Simulate outcome variable
       y = X*beta .+ randn(size) .* sigma;
```

Running the model:
```
julia> HW_data = HiWorldData(X, y);

julia> HW_model = Model(HiWorld);

julia> hw_jp = JointPosterior(HW_model, HW_data);

julia> marginal(hw_jp, x -> x.β[1])
Marginal parameter
μ: 0.8244634737023615
σ: 0.0986592330501434
Quantiles: [0.628894 0.690754 0.743527 0.743527 0.743527]


julia> marginal(hw_jp, x -> x.β[2])
Marginal parameter
μ: 0.8986692287796316
σ: 0.09196302695208347
Quantiles: [0.716489 0.823657 0.887789 0.89865 1.07577]


julia> marginal(hw_jp, x -> x.β[3])
Marginal parameter
μ: 2.8062485454846993
σ: 0.48421154364155206
Quantiles: [1.84043 2.43793 2.74962 3.16046 3.77357]


julia> marginal(hw_jp, x -> x.σ[1])
Marginal parameter
μ: 0.9563012862670179
σ: 0.07002800461069793
Quantiles: [0.834308 0.909538 0.959722 0.994694 1.08448]
```

To again compare with Stan:
```
const hw_stan = "data {
  int N;
  vector[N] y;
  matrix[N, 2] x;
}
parameters {
  real alpha;
  vector[2] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ normal(0, 1);
  y ~ normal(alpha + x * beta, sigma);
}"

hw_stan_data = Dict("N" => length(y), "x" => X[:,2:3], "y" => y)
hw_stan_model = Stanmodel(Sample(), name = "HelloWorld", model = hw_stan, monitors = ["alpha", "beta.1", "beta.2", "sigma"]);
hw_stan_res = stan(hw_stan_model, [hw_stan_data])


Warmup took (0.064, 0.050, 0.063, 0.054) seconds, 0.23 seconds total
Sampling took (0.054, 0.043, 0.057, 0.045) seconds, 0.20 seconds total

                Mean     MCSE  StdDev    5%   50%   95%  N_Eff  N_Eff/s    R_hat
lp__             -46  3.2e-02     1.4   -49   -45   -44   1994     9978  1.0e+00
accept_stat__   0.86  2.2e-03    0.14  0.59  0.90   1.0   4000    20015  1.0e+00
stepsize__       1.0  2.9e-02   0.041  0.97   1.0   1.1    2.0       10  2.5e+13
treedepth__      2.0  3.9e-03    0.24   1.0   2.0   2.0   3693    18479  1.0e+00
n_leapfrog__     3.2  1.4e-01     1.1   3.0   3.0   7.0     68      341  1.0e+00
divergent__     0.00  0.0e+00    0.00  0.00  0.00  0.00   4000    20015     -nan
energy__          48  4.8e-02     2.0    45    48    52   1810     9055  1.0e+00
alpha           0.83  1.6e-03   0.099  0.66  0.82  0.99   4000    20015  1.0e+00
beta[1]         0.90  1.4e-03   0.090  0.75  0.90   1.0   4000    20015  1.0e+00
beta[2]          2.8  7.4e-03    0.47   2.0   2.8   3.6   4000    20015  1.0e+00
sigma           0.96  1.1e-03   0.068  0.85  0.95   1.1   4000    20015  1.0e+00

Samples were drawn using hmc with nuts.
For each parameter, N_Eff is a crude measure of effective sample size,
and R_hat is the potential scale reduction factor on split chains (at 
convergence, R_hat=1).

julia> describe(hw_stan_res[2])
Iterations = 1:1000
Thinning interval = 1
Chains = 1,2,3,4
Samples per chain = 1000

Empirical Posterior Estimates:
                  Mean          SD        Naive SE       MCSE         ESS  
        alpha   0.82584693 0.095251639 0.00150606065 0.0011361759 4000.000000
       beta.1   0.89913245 0.090973327 0.00143841460 0.0016418622 4000.000000
       beta.2   2.79749220 0.485954717 0.00768361872 0.0068205937 4000.000000
        sigma   0.95532941 0.068626987 0.00108508794 0.0010426138 4000.000000


Quantiles:
                  2.5%        25.0%       50.0%        75.0%       97.5% 
        alpha   0.63904622   0.7603130   0.8246630   0.89123775   1.0082113
       beta.1   0.72446435   0.8348335   0.8989090   0.96287700   1.0769953
       beta.2   1.84282325   2.4680900   2.7950600   3.12770500   3.7521107
        sigma   0.83513470   0.9074153   0.9516395   0.99887875   1.1052230
```
