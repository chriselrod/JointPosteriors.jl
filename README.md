# JointPosteriors

[![Build Status](https://travis-ci.org/chriselrod/JointPosteriors.jl.svg?branch=master)](https://travis-ci.org/chriselrod/JointPosteriors.jl)

[![Coverage Status](https://coveralls.io/repos/chriselrod/JointPosteriors.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chriselrod/JointPosteriors.jl?branch=master)

[![codecov.io](http://codecov.io/github/chriselrod/JointPosteriors.jl/coverage.svg?branch=master)](http://codecov.io/github/chriselrod/JointPosteriors.jl?branch=master)

## Introduction

This package implements sparse grid quadrature. I'm working on documentation, organization, tests, and adding more features. And then on getting this package registered. Until then, you can still install via
```julia
julia> Pkg.clone("https://github.com/chriselrod/SparseQuadratureGrids.jl")
julia> Pkg.clone("https://github.com/chriselrod/LogDensities.jl")
julia> Pkg.clone("https://github.com/chriselrod/JointPosteriors.jl")
```

#### Example 1: Binary Classification

As an example, consider the model from:
> Beavers, D. P., Stamey, J. D., & Bekele, B. N. (2011). A Bayesian model to assess a binary measurement system when no gold standard system is available. Journal of Quality Technology, 43(1), 16.

In the model, there was some number (N) of parts that were defective with probability 1-tau. We have a test that is imperfect, with a false-effective rate of theta+, and a false-defective rate of theta-. For example, if a part is actually defective, the test will rate it as effective with probability theta+. We can test each part (n) times, and we treat each trial as independent.

So, our parameters are a total of three different probabilities. We can specify this via building a parameter struct, and specifying that it has a probability vector of length 3. We'll let the first index denote tau, the second theta_minus, and the last theta_plus.
```julia
julia> using JointPosteriors

julia> struct BinaryClassification{T} <: parameters{T,1}
         x::Vector{T}
         p::ProbabilityVector{3,T}
       end
```
Currently, when defining parameter structures, the first field must always be `x::Vector{T}`. This may change eventually (depending in part by how much people hate that).

That is all we need to initialize our model:
```julia
julia> bc_model = Model(BinaryClassification);
```

But before we really do anything with it, we need data nad a log likelihood function. We have a lot of freedom to write these parts however we'd like.
```julia
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

```julia
julia> function log_density(Θ::BinaryClassification, data::Data)

         log_π = data.α_m_m1 * log(Θ.p[2]) + data.β_m_m1 * log(1 - Θ.p[2]) + data.α_p_m1*log(Θ.p[3]) + data.β_p_m1 * log(1 - Θ.p[3]) + data.α_τ_m1 * log(Θ.p[1]) + data.β_τ_m1 * log(1 - Θ.p[1])

         for i ∈ eachindex(data.X)
           log_π += data.freq[i] * log( Θ.p[1] * (1 - Θ.p[2])^data.X[i] * Θ.p[2]^(data.NmX[i]) + (1-Θ.p[1]) * Θ.p[3]^data.X[i] * (1-Θ.p[3])^data.NmX[i] )
         end

         log_π

       end
```

We can define a convenience function for creaturing our data structure, which defaults our Beta parameters to one.
```julia
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

```julia
julia> jp = JointPosterior(bc_model, data);
```

But really, we're more interested in the marginals distributions of our three parameters. So, simply define functions finding the marginals we're interested in:
```julia
julia> τ(Θ::BinaryClassification) = Θ.p[1]
τ (generic function with 1 method)

julia> θ_minus(Θ::BinaryClassification) = Θ.p[2]
θ_minus (generic function with 1 method)

julia> θ_plus(Θ::BinaryClassification) = Θ.p[3]
θ_plus (generic function with 1 method)

```
And compute the marginals!
```julia
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
```julia
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
         simplex[3] theta;
       }
       transformed parameters {
         real OmTau;
         real OmTm;
         real OmTp;
         OmTau = 1 - tau;
         OmTm = 1 - theta[1];
         OmTp = 1 - theta[2];
       }
       model {
         vector[N] cache;
         theta[1] ~ beta(1, 2);
         theta[2] ~ beta(1, 2);
         for (i in 1:N){
           cache[i] = tau * OmTm^X[i] * theta[1]^NmX[i] + OmTau * theta[2]^X[i] * OmTp^NmX[i];
         }
         target += sum(log(cache));
       }";

julia> binary_class_stan = Stanmodel(Sample(), name = "Binary", model = binary, monitors = ["tau", "theta"]);
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
```julia
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
```julia
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
```julia
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
```julia
julia> HW_data = HiWorldData(X, y);

julia> HW_model = Model(HiWorld);

julia> hw_jp = JointPosterior(HW_model, HW_data);

julia> marginal(hw_jp, x -> x.β[1])
Marginal parameter
μ: 0.9637224944125826
σ: 0.10149066254154264
Quantiles: [0.761912 0.885964 0.944868 1.04253 1.16533]


julia> marginal(hw_jp, x -> x.β[2])
Marginal parameter
μ: 0.9786158365167603
σ: 0.10309823786508965
Quantiles: [0.776268 0.915435 0.9646 1.0459 1.18057]


julia> marginal(hw_jp, x -> x.β[3])
Marginal parameter
μ: 1.8130588530423797
σ: 0.47275126327130096
Quantiles: [0.865028 1.44886 1.72537 2.17061 2.76065]


julia> marginal(hw_jp, x -> x.σ[1])
Marginal parameter
μ: 0.9735241762516216
σ: 0.07128291739081981
Quantiles: [0.841453 0.917923 0.974857 1.01546 1.1056]
```

To again compare with Stan:
```julia
julia> using Stan, Mamba
julia> const hw_stan = "data {
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
       }";


julia> hw_stan_data = Dict("N" => length(y), "x" => X[:,2:3], "y" => y);
julia> hw_stan_model = Stanmodel(Sample(), name = "HelloWorld", model = hw_stan, monitors = ["alpha", "beta.1", "beta.2", "sigma"]);
julia> hw_stan_res = stan(hw_stan_model, [hw_stan_data])


Warmup took (0.049, 0.063, 0.065, 0.060) seconds, 0.24 seconds total
Sampling took (0.058, 0.059, 0.070, 0.046) seconds, 0.23 seconds total

                Mean     MCSE  StdDev    5%   50%   95%  N_Eff  N_Eff/s    R_hat
alpha           0.96  1.6e-03   0.099  0.81  0.96   1.1   4000    17105  1.0e+00
beta[1]         0.98  1.6e-03   0.099  0.81  0.98   1.1   4000    17105  1.0e+00
beta[2]          1.8  7.3e-03    0.46   1.0   1.8   2.6   4000    17105  1.0e+00
sigma           0.97  1.1e-03   0.071  0.87  0.97   1.1   4000    17105  1.0e+00

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
          Mean       SD       Naive SE        MCSE      ESS
 alpha 0.9640783 0.09935180 0.0015708900 0.00143550485 1000
beta.1 0.9792272 0.09888484 0.0015635066 0.00131295816 1000
beta.2 1.8068558 0.46109904 0.0072906159 0.00607401839 1000
 sigma 0.9747241 0.07094793 0.0011217853 0.00089428097 1000

Quantiles:
          2.5%       25.0%     50.0%     75.0%    97.5%  
 alpha 0.77095010 0.89520000 0.9648350 1.029915 1.1616973
beta.1 0.78276900 0.91301350 0.9785735 1.044990 1.1700938
beta.2 0.89037090 1.49718500 1.8089600 2.119275 2.6946383
 sigma 0.84931168 0.92615475 0.9699075 1.019295 1.1260908

```
Speed comparison:
```julia
julia> function run_hw()
         hw_jp = JointPosterior(HW_model, HW_data);
         marginal(hw_jp, x -> x.β[1])
         marginal(hw_jp, x -> x.β[2])
         marginal(hw_jp, x -> x.β[3])
         marginal(hw_jp, x -> x.σ[1])
       end
run_hw (generic function with 1 method)

julia> using BenchmarkTools

julia> @benchmark run_hw()
BenchmarkTools.Trial: 
  memory estimate:  14.75 MiB
  allocs estimate:  152942
  --------------
  minimum time:     28.582 ms (0.00% GC)
  median time:      32.269 ms (9.78% GC)
  mean time:        31.520 ms (6.84% GC)
  maximum time:     35.218 ms (9.30% GC)
  --------------
  samples:          159
  evals/sample:     1
```

Again, much faster. However, while the means and standard deviations are spot on, the quantile estimates in this example are poor.
The interpolation used for generating the quantiles is in dire need of revamping. They are generally biased slightly conservative, but certain conditions can cause this bias to balloon.


#### Example 3: ANOVA

The LogDensities package also includes optimized versions of several popular models.
For example, it includes a two factor random effects ANOVA with a folded Cauchy prior on the second factor variance, and improper priors elsewhere as in
> Weaver, B. P., Hamada, M. S., Vardeman, S. B., & Wilson, A. G. (2012). A bayesian approach to the analysis of gauge r&r data. Quality Engineering, 24(4), 486-500.

Specifying the model is straightforward:
```julia
julia> anova = Model(LogDensities.TF_RE_ANOVA);
```
We can use the Distributions package to generate a sample data set:
```julia
julia> using Distributions

julia> function gen_data(μ::Real, σ_P::Real, σ_O::Real, σ_PO::Real, σ_R::Real, P::Int64, O::Int64, R::Int64)::Tuple{Array{Float64,1},Array{Int64,1},Array{Int64,1}}
         PO = P*O
         POR = PO*R
         θp_true = rand(Normal(0, σ_P), P)
         θo_true = rand(Normal(0, σ_O), O)
         θpo_true = Array{Float64}(P, O)
         Npo = Normal(0, σ_PO)
         for p ∈ 1:P
           θpo_true[p,:] .= rand(Npo, O)
         end
         yp = Vector{Int64}(POR)
         yo = Vector{Int64}(POR)
         y = Vector{Float64}(POR)
         i = 1:R
         for p ∈ 1:P
           for o ∈ 1:O
             y[i,:] .= rand(Normal(μ + θp_true[p] + θo_true[o] + θpo_true[p,o], σ_R), R)
             yp[i] = p
             yo[i] = o
             i += R
           end
         end
         y, yp, yo
       end
gen_data (generic function with 1 method)

julia> y, yp, yo = gen_data(15, √99, √0.6, √0.3, √.1, 40, 12, 12);
```
The vector y contains data, while vectors yp and yo indicate group membership for the first and second factors, respectively. That is,
```julia
julia> y[200], yp[200], yo[200]
(27.661953803848693, 2, 5)
```
The 200th measurement was made of the second part (factor 1) by the fifth operator (factor 2). And that measurement was roughly 27.7.

Now that we have data, we can analyze it. Calling the two factor random effects ANOVA data function from the log densities module, and constructing the joint posterior:
```julia
julia> d = LogDensities.TF_RE_ANOVA_Data(y, yp, yo);
julia> typeof(d)
LogDensities.TF_RE_ANOVA_Data_balanced
julia> jp = JointPosterior(anova, d);
```
One of the primary parameters of interest was the ratio of variability not attributable solely to the part (termed gauge variability) to the total variability. Finding the marginal:
```julia
julia> function rGT(Θ::LogDensities.TF_RE_ANOVA)
         σg2 = sum(Θ.σ[2:end])
         sqrt(σg2 / (σg2 + Θ.σ[1]))
       end
rGT (generic function with 1 method)

julia> marginal_rGT = marginal(jp, rGT)
Marginal parameter
μ: 0.08517244124701978
σ: 0.015172385858816277
Quantiles: [0.0605517 0.0741515 0.0844895 0.0939557 0.119678]
```
Again, we evoke the Stan comparison. The model below simplified the likelihood by using cell means instead of iterating over each replication, although it is possible to simplify the calculations much further. A difference between the model below (and that from the LogDensities package) is that they pin the grand mean at the sample mean instead of using a flat prior.
```julia
julia> const tfanova = "
       data {
         int Np1;
         int p;
         int o;
         int POp1;
         matrix[p, o] cat_count;
         matrix[p, o] y_bars;
         real ns2h;
       }
       transformed data {
         real n;
         real po;
         matrix[p, o] invRootCounts;
         invRootCounts = inv_sqrt(cat_count);
         n = sum(cat_count);
         po = p * o;
       }
       parameters {
         real mu_G;
         vector[p] theta_p;
         vector[o] theta_o;
         matrix[p,o] theta_po;
         real<lower = 0> sigma_p;
         real<lower = 0> sigma_o;
         real<lower = 0> sigma_po;
         real<lower = 0> sigma_r;
       }
       model {
         matrix[p,o] quad;
         for (i in 1:p){
           theta_po[i, ] ~ normal(0, sigma_po);
           for (j in 1:o){
             y_bars[i,j] ~ normal( mu_G + theta_p[i] + theta_o[j] + theta_po[i, j], invRootCounts[i,j] * sigma_r);
           }
         }
         mu_G ~ normal(0, 50);
         theta_p ~ normal(0, sigma_p);
         theta_o ~ normal(0, sigma_o);
         sigma_o ~ cauchy(0, 20);
         target += ns2h / sigma_r^2 - (n - po) * log(sigma_r) ;
       }
       generated quantities {
         real sigma_g2;
         real sigma_g;
         real sigma_t;
         real rGT;
         sigma_g2 = sigma_o^2 + sigma_po^2 + sigma_r^2;
         sigma_g = sqrt(sigma_g2);
         sigma_t = sqrt(sigma_g2 + sigma_p^2);
         rGT = sigma_g / sigma_t;
       }";

julia> using Stan, Mamba
julia> anova_stan = Stanmodel(Sample(), name = "ANOVA", model = tfanova, monitors = ["sigma_g", "sigma_t", "rGT"]);
File /mnt/ssd/Projects/SparseGrid/tmp/ANOVA.stan will be updated.
julia> function StanDataANOVA(data::Data)
         Dict( "Np1" => data.N + 1, "p" => data.P, "Pp1" => data.P + 1, "o" => data.O, "POp1" => data.PO + 1, "cat_count" => fill(data.R, data.P, data.O), "y_bars" => data.δ .+ data.μ_hat, "ns2h" =>  - data.s2 / 2.0)
       end
StanDataANOVA (generic function with 1 method)

julia> res = stan(anova_stan, [StanDataANOVA(d)])
Warmup took (73, 67, 75, 81) seconds, 4.9 minutes total
Sampling took (70, 103, 69, 72) seconds, 5.2 minutes total
                      Mean     MCSE   StdDev        5%       50%       95%  N_Eff  N_Eff/s    R_hat
sigma_p            1.1e+01  4.5e-02  1.3e+00   9.1e+00   1.1e+01   1.3e+01    802  2.6e+00  1.0e+00
sigma_o            6.3e-01  6.3e-03  1.7e-01   4.3e-01   6.0e-01   9.4e-01    705  2.2e+00  1.0e+00
sigma_po           5.6e-01  6.5e-04  2.0e-02   5.3e-01   5.6e-01   6.0e-01    945  3.0e+00  1.0e+00
sigma_r            3.2e-01  8.4e-05  3.1e-03   3.1e-01   3.2e-01   3.2e-01   1376  4.4e+00  1.0e+00
sigma_g2           8.5e-01  9.2e-03  2.6e-01   6.0e-01   7.8e-01   1.3e+00    766  2.4e+00  1.0e+00
sigma_g            9.1e-01  4.5e-03  1.2e-01   7.7e-01   8.9e-01   1.1e+00    735  2.3e+00  1.0e+00
sigma_t            1.1e+01  4.5e-02  1.3e+00   9.1e+00   1.1e+01   1.3e+01    800  2.5e+00  1.0e+00
rGT                8.4e-02  4.7e-04  1.5e-02   6.4e-02   8.3e-02   1.1e-01    958  3.1e+00  1.0e+00
julia> describe(res[2])
Iterations = 1:1000
Thinning interval = 1
Chains = 1,2,3,4
Samples per chain = 1000

Empirical Posterior Estimates:
            Mean         SD        Naive SE        MCSE        ESS   
sigma_g  0.91316001 0.123253102 0.00194880266 0.00437419976 793.95976
sigma_t 10.99113350 1.280206526 0.02024184248 0.04232975496 914.67870
    rGT  0.08415345 0.014655471 0.00023172334 0.00046763508 982.16706

Quantiles:
            2.5%       25.0%       50.0%       75.0%        97.5%   
sigma_g 0.759504575  0.82847025  0.88556350  0.96803125  1.222992500
sigma_t 8.799354000 10.08725000 10.89855000 11.78357500 13.809942500
    rGT 0.061066755  0.07390550  0.08271435  0.09192043  0.118740275
```

Four chains with 1,000 posterior samples (but a total effective sample size of < 1,000) took just over 10 minutes. For comparison:
```julia
julia> function run_anova()
         jp = JointPosterior(anova, d)
         marginal_rGT = marginal(jp, rGT)
       end
run_anova (generic function with 1 method)

julia> using BenchmarkTools

julia> @benchmark run_anova()
BenchmarkTools.Trial: 
  memory estimate:  2.94 MiB
  allocs estimate:  48594
  --------------
  minimum time:     13.004 ms (0.00% GC)
  median time:      13.573 ms (0.00% GC)
  mean time:        14.041 ms (3.28% GC)
  maximum time:     18.057 ms (20.52% GC)
  --------------
  samples:          356
  evals/sample:     1
```
