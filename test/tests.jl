


using JointPosteriors

struct BinaryClassification{T} <: parameters{T,1}
  x::Vector{T}
  p::ProbabilityVector{3,T}
end



function log_density(Θ::BinaryClassification, data::Data)

  log_π = data.αm_m1 * log(Θ.p[3]) + data.βm_m1 * log(1 - Θ.p[3]) + data.αp_m1*log(Θ.p[2]) + data.βp_m1 * log(1 - Θ.p[2]) + data.ατ_m1 * log(Θ.p[1]) + data.βτ_m1 * log(1 - Θ.p[1])

  for i ∈ eachindex(data.X)
    log_π += data.freq[i] * log( Θ.p[1] * (1 - Θ.p[3])^data.X[i] * Θ.p[3]^(data.NmX[i]) + (1-Θ.p[1]) * Θ.p[2]^data.X[i] * (1-Θ.p[2])^data.NmX[i] )
  end

  log_π

end

struct BinaryClassificationData <: Data
  X::Array{Int64,1}
  freq::Array{Int64,1}
  NmX::Array{Int64, 1}
  N::Int64
  αm_m1::Float64
  βm_m1::Float64
  αp_m1::Float64
  βp_m1::Float64
  ατ_m1::Float64
  βτ_m1::Float64

end

function BinaryClassificationData(x::Array{Int64, 1}, Freq::Array{Int64,1}; αm::Real = 1, βm::Real = 1, αp::Real = 1, βp::Real = 1, ατ::Real = 1, βτ::Real = 1)
  N = x[end]
  non_zero_freq = sum( Freq .!= 0 )
  X = zeros( Int64, non_zero_freq )
  freq = zeros( Int64, non_zero_freq )
  j = 1
  for i ∈ 1:non_zero_freq
    while Freq[j] == 0
      j += 1
    end
    X[i] = x[j]
    freq[i] = Freq[j]
    j += 1
  end
  BinaryClassificationData(X, freq, N .- X, N, αm - 1, βm - 1, αp - 1, βp - 1, ατ - 1, βτ - 1)
end

X = [i for i ∈ 0:9];
freq = [10, 2, 2, 1, 2, 0, 0, 3, 2, 16];
data = BinaryClassificationData(X, freq, βm = 2, βp = 2);

bc_model = Model(BinaryClassification)

jp = JointPosterior(bc_model, data);

τ(Θ::BinaryClassification) = Θ.p[1]
θ_plus(Θ::BinaryClassification) = Θ.p[2]
θ_minus(Θ::BinaryClassification) = Θ.p[3]


marginal_τ = marginal(jp, τ)

marginal_θ_plus = marginal(jp, θ_plus)
marginal_θ_minus = marginal(jp, θ_minus)


f(x) = x.p[1]
d = Distributions.Normal
q = SparseQuadratureGrids.GenzKeister

values_positive = Array{Float64}(length(jp.weights_positive));
values_negative = Array{Float64}(length(jp.weights_negative));
for i ∈ eachindex(values_positive)
  values_positive[i] = f(JointPosteriors.transform!(jp.M.Θ, jp.M.Grid.nodes_positive[:,i], jp.Θ_hat, jp.L, q))
end
for i ∈ eachindex(values_negative)
  values_negative[i] = f(JointPosteriors.transform!(jp.M.Θ, jp.M.Grid.nodes_negative[:,i], jp.Θ_hat, jp.L, q))
end
μ = dot(jp.weights_positive, values_positive) + dot(jp.weights_negative, values_negative)
σ = sqrt(dot(jp.weights_positive, values_positive .^ 2) + dot(jp.weights_negative, values_negative .^ 2) - μ^2)

poly = JointPosteriors.polynomial_interpolation(jp.weights_positive, jp.weights_negative, values_positive, values_negative, d)

cdf_error(zeros(6), poly)


function prec(n::Int, ρ::Real)
  p = zeros(n,n)
  np1 = n+1
  np1h2 = (np1/2)^2
  β2 = 1 / ( np1h2 - n )
  β1 = -np1*β2
  β0 = 2 + np1h2*β2
  p[1] = p[end] = 3
  det_im1 = 3
  det_im2 = 1
  a0 = 3
  for i ∈ 2:n
    p[i,i] = a = (β0 + β1*i + β2*i^2)
    b = a*a0*ρ/4
    p[i,i-1] = p[i-1,i] = -sqrt(b)
    det_i = a*det_im1-b*det_im2
    det_im2 = det_im1
    det_im1 = det_i
    a0 = a
  end
  p, det_im1
end
function cet(δ::Vector, n::Int, ρ::Real)
  np1 = n+1
  np1h2 = (np1/2)^2
  β2 = 1 / ( np1h2 - n )
  β1 = -np1*β2
  β0 = 2 + np1h2*β2
  det_im1 = 3.0
  det_im2 = 1.0
  out = 3.0 * δ[1]^2
  a0 = 3.0
  for i ∈ 2:n
    a = (β0 + β1*i + β2*i^2)
    b = a*a0*ρ/4
    out += δ[i]^2*a - 2δ[i-1]*δ[i]*√b

    det_i = a*det_im1-b*det_im2
    det_im2 = det_im1
    det_im1 = det_i
    a0 = a
  end
  out, det_im1
end


itp = JointPosteriors.InterpolateIntegral(jp.weights_positive, jp.weights_negative, values_positive, values_negative, d)



using JointPosteriors
struct HiWorld{T} <: parameters{T,1}
  x::Vector{T}
  β::RealVector{3,T}
  σ::PositiveVector{1,T}
end

struct HiWorldData <: Data
  X::Array{Float64,2}
  y::Vector{Float64}
end

function log_density(Θ::HiWorld, data::Data)
  lpdf_normal(Θ.β, 0, 10) + lpdf_normal(Θ.σ[1], 0, 1) + lpdf_normal(data.y, data.X * Θ.β, Θ.σ[1])
end


# True parameter values
sigma = 1;
beta = [1, 1, 2.5];

# Size of dataset
size = 100;

# Predictor variable
X = hcat(ones(size), randn(size,2));
X[:,3] .*= 0.2;

# Simulate outcome variable
y = X*beta .+ randn(size) .* sigma;



HW_data = HiWorldData(X, y);
HW_model = Model(HiWorld);
hw_jp = JointPosterior(HW_model, HW_data);

marginal(hw_jp, x -> x.β[1])
marginal(hw_jp, x -> x.β[2])
marginal(hw_jp, x -> x.β[3])
marginal(hw_jp, x -> x.σ[1])





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
using Stan, Mamba
set_cmdstan_home!("/mnt/ssd/cmdstan-2.14.0")
#set_cmdstan_home!("/home/christel/Downloads/cmdstan-2.14.0")

hw_stan_data = Dict("N" => length(y), "x" => X[:,2:3], "y" => y)
hw_stan_model = Stanmodel(Sample(), name = "HelloWorld", model = hw_stan, monitors = ["alpha", "beta.1", "beta.2", "sigma"]);
hw_stan_res = stan(hw_stan_model, [hw_stan_data])
describe(hw_stan_res[2])

function run_hw()
  hw_jp = JointPosterior(HW_model, HW_data);
  marginal(hw_jp, x -> x.β[1])
  marginal(hw_jp, x -> x.β[2])
  marginal(hw_jp, x -> x.β[3])
  marginal(hw_jp, x -> x.σ[1])
end
using BenchmarkTools
@benchmark run_hw()






using JointPosteriors

include("/mnt/ssd/MSA/Univariate/dataGen.jl")

tv = true_values_uni(15, √99, √0.6, √0.3, √.1);
y, yp, yo = gen_data(tv, 40, 12, 12);

d = LogDensities.TF_RE_ANOVA_Data(y, yp, yo);
anova = Model(LogDensities.TF_RE_ANOVA);

jp = JointPosterior(anova, d);

function rGT(Θ::LogDensities.TF_RE_ANOVA)
  σg2 = sum(Θ.σ[2:end])
  sqrt(σg2 / (σg2 + Θ.σ[1]))
end

marginal_rGT = marginal(jp, rGT)

function run_anova()
  jp = JointPosterior(anova, d)
  marginal_rGT = marginal(jp, rGT)
end
73, 67, 75, 81) seconds, 4.9 minutes total
Sampling took (70, 103, 69, 72)
const tfanova = "
data {
  int Np1;
  int p;
  int o;
  int POp1;
  matrix[p, o] cat_count;
  matrix[p, o] y_bars;
  real ns2h;
  real cauchy_spread;
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
  sigma_o ~ cauchy(0, cauchy_spread);
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
}
"

using Stan, Mamba#, Gadfly
set_cmdstan_home!("/mnt/ssd/cmdstan-2.14.0")
anova_stan = Stanmodel(Sample(), name = "ANOVA", model = tfanova, monitors = ["sigma_g", "sigma_t", "rGT"]);

function StanDataANOVA(data::TF_RE_ANOVA_Data)
  Dict( "Np1" => data.N + 1, "p" => data.P, "Pp1" => data.P + 1, "o" => data.O, "POp1" => data.PO + 1, "cat_count" => fill(data.R, data.P, data.O), "y_bars" => data.δ .+ data.μ_hat, "ns2h" =>  - data.s2 / 2.0)
end

res = stan(anova_stan, [StanDataANOVA(data)])
describe(res[2])
