
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




using Stan, Mamba
set_cmdstan_home!("/mnt/ssd/cmdstan-2.14.0")

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
describe(hw_stan_res[2])
