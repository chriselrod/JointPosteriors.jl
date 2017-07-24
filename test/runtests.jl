using JointPosteriors
using Base.Test

# write your own tests here
struct BinaryClassification{T} <: parameters{T,1}
  x::Vector{T}
  p::ProbabilityVector{3,T}
end
struct BinaryClassificationData <: Data
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

function log_density(Θ::BinaryClassification, data::Data)
  log_π = data.α_m_m1 * log(Θ.p[2]) + data.β_m_m1 * log(1 - Θ.p[2]) + data.α_p_m1*log(Θ.p[3]) + data.β_p_m1 * log(1 - Θ.p[3]) + data.α_τ_m1 * log(Θ.p[1]) + data.β_τ_m1 * log(1 - Θ.p[1])
  for i ∈ eachindex(data.X)
    log_π += data.freq[i] * log( Θ.p[1] * (1 - Θ.p[2])^data.X[i] * Θ.p[2]^(data.NmX[i]) + (1-Θ.p[1]) * Θ.p[3]^data.X[i] * (1-Θ.p[3])^data.NmX[i] )
  end
  log_π
end

function BinaryClassificationData(X::Array{Int, 1}, freq::Array{Int,1}, n::Int; αm::Real = 1, βm::Real = 1, αp::Real = 1, βp::Real = 1, ατ::Real = 1, βτ::Real = 1)
  BinaryClassificationData(X, freq, n .- X, αm - 1, βm - 1, αp - 1, βp - 1, ατ - 1, βτ - 1)
end
X = [0, 1, 2, 3, 4, 7, 8, 9];
freq = [10, 2, 2, 1, 2, 3, 2, 16];
data = BinaryClassificationData(X, freq, 9, βm = 2, βp = 2);

bc_model_gk = Model(BinaryClassification)
jp_gk = JointPosterior(bc_model_gk, data)

bc_model_kp = Model(BinaryClassification, q = SparseQuadratureGrids.KronrodPatterson)
jp_kp = JointPosterior(bc_model_kp, data)

@testset begin
  @test isapprox(marginal(jp_gk, x -> x.p[1]).μ, 0.5504, rtol = 2e-4)
  @test isapprox(marginal(jp_kp, x -> x.p[1]).μ, 0.5504, rtol = 2e-4)
end
