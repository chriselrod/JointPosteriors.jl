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
m_gk_grid = marginal(jp_gk, x -> x.p[1])
m_gk_normal = marginal(jp_gk, x -> x.p[1], Normal)

bc_model_kp = Model(BinaryClassification, q = SparseQuadratureGrids.KronrodPatterson)
jp_kp = JointPosterior(bc_model_kp, data)
m_kp_grid = marginal(jp_kp, x -> x.p[1])
m_kp_normal = marginal(jp_kp, x -> x.p[1], Normal)

@testset begin
  @testset begin
    @test isapprox(m_gk_grid.μ, 0.5503061164407677, rtol = 2e-8)
    @test isapprox(m_gk_grid.σ, 0.07661935890124291, rtol = 2e-8)
    @test isapprox(quantile(m_gk_normal, .025), 0.4008858361449898, rtol = 2e-8)
    @test isapprox(quantile(m_gk_normal, .25), 0.4971142935368797, rtol = 2e-8)
    @test isapprox(quantile(m_gk_normal, .5), 0.5501321112304882, rtol = 2e-8)
    @test isapprox(quantile(m_gk_normal, .75), 0.6028233244941255, rtol = 2e-8)
    @test isapprox(quantile(m_gk_normal, .975), 0.6970692930288663, rtol = 2e-8)
    @test isapprox(quantile(m_gk_grid, .025), 0.39011219404999875, rtol = 2e-6)
    @test isapprox(quantile(m_gk_grid, .25), 0.4993133361132443, rtol = 2e-6)
    @test isapprox(quantile(m_gk_grid, .5), 0.5473127946502362, rtol = 2e-6)
    @test isapprox(quantile(m_gk_grid, .75), 0.6035318273300528, rtol = 2e-6)
    @test isapprox(quantile(m_gk_grid, .975), 0.7047551520876153, rtol = 2e-6)
  end
  @testset begin
    @test isapprox(m_kp_grid.μ, 0.5504463054796331, rtol = 2e-8)
    @test isapprox(m_kp_grid.σ, 0.07777781894846478, rtol = 2e-8)
    @test isapprox(quantile(m_kp_normal, .025), 0.39106678659093086, rtol = 2e-8)
    @test isapprox(quantile(m_kp_normal, .25), 0.49481672560058265, rtol = 2e-8)
    @test isapprox(quantile(m_kp_normal, .5), 0.5488704022945908, rtol = 2e-8)
    @test isapprox(quantile(m_kp_normal, .75), 0.6015558314490006, rtol = 2e-8)
    @test isapprox(quantile(m_kp_normal, .975), 0.6960897346671988, rtol = 2e-8)
    @test isapprox(quantile(m_kp_grid, .025), 0.3931513592679271, rtol = 2e-6)
    @test isapprox(quantile(m_kp_grid, .25), 0.49486042524846835, rtol = 2e-6)
    @test isapprox(quantile(m_kp_grid, .5), 0.549428799113842, rtol = 2e-6)
    @test isapprox(quantile(m_kp_grid, .75), 0.6024636935286022, rtol = 2e-6)
    @test isapprox(quantile(m_kp_grid, .975), 0.6960955354476359, rtol = 2e-6)
  end
end
