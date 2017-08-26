using JointPosteriors
using Base.Test


model = ( ProbabilityVector(3), )

struct BinaryClassificationData
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

function log_density(data, p)
    @inbounds log_π = data.α_m_m1 * log(p[2]) + data.β_m_m1 * log(1 - p[2]) + data.α_p_m1*log(p[3]) + data.β_p_m1 * log(1 - p[3]) + data.α_τ_m1 * log(p[1]) + data.β_τ_m1 * log(1 - p[1])

    @inbounds @simd for i ∈ eachindex(data.X)
        log_π += data.freq[i] * log( p[1] * (1 - p[2])^data.X[i] * p[2]^(data.NmX[i]) + (1-p[1]) * p[3]^data.X[i] * (1-p[3])^data.NmX[i] )
    end
    log_π
end

function BinaryClassificationData(X::Array{Int, 1}, freq::Array{Int,1}, n::Int; αm::Real = 1, βm::Real = 1, αp::Real = 1, βp::Real = 1, ατ::Real = 1, βτ::Real = 1)
  BinaryClassificationData(X, freq, n .- X, αm - 1, βm - 1, αp - 1, βp - 1, ατ - 1, βτ - 1)
end
X = [0, 1, 2, 3, 4, 7, 8, 9];
freq = [10, 2, 2, 1, 2, 3, 2, 16];
data = BinaryClassificationData(X, freq, 9, βm = 2, βp = 2);




function run_tests(model, data, μ, σ, quants, f, tol = 1e-2)
    builds = (SparseQuadratureGrids.AdaptiveRaw, SparseQuadratureGrids.Adaptive, SparseQuadratureGrids.SmolyakRaw, SparseQuadratureGrids.Smolyak)
    q_rules = (SparseQuadratureGrids.GenzKeister, SparseQuadratureGrids.KronrodPatterson)
    @testset begin
        @testset "Build: $b, Q rule: $q." for b ∈ builds, q ∈ q_rules
            mod = Model(model, b{q})
            jp = fit(mod, data)
            m_grid = marginal(jp, f)
            m_norm = marginal(jp, f, Normal)
            @test isapprox(m_norm.μ, μ, rtol = tol)
            @test isapprox(m_norm.σ, σ, rtol = tol)
            @test isapprox(quantile(m_grid, .025), quants[1], rtol = tol)
            @test isapprox(quantile(m_grid, .25), quants[2], rtol = tol)
            @test isapprox(quantile(m_grid, .5), quants[3], rtol = tol)
            @test isapprox(quantile(m_grid, .75), quants[4], rtol = tol)
            @test isapprox(quantile(m_grid, .975), quants[5], rtol = tol)
            println("\n")
            println(quantile.(m_norm, [.025,.25,.5,.75,.975]))
            println("\n")
            @test isapprox(quantile(m_norm, .025), quants[1], rtol = tol)
            @test isapprox(quantile(m_norm, .25), quants[2], rtol = tol)
            @test isapprox(quantile(m_norm, .5), quants[3], rtol = tol)
            @test isapprox(quantile(m_norm, .75), quants[4], rtol = tol)
            @test isapprox(quantile(m_norm, .975), quants[5], rtol = tol)
        end
    end
end

quants = [0.391, 0.495, 0.55, 0.602, 0.696]
run_tests(model, data, 0.5504, 0.077, quants, p -> p[1], 10^-1.5)
