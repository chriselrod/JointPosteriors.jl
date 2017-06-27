
using JointPosteriors

struct BinaryClassification{T} <: parameters{T}
  x::Vector{T}
  p::ProbabilityVector{3,T}
end


function log_density(Θ::BinaryClassification, data::Data)

  log_π = data.αm_m1 * log(Θ.p[3]) + data.βm_m1 * log(1 - Θ.p[3]) + data.αp_m1*log(Θ.p[2]) + data.βp_m1 * log(1 - Θ.p[2]) + data.α0_m1 * log(Θ.p[1]) + data.β0_m1 * log(1 - Θ.p[1])

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
  α0_m1::Float64
  β0_m1::Float64

end

function BinaryClassificationData(x::Array{Int64, 1}, Freq::Array{Int64,1}; αm::Real = 1, βm::Real = 1, αp::Real = 1, βp::Real = 1, α0::Real = 1, β0::Real = 1)
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


  BinaryClassificationData(X, freq, N .- X, N, αm - 1, βm - 1, αp - 1, βp - 1, α0 - 1, β0 - 1)
end

X = [i for i ∈ 0:9];
freq = [10, 2, 2, 1, 2, 0, 0, 3, 2, 16];
data = BinaryClassificationData(X, freq, βm = 2, βp = 2);

bc_model = Model(BinaryClassification)

jp = JointPosterior(bc_model, data)

function τ(Θ::BinaryClassification)
  Θ.p[1]
end
function θ_plus(Θ::BinaryClassification)
  Θ.p[2]
end
function θ_minus(Θ::BinaryClassification)
  Θ.p[3]
end

marginal_τ = marginal(jp, τ)
marginal_θ_plus = marginal(jp, θ_plus)
marginal_θ_minus = marginal(jp, θ_minus)
