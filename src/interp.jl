abstract type InterpolateIntegral end

struct weights_values <: AbstractArray{Float64,1}
  weights::Vector{Float64}
  values::Vector{Float64}
end

function simultaneous_sort!(wv::weights_values)
  si = sortperm(wv.values)
  wv.values .= wv.values[si]
  wv.weights .= wv.weights[si]
  wv
end

function interpolate_weight_values(wv::weights_values)
  simultaneous_sort!(wv)
  interpolate((wv.values,), cumsum(wv.weights), Gridded(Linear()))
end


struct polynomial_interpolation{d <: ContinuousUnivariateDistribution}
  X::Array{Float64, 2}
  y::Vector{Float64}
  n::Int
  diag::Vector{Float64}
  off_diag::Vector{Float64}
end


OneParamReal = Union{TDist, VonMises}
OneParamRealScale = Union{Cauchy, Gumbel, Laplace, Logistic, Normal}
#ThreeParamReal = Union{NormalInverseGaussian}#Skip
#TwoParamProb = Union{Beta}
OneParamPositive = Union{Chi, Chisq, Exponential, Rayleigh}
OneParamPositiveScale = Union{InverseGaussian, Levy, LogNormal}
TwoParamPositive = Union{BetaPrime, FDist, Frechet, Gamma, InverseGamma, Pareto, Weibull}
#Do GeneralizedExtremeValue seperately, because it needs 3 params?
#Do Arcsine, SymTriangularDist, TriangularDist, Uniform seperately

RealDist = Union{OneParamReal, OneParamRealScale}
PositiveDist = Union{OneParamPositive, OneParamPositiveScale, TwoParamPositive}

OneParam = Union{OneParamReal, OneParamRealScale, OneParamPositive, OneParamPositiveScale}
TwoParam = Union{TwoParamPositive, Beta}

function polynomial_interpolation(value_nodes::Array{Float64,2}, weight_nodes::Vector{Float64}, d::Type{<:ContinuousUnivariateDistribution}, n::Int)
  diag = Vector{Float64}(n)
  off_diag = Vector{Float64}(n-1)
  diag[1] = diag[end] = 3.0
  np1 = n + 1
  np1h2 = ( np1 / 2 ) ^ 2
  β2 = 1 / ( np1h2 - n )
  β1 = -np1*β2
  β0 = 2 + np1h2*β2
  for i ∈ 2:cld(n,2)
    diag[i] = diag[np1-i] = β0 + β1*i + β2*i^2
    off_diag[i-1] = off_diag[np1-i] = √(diag[i] * diag[i-1]) / 2
  end
  if rem(n, 2) == 0
    i = cld(n,2)
    off_diag[i] = √(diag[i] * diag[i-1]) / 2
  end
  polynomial_interpolation{d}(value_nodes, weight_nodes, n, diag, off_diag)
end

function polynomial_interpolation(range::StepRangeLen, itp::Interpolations.GriddedInterpolation, d::Type{<:RealDist}, n::Int = length(range))
  weight_nodes = Vector{Float64}(n)
  value_nodes = ones(length(range),4)
  @inbounds for (i,v) ∈ enumerate(range)
    weight_nodes[i] = itp[v]
    value_nodes[i,2] = v
    value_nodes[i,3] = v^2
    value_nodes[i,4] = v^3
  end
  polynomial_interpolation(value_nodes, weight_nodes, d, n)
end
function polynomial_interpolation(range::StepRangeLen, itp::Interpolations.GriddedInterpolation, d::Type{<:PositiveDist}, n::Int = length(range))
  weight_nodes = Vector{Float64}(n)
  value_nodes = ones(length(range),4)
  @inbounds for (i,v) ∈ enumerate(range)
    weight_nodes[i] = itp[v]
    value_nodes[i,2] = log(v)
    value_nodes[i,3] = value_nodes[i,2]^2
    value_nodes[i,4] = value_nodes[i,2]^3
  end
  polynomial_interpolation(value_nodes, weight_nodes, d, n)
end
function polynomial_interpolation(range::StepRangeLen, itp::Interpolations.GriddedInterpolation, d::Type{Beta}, n::Int = length(range))
  weight_nodes = Vector{Float64}(n)
  value_nodes = ones(length(range),4)
  @inbounds for (i,v) ∈ enumerate(range)
    weight_nodes[i] = itp[v]
    value_nodes[i,2] = LogDensities.logit(v)
    value_nodes[i,3] = value_nodes[i,2]^2
    value_nodes[i,4] = value_nodes[i,2]^3
  end
  polynomial_interpolation(value_nodes, weight_nodes, d, n)
end

function polynomial_interpolation(itp::Interpolations.GriddedInterpolation, d::Type{<:ContinuousUnivariateDistribution} = Normal, n::Int = 100)
    polynomial_interpolation(linspace(itp.knots[1][1], itp.knots[1][end], n), itp, d, n)
end
Base.length(::polynomial_interpolation{<: OneParam}) = 7
Base.length(::polynomial_interpolation{<: TwoParam}) = 8

function update_β!(β::Vector{<:Real}, Θ::Vector{<:Real})
  β[1] = Θ[1]
  β[2] = exp(Θ[2])
  β[4] = exp(Θ[4])
  β[3] = √(3β[2]*β[4])*(2/(1+exp(-Θ[3]))-1)
end
function construct_β(Θ::Vector{T}) where {T<:Real}
  β = Vector{T}(4)
  β[1] = Θ[1]
  β[2] = exp(Θ[2])
  β[4] = exp(Θ[4])
  β[3] = √(3β[2]*β[4])*(2/(1+exp(-Θ[3]))-1)
  β
end
function cdf_quantile(β::Vector{<:Real}, x::Array{Float64,2}, d::ContinuousUnivariateDistribution)
  Distributions.cdf(d, x * β)
end

function cdf_error(Θ::Vector{<:Real}, poly::polynomial_interpolation)
  δ = predict_cdf(Θ, poly) .- poly.y
  ρ = LogDensities.logistic(Θ[end-1])
  det_im1 = poly.diag[1]
  det_im2 = 1.0
  out = poly.diag[1] * δ[1]^2
  for i ∈ 2:poly.n
    out += δ[i]^2*poly.diag[i] - 2δ[i-1]*δ[i]*poly.off_diag[i-1]*ρ
    det_i = poly.diag[i]*det_im1 - ρ*poly.off_diag[i-1]*det_im2
    det_im2 = det_im1
    det_im1 = det_i
  end
  out*exp(-Θ[end]) + poly.n*Θ[end] - log(det_im1)
end


function predict_cdf(Θ::Vector{<:Real}, poly::polynomial_interpolation{T}) where {T <: OneParamReal}
  Distributions.cdf(T(exp(Θ[5])), poly.X * construct_β(Θ))
end
function predict_cdf(Θ::Vector{<:Real}, poly::polynomial_interpolation{T}) where {T <: OneParamRealScale}
  Distributions.cdf(T(0, exp(Θ[5])), poly.X * construct_β(Θ))
end
function predict_cdf(Θ::Vector{<:Real}, poly::polynomial_interpolation{T}) where {T <: OneParamPositive}
  Distributions.cdf(T(exp(Θ[5])), exp.(poly.X * construct_β(Θ)))
end
function predict_cdf(Θ::Vector{<:Real}, poly::polynomial_interpolation{T}) where {T <: OneParamPositiveScale}
  Distributions.cdf(T(0, exp(Θ[5])), exp.(poly.X * construct_β(Θ)))
end
function predict_cdf(Θ::Vector{<:Real}, poly::polynomial_interpolation{T}) where {T <: TwoParamPositive}
  Distributions.cdf(T(exp(Θ[5]), exp(Θ[6])), exp.(poly.X * construct_β(Θ)))
end
function predict_cdf(Θ::Vector{<:Real}, poly::polynomial_interpolation{Beta})
  Distributions.cdf(Beta(exp(Θ[5]), exp(Θ[6])), LogDensities.logistic.(poly.X * construct_β(Θ)))
end


struct GLM{T <: ContinuousUnivariateDistribution} <: InterpolateIntegral
  β::Vector{Float64}
  d::T
end
function GLM(Θ::Vector{<:Real}, d::Type{T}) where {T <: OneParamReal}
  GLM(construct_β(Θ), T(exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, d::Type{T}) where {T <: OneParamRealScale}
  GLM(construct_β(Θ), T(0, exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, d::Type{T}) where {T <: OneParamPositive}
  GLM(construct_β(Θ), T(exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, d::Type{T}) where {T <: OneParamPositiveScale}
  GLM(construct_β(Θ), T(0, exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, d::Type{T}) where {T <: TwoParamPositive}
  GLM(construct_β(Θ), T(exp(Θ[5]), exp(Θ[6])))
end
function GLM(Θ::Vector{<:Real}, d::Type{Beta})
  GLM(construct_β(Θ), Beta(exp(Θ[5]), exp(Θ[6])))
end

function GLM(itp::Interpolations.GriddedInterpolation, d::Type{<:ContinuousUnivariateDistribution})
  poly = polynomial_interpolation(itp, d)
  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(x -> cdf_error(x, poly), zeros(length(poly)), autodiff = :forward), method = NewtonTrustRegion()))#LBFGS
  GLM(Θ_hat, d)
end

function GLM(wv::weights_values, d::Type{<:ContinuousUnivariateDistribution})
  GLM(interpolate_weight_values(wv), d)
end

function GLM(wv::weights_values)
  itp = interpolate_weight_values(wv)
  if min(itp.knots[1][1], itp.knots[1][1]) < 0
    return GLM(itp, Normal)
  elseif max(itp.knots[1][end], itp.knots[1][end]) > 1
    return GLM(itp, Gamma)
  else
    return GLM(itp, Beta)
  end
end
function poly(β::Vector, x::Real)
  @inbounds β[1] + β[2]*x + β[3]*x^2 + β[4]*x^3
end
function Distributions.cdf(itp::GLM{T}, x::Real) where {T <: RealDist}
  Distributions.cdf(itp.d, poly(itp.β, x))
end
function Distributions.cdf(itp::GLM{T}, x::Real) where {T <: PositiveDist}
  Distributions.cdf(itp.d, exp(poly(itp.β, log(x))))
end
function Distributions.cdf(itp::GLM{Beta}, x::Real)
  Distributions.cdf(itp.d, LogDensities.logistic(poly(itp.β, LogDensities.logit(x))))
end

function find_root(β::Vector{Float64}, δ::Float64)
  a = β[4]
  b = β[3]
  c = β[2]
  d = β[1] - δ
  Δ0 = b^2 - 3a*c
  Δ1 = 2b^3 - 9a*b*c + 27a^2*d
  Δreal = Δ1^2 - 4Δ0^3
  if Δreal > 0
    Δ1prΔreal = (Δ1 + √Δreal)/2
    if Δ1prΔreal > 0
      Creal = cbrt(Δ1prΔreal)
      return - (b + Creal + Δ0/Creal)/3a
    else
      C = complex(Δ1prΔreal)^(1/3)
    end
  else
    rΔ = sqrt(complex(Δreal))
    C = complex((Δ1 + rΔ)/2)^(1/3)
  end

  ξ = -0.5 + √3/2im
  x = Vector{Complex{Float64}}(3)
  x[1] = - (b + C + Δ0/C)/3a
  for i ∈ 2:3
    C *= ξ
    x[i] = - (b + C + Δ0/C)/3a
  end
  real_roots = imag(x) .≈ 0
  if sum(real_roots) == 1
    return real(x[real_roots][1])
  elseif sum(real_roots) > 1
    rrs = real.(x[real_roots])
    mrrs = mean(rrs)
    if std(rrs) / mrrs > 1e-10
      warn("Multiple real roots in quantile estimation; returning average.")
    end
    return mrrs
  else
    throw("No real roots found.")
  end
end

function Base.quantile(itp::GLM{T}, x::Real) where {T <: RealDist}
  if x <= 0
    return - Inf
  elseif x >= 1
    return Inf
  end
  find_root(itp.β, Base.quantile(itp.d, x))
end
function Base.quantile(itp::GLM{T}, x::Real) where {T <: PositiveDist}
  if x <= 0
    return 0.0
  elseif x >= 1
    return Inf
  end
  exp(find_root(itp.β, log(Base.quantile(itp.d, x))))
end
function Base.quantile(itp::GLM{Beta}, x::Real)
  if x <= 0
    return 0.0
  elseif x >= 1
    return 1.0
  end
  logistic(find_root(itp.β, LogDensities.logit(Base.quantile(itp.d, x))))
end


struct Grid <: InterpolateIntegral
  weights::Array{Float64,1}
  values::Array{Float64,1}
end
function Grid(wv::weights_values)
  itp = interpolate_weight_values(wv)
  value_nodes = [linspace(itp.knots[1][1], itp.knots[1][end], 100)...]
  weight_nodes = zeros(100)
  weight_nodes[end] = 1
  for i ∈ 2:99
    weight_nodes[i] = itp[value_nodes[i]]
  end
  Grid(weight_nodes, value_nodes)
end
function Distributions.cdf(itp::Grid, x::Real)
  if x < itp.values[1]
    return 0.0
  elseif x > itp.values[end]
    return 1.0
  else
    return grid_interp(itp.values, itp.weights, searchsortedfirst(itp.values, x), x)
  end
end
function Base.quantile(itp::Grid, x::Real)
  if x <= 0
    return - Inf
  elseif x >= 1
    return Inf
  elseif x < 0.5
    i = searchsortedfirst(itp.weights, x)
  else
    i = searchsortedlast(itp.weights, x) + 1
  end
  grid_interp(itp.weights, itp.values, i, x)
end
function grid_interp(x::Array{Float64,1}, y::Array{Float64,1}, i::Int, z::Real)
  y[i-1] + (z - x[i-1]) * (y[i] - y[i-1])/(x[i]-x[i-1])
end
