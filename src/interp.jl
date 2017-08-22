abstract type InterpolateIntegral end

struct weights_values <: AbstractArray{Float64,1}
  weights::Vector{Float64}
  values::Vector{Float64}
end
struct polynomial_interpolation{d <: ContinuousUnivariateDistribution}
  X::Array{Float64, 2}
  y::Vector{Float64}
  n::Int
  diag::Vector{Float64}
  off_diag::Vector{Float64}
end
struct Grid <: InterpolateIntegral
  weights::Array{Float64,1}
  values::Array{Float64,1}
end
struct GLM{T <: ContinuousUnivariateDistribution} <: InterpolateIntegral
  β::Vector{Float64}
  d::T
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

function polynomial_interpolation(value_nodes::Array{Float64,2}, weight_nodes::Vector{Float64}, ::Type{d}, n::Int) where {d <: ContinuousUnivariateDistribution}
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
  value_nodes = ones(n,4)
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
  value_nodes = ones(n,4)
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
  value_nodes = ones(n,4)
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
  @inbounds begin
    β[1] = Θ[1]
    β[2] = exp(Θ[2])
    β[4] = exp(Θ[4])
    β[3] = √(3β[2]*β[4])*(2/(1+exp(-Θ[3]))-1)
  end
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



function GLM(Θ::Vector{<:Real}, ::Type{T}) where {T <: OneParamReal}
  GLM(construct_β(Θ), T(exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, ::Type{T}) where {T <: OneParamRealScale}
  GLM(construct_β(Θ), T(0, exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, ::Type{T}) where {T <: OneParamPositive}
  GLM(construct_β(Θ), T(exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, ::Type{T}) where {T <: OneParamPositiveScale}
  GLM(construct_β(Θ), T(0, exp(Θ[5])))
end
function GLM(Θ::Vector{<:Real}, ::Type{T}) where {T <: TwoParamPositive}
  GLM(construct_β(Θ), T(exp(Θ[5]), exp(Θ[6])))
end
function GLM(Θ::Vector{<:Real}, ::Type{Beta})
  GLM(construct_β(Θ), Beta(exp(Θ[5]), exp(Θ[6])))
end

function GLM(itp::Interpolations.GriddedInterpolation, ::Type{d}) where {d <: ContinuousUnivariateDistribution}
  poly = polynomial_interpolation(itp, d)
  cdf_err(x::Vector) = cdf_error(x, poly)
  Θ_hat::Vector{Float64} = Optim.minimizer(optimize(TwiceDifferentiable(cdf_err, zeros(length(poly)), autodiff = :forward), method = NewtonTrustRegion()))#LBFGS
  GLM(Θ_hat, d)
end

function GLM(wv::weights_values, ::Type{d}) where {d <: ContinuousUnivariateDistribution}
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
  @inbounds out = β[1] + β[2]*x + β[3]*x^2 + β[4]*x^3
  out
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

function calculate_common!(x, m, d)
    if x != m.ϕ
        copy!(m.ϕ, x)
        update_αβ!(m)
        At_mul_B!(m.Vβ, m.V, m.β)
#        VFβ .= cdf.(d, m.Vβ)
#        δ .= VFβ .- w
        m.δ .= cdf.(d, m.Vβ) .- m.w
        ToeplitzSymTriQuadForm!(m)

    end
end
function l_likelihood(x, m, d)
    calculate_common!(x, m, d)
    m.Q[1] - log_det_tstd(m.Λ[2]) + n*m.ϕ[8]
end
function mul_tstd_x!(z, ρ, x)

    lag, current, next = 0.0, x[1], x[2]
    z[1] = current - ρ * next
    @inbounds for i ∈ 3:length(x)
        lag, current, next = current, next, x[i]
        z[i-1] = current - ρ * (lag + next)
    end
    z[end] = next - ρ * current

end

x = randn(10); μ = 0.1 .* randn(10);
function fst(x, μ)
  δ = ( 1 .+ erf.( x ./ √2 ) ) ./ 2 .- μ
  ToeplitzSymTriQuadForm(δ, 0.25, 1.0)
end


δ = ( 1 .+ erf.( x ./ √2 ) ) ./ 2 .- μ


function nscore!(stor, x, m, d)
    calculate_common!(x, m, d)

    ρ = m.Λ[2]

    Vfβ.diag .= pdf.(d, m.Vβ)
    mul_tstd_x!(m.nbuff1, ρ, m.δ)
    A_mul_B!(m.nbuff2, m.Vfβ, m.buff1)
    A_mul_B!(m.∇β, m.V, m.buff2)
    m.∇β .*= 2/m.Λ[1]
    A_mul_B!(m.∇α, m.α, m.m.∇β)

    epsilon = eps()
    for i, grad ∈ enumerate(m.∇α)
      stor[i] = grad
    end
    stor[8] = m.Q[2] + m.n
    stor[9] = m.Q[3] - imag(log_det_tstd(ρ+im*epsilon, m.n)) / epsilon
    nlj_grad!(score::AbstractVector, m::MarginalBuffer)
end

function ToeplitzSymTriQuadForm(δ::Vector{T}, ρ::Number, σ²::Number) where T
    δ²_sum = zero(T)
    δcross_sum = zero(T)
    δ_lag = zero(T)
    @simd for δᵢ ∈ δ
        δ²_sum += δᵢ^2
        δcross_sum += δᵢ * δ_lag
        δ_lag = δᵢ
    end
    (δ²_sum - 2ρ*δcross_sum) / 2σ²
end
function ToeplitzSymTriQuadForm!(m::MarginalBuffer)
    δ²_sum = 0.0
    δcross_sum = 0.0
    δ_lag = 0.0
    @simd for δᵢ ∈ m.δ
        δ²_sum += δᵢ^2
        δcross_sum += δᵢ * δ_lag
        δ_lag = δᵢ
    end
    σ² = m.Λ[1]
    out = (δ²_sum - 2m.Λ[2]*δcross_sum) / σ²
    m.Q[1] = out
    m.Q[2] = - out / σ² #derivative with respect to σ²
    m.Q[3] = - 2δcross_sum / σ² #derivative with respect to ρ
    out
end


function l_likelihood(x, m, d)
    calculate_common!(x, m, d)
    m.Q[1] - log_det_tstd(m.Λ[2]) + n*m.ϕ[8]
end

function log_det_tstd(ρ::T, n::Int) where T
    lag1 = out = one(T)
    lag2 = zero(T)
    ρ² = ρ^2
    for i ∈ 1:n
        out -= ρ² * lag2
        lag2, lag1 = lag1, out
    end
    log(out)
end




function update_αβ!(m::MarginalBuffer)
    a = exp(m.ϕ[1])
    c = exp(m.ϕ[2])
    eϕ₃ = exp(m.ϕ[3])
    b = √(3a*c)*(eϕ₃ - 1)/(eϕ₃ + 1)
    m = exp(m.ϕ[5])
    eϕ₆ = exp(m.ϕ[6])
    l = √(3m)*(eϕ₆ - 1)/(eϕ₆ + 1)
    m.Λ[1] = exp(m.ϕ[8])
    eϕ₉ = exp(m.ϕ[9])
    m.Λ[2] = 0.5eϕ₉/(1+eϕ₉)
    d = m.ϕ[4]
    n = m.ϕ[7]
    l² = l^2
    l³ = l*l²
    m² = m^2
    m³ = m*m²
    n² = n^2
    n³ = n*n²
    bmn = b*m*n; aln² = a*l*n²; amn = a*m*n; bln = b*l*n; alm = a*l*m; blm = b*l*m
    al²m = alm*l
    alm² = a*l*m²; aln = a*l*n
    m.β[1] = a*n³ + b*n² + c*n + d
    m.β[2] = 3amn*n + 2b*m*n + c*m
    m.β[3] = 3aln² + 3amn*m + 2bln + b*m² + c*l + c
    m.β[4] = 6alm*n + a*m³ + 3a*n² + 2blm + 2b*n
    m.β[5] = 3aln*l + 3alm*m + 6amn + b*l² + 2b*m
    m.β[6] = 3alm*l + 6aln + 3a*m² + 2b*l
    m.β[7] = a*l³ + 6alm + 3a*n + b
    m.β[8] = 3a*l² + 3a*m
    m.β[9] = 3a*l
    m.β[10] = a
    bt = √(12a*c)*eϕ₃/(eϕ₃+1)^2
    lt = √(12m)*eϕ₆/(eϕ₆+1)^2
    m.α[1] = a*n³ + b*n²/2
    m.α[2] = b*n²/2 + n*c
    m.α[3] = n²*bt
    m.α[4] = 1
    m.α[7] = 3n²*a + 2n*b + c
    m.α[8] = 3amn*n + bmn
    m.α[9] = bmn + c*m
    m.α[10] = 2m*n*bt
    m.α[12] = 3amn*n + 2bmn + c*m
    m.α[14] = 6amn +2b*m
    m.α[15] = 3aln² + 3amn*m + bln + b*m²/2
    m.α[16] = bln + b*m²/2 + c*l + c
    m.α[17] = (2l*n + m²)*bt
    m.α[19] = 3aln²/2 + bln + c*l/2 + m*(6amn + 2b*m)
    m.α[20] = lt*(3a*n² + 2b*n + c)
    m.α[21] = 6a*l*n + 3a*m² + 2b*l
    m.α[22] = 6l*amn + a*m³ + 3a*n² + blm + b*n
    m.α[23] = blm + b*n
    m.α[24] = (2l*m + 2n)*bt
    m.α[26] = 9l*amn + 3a*m³ + 3blm
    m.α[27] = lt*(6a*m*n + 2b*m)
    m.α[28] = 6a*l*m + 6a*n + 2b
    m.α[29] = 3aln*l + 3alm² + 6amn + b*l²/2 + b*m
    m.α[30] = b*l²/2 + b*m
    m.α[31] = (l² + 2m)*bt
    m.α[33] = 3aln*l + 3alm²/2 + b*l² + 6alm² + 6amn + 2b*m
    m.α[34] = lt*(6aln + 3a*m² + 2b*l)
    m.α[35] = 3a*l² + 6a*m
    m.α[36] = 3alm*l + 6aln + 3a*m² + b*l
    m.α[37] = b*l
    m.α[38] = 2l*bt
    m.α[40] = 6alm*l + 3aln + b*l + 6a*m²
    m.α[41] = (6alm + 6a*n + 2b)*lt
    m.α[42] = 6a*l
    m.α[43] = a*l³ + 6alm + 3a*n + b/2
    m.α[44] = b/2
    m.α[45] = bt
    m.α[47] = l/2*(3a*l² + 6a*m) + m*6a*l
    m.α[48] = lt*(3a*l² + 6a*m)
    m.α[49] = 3a
    m.α[50] = a*(3l² + 3m)
    m.α[54] = 3a*l² + 3a*m
    m.α[55] = 6a*l*lt
    m.α[57] = 3a*l
    m.α[61] = 3a*l/2
    m.α[62] = lt*3a
    m.α[64] = a
end


function log_jacobian(m::MarginalBuffer)
  3(m.ϕ[1]+m.ϕ[2]+m.ϕ[5])/2 - nlogit_lj(m.ϕ[3]) - nlogit_lj(m.ϕ[6]) + m.ϕ[8]-nlogit_lj(m.ϕ[9])
end
function nlj_grad!(score::AbstractVector, m::MarginalBuffer)
  score[1] -= 1.5
  score[2] -= 1.5
  score[3] -= dnlogit_lj(m.ϕ[3])
  score[5] -= 1.5
  score[6] -= dnlogit_lj(m.ϕ[6])
  score[8] -= 1.0
  score[9] -= dnlogit_lj(m.ϕ[9])
end
function nlogit_lj(l, u, x)
  eˣ = exp(x)
  log(2 + eˣ + 1/eˣ) - log(u - l)
end
function nlogit_lj(b, x)
  eˣ = exp(x)
  log(2 + eˣ + 1/eˣ) - log(b)
end
function nlogit_lj(x)
  eˣ = exp(x)
  log(2 + eˣ + 1/eˣ)
end
function dnlogit_lj(x)
  eˣ = exp(x)
  e²ˣ = eˣ^2
  (1 - e²ˣ) / (2eˣ + e²ˣ + 1)
end

function GLM(m::MarginalBuffer, d::ContinuousUnivariateDistribution)

end


function one_cubic_root(a::Real, c::Real, b::Real, d::Real)
    Δₒ = b^2 - 3a*c
    Δ₁ = 2b^3 - 9a*b*c + 27a^2*d
    Δ₂ = Δ₁^2 - 4Δ₀^3
    C = cbrt( (Δ₁ + sqrt(Δ₁^2 - 4Δ₀^3)) / 2 )
    -(b + C + Δ₀ / C) / 3a
end
function one_cubic_root(c::Real, b::Real, d::Real)
    Δₒ = b^2 - 3c
    Δ₁ = 2b^3 - 9b*c + 27d
    Δ₂ = Δ₁^2 - 4Δ₀^3
    C = cbrt( (Δ₁ + sqrt(Δ₁^2 - 4Δ₀^3)) / 2 )
    -(b + C + Δ₀ / C) / 3
end
function find_root(β::Vector{Float64}, δ::Float64)
    r1 = one_cubic_root(β[1], β[2], β[3], β[4] - δ)
    one_cubic_root(β[5], β[6], β[7] - r1)
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
