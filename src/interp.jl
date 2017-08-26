abstract type ApproxCDF end
abstract type InterpolateCDF <: ApproxCDF end
abstract type SmoothCDF <: ApproxCDF end

struct weights_values <: AbstractArray{Float64,1}
  weights::Vector{Float64}
  values::Vector{Float64}
end
struct Grid <: InterpolateCDF
  weights::Array{Float64,1}
  values::Array{Float64,1}
end
struct NestedPolyGLM{D <: ContinuousUnivariateDistribution} <: SmoothCDF
  β::Vector{Float64}
  θ::Vector{Float64}
  d::D
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


# The problem has already been centered and scaled.
function unconstrained_cdf(d::Distributions.Normal, x::T) where T <: Number
  (1 + erf(x/√2)) / 2
end
function unconstrained_pdf(d::Distributions.Normal, x::T) where T <: Number
  exp(-x^2/2)*(2π)^-0.5
end

function calculate_common!(x, m, d)
    if x != m.ϕ
        copy!(m.ϕ, x)
        update_αβ!(m)
        At_mul_B!(m.Vβ, m.V, m.β)
        m.δ .= unconstrained_cdf.(d, m.Vβ) .- m.w
        ToeplitzSymTriQuadForm!(m)
    end
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




function ntscore!(stor, x, m, d)
    calculate_common!(x, m, d)

    ρ = m.Λ[2]

    mul_tstd_x!(m.nbuff1, ρ, m.δ)
    m.Vfβ .= unconstrained_pdf.(d, m.Vβ) .* m.nbuff1
    A_mul_B!(m.∇β, m.V, m.Vfβ)
    @inbounds for i ∈ 1:10
      m.∇β[i] = 2m.∇β[i] / m.Λ[1]#+ 10m.β[i]
    end
  #  m.∇β[10] -= 10
    A_mul_B!(m.∇α, m.α, m.∇β)

    for (i, grad) ∈ enumerate(m.∇α)
      stor[i] = grad
    end
    stor[1] += 2m.ϕ[1]
    stor[8] = m.Q[2] + m.n
    epsilon = eps()
    stor[9] = (m.Q[3] - imag(log_det_tstd(ρ+im*epsilon, m.n)) / epsilon) * logit_lj(m.ϕ[9])/4
    nlj_grad!(stor, m)
    stor ./= m.n
end

function ntl_likelihood!(x, m, d)
    calculate_common!(x, m, d)
    ( m.Q[1] - log_det_tstd(m.Λ[2], m.n) - 2log_jacobian(m) + abs2(m.ϕ[1])  ) / m.n + m.ϕ[8] #5dot(m.β,m.β) - 10m.β[10]
end
#+ dot(m.β, m.β) - 2m.β[10]

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
    m.Q[2] = - out #derivative with respect to m.ϕ[8] = log(σ²)
    m.Q[3] = - 2δcross_sum / σ² #derivative with respect to ρ
    out
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





function update_βθ!(mb::MarginalBuffer)
    a = exp(mb.ϕ[1])
    c = exp(mb.ϕ[2])
    eϕ₃ = exp(mb.ϕ[3])
    b = √(3a*c)*(eϕ₃ - 1)/(eϕ₃ + 1)
    m = exp(mb.ϕ[5])
    eϕ₆ = exp(mb.ϕ[6])
    l = √(3m)*(eϕ₆ - 1)/(eϕ₆ + 1)
    mb.Λ[1] = exp(mb.ϕ[8])
    eϕ₉ = exp(mb.ϕ[9])
    mb.Λ[2] = 0.25eϕ₉/(1+eϕ₉)
    d = mb.ϕ[4]
    n = mb.ϕ[7]
    l² = l^2
    l³ = l*l²
    m² = m^2
    m³ = m*m²
    n² = n^2
    n³ = n*n²
    bmn = b*m*n; aln² = a*l*n²; amn = a*m*n; bln = b*l*n; alm = a*l*m; blm = b*l*m
    al²m = alm*l
    alm² = a*l*m²; aln = a*l*n
    mb.β[1] = a*n³ + b*n² + c*n + d
    mb.β[2] = 3amn*n + 2b*m*n + c*m
    mb.β[3] = 3aln² + 3amn*m + 2bln + b*m² + c*l
    mb.β[4] = 6alm*n + a*m³ + 3a*n² + 2blm + 2b*n + c
    mb.β[5] = 3aln*l + 3alm*m + 6amn + b*l² + 2b*m
    mb.β[6] = 3alm*l + 6aln + 3a*m² + 2b*l
    mb.β[7] = a*l³ + 6alm + 3a*n + b
    mb.β[8] = 3a*l² + 3a*m
    mb.β[9] = 3a*l
    mb.θ[1] = mb.β[10] = a
    mb.θ[2] = c
    mb.θ[3] = b
    mb.θ[4] = d
    mb.θ[5] = m
    mb.θ[6] = l
    mb.θ[7] = n
end

function update_αβ!(mb::MarginalBuffer)
    a = exp(mb.ϕ[1])
    c = exp(mb.ϕ[2])
    eϕ₃ = exp(mb.ϕ[3])
    b = √(3a*c)*(eϕ₃ - 1)/(eϕ₃ + 1)
    m = exp(mb.ϕ[5])
    eϕ₆ = exp(mb.ϕ[6])
    l = √(3m)*(eϕ₆ - 1)/(eϕ₆ + 1)
    mb.Λ[1] = exp(mb.ϕ[8])
    mb.Λ[2] = 1/( 4 * (1+exp(-mb.ϕ[9])) )
    d = mb.ϕ[4]
    n = mb.ϕ[7]
    l² = l^2
    l³ = l*l²
    m² = m^2
    m³ = m*m²
    n² = n^2
    n³ = n*n²
    bmn = b*m*n; aln² = a*l*n²; amn = a*m*n; bln = b*l*n; alm = a*l*m; blm = b*l*m
    al²m = alm*l
    alm² = a*l*m²; aln = a*l*n
    mb.β[1] = a*n³ + b*n² + c*n + d
    mb.β[2] = 3amn*n + 2b*m*n + c*m
    mb.β[3] = 3aln² + 3amn*m + 2bln + b*m² + c*l
    mb.β[4] = 6alm*n + a*m³ + 3a*n² + 2blm + 2b*n + c
    mb.β[5] = 3aln*l + 3alm*m + 6amn + b*l² + 2b*m
    mb.β[6] = 3alm*l + 6aln + 3a*m² + 2b*l
    mb.β[7] = a*l³ + 6alm + 3a*n + b
    mb.β[8] = 3a*l² + 3a*m
    mb.β[9] = 3a*l
    mb.β[10] = a
    bt = √(12a*c)*eϕ₃/(eϕ₃+1)^2
    lt = √(12m)*eϕ₆/(eϕ₆+1)^2
    mb.α[1] = a*n³ + b*n²/2
    mb.α[2] = b*n²/2 + n*c
    mb.α[3] = n²*bt
    mb.α[4] = 1
    mb.α[7] = 3n²*a + 2n*b + c

    mb.α[8] = 3amn*n + bmn
    mb.α[9] = bmn + c*m
    mb.α[10] = 2m*n*bt
    mb.α[12] = 3amn*n + 2bmn + c*m
    mb.α[14] = 6amn +2b*m

    mb.α[15] = 3aln² + 3amn*m + bln + b*m²/2
    mb.α[16] = bln + b*m²/2 + c*l
    mb.α[17] = (2l*n + m²)*bt
    mb.α[19] = 3aln²/2 + bln + c*l/2 + m*(6amn + 2b*m)
    mb.α[20] = lt*(3a*n² + 2b*n + c)
    mb.α[21] = 6a*l*n + 3a*m² + 2b*l

    mb.α[22] = 6l*amn + a*m³ + 3a*n² + blm + b*n
    mb.α[23] = blm + b*n + c
    mb.α[24] = (2l*m + 2n)*bt
    mb.α[26] = 9l*amn + 3a*m³ + 3blm
    mb.α[27] = lt*(6a*m*n + 2b*m)
    mb.α[28] = 6a*l*m + 6a*n + 2b

    mb.α[29] = 3aln*l + 3alm² + 6amn + b*l²/2 + b*m
    mb.α[30] = b*l²/2 + b*m
    mb.α[31] = (l² + 2m)*bt
    mb.α[33] = 3aln*l + 3alm²/2 + b*l² + 6alm² + 6amn + 2b*m
    mb.α[34] = lt*(6aln + 3a*m² + 2b*l)
    mb.α[35] = 3a*l² + 6a*m
    mb.α[36] = 3alm*l + 6aln + 3a*m² + b*l
    mb.α[37] = b*l
    mb.α[38] = 2l*bt
    mb.α[40] = 6alm*l + 3aln + b*l + 6a*m²
    mb.α[41] = (6alm + 6a*n + 2b)*lt
    mb.α[42] = 6a*l
    mb.α[43] = a*l³ + 6alm + 3a*n + b/2
    mb.α[44] = b/2
    mb.α[45] = bt
    mb.α[47] = l/2*(3a*l² + 6a*m) + m*6a*l
    mb.α[48] = lt*(3a*l² + 6a*m)
    mb.α[49] = 3a
    mb.α[50] = a*(3l² + 3m)
    mb.α[54] = 3a*l² + 3a*m
    mb.α[55] = 6a*l*lt
    mb.α[57] = 3a*l
    mb.α[61] = 3a*l/2
    mb.α[62] = lt*3a
    mb.α[64] = a
end


function log_jacobian(m::MarginalBuffer)
  3(m.ϕ[1]+m.ϕ[2]+m.ϕ[5])/2 - nlogit_lj(m.ϕ[3]) - nlogit_lj(m.ϕ[6]) + m.ϕ[8]-nlogit_lj(m.ϕ[9])
end
function nlj_grad!(score::AbstractVector, m::MarginalBuffer)
  score[1] -= 3.0
  score[2] -= 3.0
  score[3] -= 2dnlogit_lj(m.ϕ[3])
  score[5] -= 3.0
  score[6] -= 2dnlogit_lj(m.ϕ[6])
  score[8] -= 2.0
  score[9] -= 2dnlogit_lj(m.ϕ[9])
  score
end
#See for cubic version.
function log_jacobian_C(m::MarginalBuffer)
  3(m.ϕ[1]+m.ϕ[2])/2 - nlogit_lj(m.ϕ[3]) + m.ϕ[5]-nlogit_lj(m.ϕ[6])
end
function nlj_grad_C!(score::AbstractVector, m::MarginalBuffer)
  score[1] -= 3.0
  score[2] -= 3.0
  score[3] -= 2dnlogit_lj(m.ϕ[3])
  score[5] -= 2.0
  score[6] -= 2dnlogit_lj(m.ϕ[9])
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
function logit_lj(x)
  eˣ = exp(x)
  1/(2 + eˣ + 1/eˣ)
end
function dnlogit_lj(x)
  eˣ = exp(x)
  e²ˣ = eˣ^2
  (1 - e²ˣ) / (2eˣ + e²ˣ + 1)
end



function polyexpreval(β::AbstractVector{S}, x::T) where T where S
  x_i = convert(promote_type(Float64, T), x)
  out = β[1] + x_i*β[2]
  @inbounds for i ∈ 3:length(β)
    x_i *= x
    out += x_i * β[i]
  end
  out
end

function ∂polyexpreval(β::AbstractVector{S}, x::T) where T where S
  fₓ = convert(promote_type(S, T), β[1])
  ∂fₓ = zero(promote_type(S, T))
  x_i = one(promote_type(S, T))
  @inbounds for i ∈ 1:length(β)-1
    βᵢ = β[i+1]
    ∂fₓ += x_i * i * βᵢ
    x_i *= x
    fₓ += x_i * βᵢ
  end
  fₓ, ∂fₓ
end



function Distributions.cdf(itp::NestedPolyGLM{Distributions.Normal{Float64}}, x::Real)
  unconstrained_cdf(itp.d, polyexpreval(itp.β, cbrt(cbrt((x-itp.d.μ)/itp.d.σ))))
end
function Base.quantile(itp::NestedPolyGLM{Distributions.Normal{Float64}}, x::Real)
  nested_root(itp.θ, √(2)*erfinv(2x-1))^9*itp.d.σ + itp.d.μ
end
function Distributions.pdf(itp::NestedPolyGLM{Distributions.Normal{Float64}}, x::Real)
  x¹₉ = cbrt(cbrt((x-itp.d.μ)/itp.d.σ))
  fₓ, ∂fₓ = ∂polyexpreval(itp.β, x¹₉)
  unconstrained_pdf(itp.d, fₓ) * ∂fₓ / (9x¹₉^8 * itp.d.σ)
end

function NestedPolyGLM(m::LogDensities.MarginalBuffer, d::ContinuousUnivariateDistribution)
  f = x -> ntl_likelihood!(x, m, d)
  g! = (stor, x) -> ntscore!(stor, x, m, d)
#  copy!(m.ϕ, Optim.minimizer(Optim.optimize(f, g!, m.init, NewtonTrustRegion())))
  copy!(m.init, Optim.minimizer(Optim.optimize(f, g!, m.init, BFGS(; linesearch = BackTracking()))))
  update_βθ!(m)
  NestedPolyGLM(m.β, m.θ, d)
end



function one_cubic_root(a::Real, c::Real, b::Real, d::Real)
    Δₒ = b^2 - 3a*c
    Δ₁ = 2b^3 - 9a*b*c + 27a^2*d
#    Δ₂ = Δ₁^2 - 4Δ₀^3
    C = cbrt( (Δ₁ + sqrt(Δ₁^2 - 4Δₒ^3)) / 2 )
    -(b + C + Δₒ / C) / 3a
end
function one_cubic_root(c::Real, b::Real, d::Real)
    Δₒ = b^2 - 3c
    Δ₁ = 2b^3 - 9b*c + 27d
#    Δ₂ = Δ₁^2 - 4Δ₀^3
    C = cbrt( (Δ₁ + sqrt(Δ₁^2 - 4Δₒ^3)) / 2 )
    -(b + C + Δₒ / C) / 3
end
function nested_root(β::Vector{Float64}, δ::Float64)
    r1 = one_cubic_root(β[1], β[2], β[3], β[4] - δ)
    one_cubic_root(β[5], β[6], β[7] - r1)
end
#Leaving this, because a brute force version could be useful.
#Although, I don't realistically see the case where I'd allow
#a cubic polynomial to have more than 1 root.
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
