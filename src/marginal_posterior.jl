

struct marginal{Ω, T <: ApproxCDF}
  wv::Ω
  μ::Float64
  σ::Float64
  itp::T
end
function update_MarginalBuffer!(jp::JointPosterior{P}, f::Function) where P
    m = get!(() -> MarginalBuffer(length(jp.density)), jp.M.MarginalBuffers, f )
    @inbounds for (i,j) ∈ enumerate(jp.Θ)
        m.v[i] = f(j.Θ...)
    end
    Vandermonde!(m, jp.density, calc_μ_σ(m.v, jp.density)...)
end

function Vandermonde(m::MarginalBuffer, density::Vector{Float64})
    sortperm!(m.ind, m.v)
    cumulative_w = 0.0
    μ = 0.0
    EX² = 0.0
    k = 2
    @inbounds for (i, j) ∈ enumerate(m.ind)
        dᵢ = density[j]
        cumulative_w += dᵢ
        m.w[i] = cumulative_w
        vᵢ = m.v[j]
        μ += vᵢ*dᵢ
        vᵢ² = vᵢ^2
        EX² += vᵢ²*dᵢ
        m.V[k] = vᵢ
        m.V[k+1] = vᵢ²  * exp(-2)
        m.V[k+2] = vᵢ^3 * exp(-4)
        m.V[k+3] = vᵢ^4 * exp(-6)
        m.V[k+4] = vᵢ^5 * exp(-8)
        m.V[k+5] = vᵢ^6 * exp(-10)
        m.V[k+6] = vᵢ^7 * exp(-12)
        m.V[k+7] = vᵢ^8 * exp(-14)
        m.V[k+8] = vᵢ^9 * exp(-16)
        k += 10
    end
    m, μ, sqrt(EX² - μ^2)
end
function Vandermonde!(m::MarginalBuffer, density::Vector{T}, μ::Number, σ::Number) where T <:Number
    sortperm!(m.ind, m.v)
    cumulative_w = zero(T)
    k = 2
    @inbounds for (i, j) ∈ enumerate(m.ind)
        dᵢ = density[j]
        cumulative_w += dᵢ
        m.w[i] = cumulative_w
        vᵢ = (m.v[j] - μ)/σ
        vᵢ² = vᵢ^2
        vᵢ4 = vᵢ²^2
        m.V[k] = vᵢ
        m.V[k+1] = vᵢ²
        m.V[k+2] = vᵢ*vᵢ²
        m.V[k+3] = vᵢ4
        m.V[k+4] = vᵢ4*vᵢ
        m.V[k+5] = vᵢ4*vᵢ²
        m.V[k+6] = vᵢ4*vᵢ²*vᵢ
        m.V[k+7] = vᵢ4^2
        m.V[k+8] = vᵢ4^2*vᵢ
        k += 10
    end
    m, μ, σ
end
@inline update_MarginalBuffer!(jp::JointPosteriorRaw, f::Function) = update_MarginalBuffer!(jp.M, jp.grid.cache, jp.grid.density, f::Function)
@inline update_MarginalBuffer!(M::Model, cache::MatrixVecSVec, density::Vector, f::Function ) = update_MarginalBuffer!(M, cache.M, density, f::Function)
function update_MarginalBuffer!(M::Model, cache::Matrix, density::Vector, f::Function)
    m = get!(() -> MarginalBuffer(length(density)), M.MarginalBuffers, f )
    svf = SlidingVecFun(M.Θ, f, cache)
    @inbounds for i ∈ eachindex(m.v)
        m.v[i] = svf()
    end
    Vandermonde!(m, density, calc_μ_σ(m.v, density)...)
end

function calc_μ_σ(x::AbstractVector{T}, w::AbstractVector{T}) where T
  μ = dot(x, w)
  EX² = zero(T)
  @inbounds for i ∈ eachindex(x)
    EX² += x[i]^2 * w[i]
  end
  μ, √(EX² - μ^2)
end

function SlidingVecFun( Θ::ModelParam, f::Function, M::Matrix )
    g = function ()
        update!(Θ)
        f(Θ.Θ...)
    end
    svf = SlidingVecFun(g, Θ.v)
    set!(svf, M)
    svf
end

function weights_values(jp::JointPosterior{P}, f::Function) where {P}
    values = similar(jp.density)
    weights = similar(jp.density)
    @inbounds for (i,j) in enumerate(jp.Θ)
        values[i] = f(j.Θ...)
    end
    weights_values(copy(jp.density), values)
end
@inline weights_values(jp::JointPosteriorRaw, f::Function) = weights_values(jp.M.Θ, jp.grid.cache, copy(jp.grid.density), f::Function)
@inline weights_values(Θ::ModelParam, cache::MatrixVecSVec, density::Vector, f::Function ) = weights_values(Θ, cache.M, density, f::Function)
function weights_values(Θ::ModelParam, cache::Matrix, density::Vector, f::Function)
  values = similar(density) #May add dict to model that will allow for saving these.
  svf = SlidingVecFun(Θ, f, cache)
  @inbounds for i ∈ eachindex(values)
    values[i] = svf()
  end
  weights_values(density, values)
end

@inline marginal(jp::JointDist, f::Function, ::Type{Grid}) = marginal(jp::JointDist, f::Function)
function marginal(jp::JointDist, f::Function)
  wv = weights_values(jp, f)
  μ = dot(wv.weights, wv.values)
  σ = sqrt(dot(wv.weights, wv.values .^ 2) - μ^2)
  marginal(wv, μ, σ, Grid(wv))
end
function marginal(jp::JointDist, f::Function, ::Type{T}) where {T <: ContinuousUnivariateDistribution}
    marginal(update_MarginalBuffer!(jp, f)..., T)
end
function marginal(m::MarginalBuffer, μ::Real, σ::Real, ::Type{Normal})
    marginal(m, μ, σ, NestedPolyGLM(m, Normal(μ, σ)))
end
function marginal(jp::MarginalBuffer, μ::Real, σ::Real, ::Type{Gamma})
  throw("Currently unsupported.")
  marginal(m, μ, σ, NestedPolyGLM(m, Gamma((μ/σ)^2, σ^2/μ)))
end


function marginal(wv::weights_values, μ::Float64, σ::Float64, ::Type{Grid})
  marginal(wv, μ, σ, Grid(wv))
end

function Distributions.cdf(m::marginal, x)
  Distributions.cdf(m.itp, x)
end
function Distributions.pdf(m::marginal, x)
  Distributions.pdf(m.itp, x)
end
function Base.quantile(m::marginal, x)
  Base.quantile(m.itp, x)
end

function Base.show(io::IO, m::marginal)
  println("Marginal parameter")
  println("μ: ", m.μ)
  println("σ: ", m.σ)
  println("Quantiles: ", quantile.(m, [.025 .25 .5 .75 .975]))
end
