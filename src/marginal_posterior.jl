

struct marginal{T <: InterpolateIntegral}
  wv::weights_values
  μ::Float64
  σ::Float64
  itp::T
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

function marginal(jp::JointDist, f::Function, ::Type{T} = Grid) where {T}
  wv = weights_values(jp, f)
  μ = dot(wv.weights, wv.values)
  σ = sqrt(dot(wv.weights, wv.values .^ 2) - μ^2)

  marginal(wv, μ, σ, T)

end
function marginal(wv::weights_values, μ::Float64, σ::Float64, ::Type{T}) where {T <: ContinuousUnivariateDistribution}
  marginal(wv, μ, σ, GLM(wv, T))
end
function marginal(wv::weights_values, μ::Float64, σ::Float64, ::Type{Grid})
  marginal(wv, μ, σ, Grid(wv))
end
function marginal(wv::weights_values, μ::Float64, σ::Float64, ::Type{GLM})
  marginal(wv, μ, σ, GLM(wv))
end


function Distributions.cdf(m::marginal, x)
  Distributions.cdf(m.itp, x)
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
