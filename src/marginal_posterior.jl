

struct marginal{T <: InterpolateIntegral}
  wv::weights_values
  μ::Float64
  σ::Float64
  itp::T
end

function weights_values{p, q, P}(jp::JointPosterior{p, q, P}, f::Function)
  values = similar(jp.grid.density) #You're likely to be interested in multiple marginals, hence no saving on allocations.
  @inbounds for i ∈ eachindex(values)
    values[i] = f(transform!(jp.M.Θ, ( @views jp.grid.nodes[:,i] ), jp.Θ_hat, jp.U, q))
  end
  weights_values(copy(jp.grid.density), values)
end

function marginal{T}(jp::JointPosterior, f::Function, ::Type{T} = Grid)
  wv = weights_values(jp, f)
  μ = dot(wv.weights, wv.values)
  σ = sqrt(dot(wv.weights, wv.values .^ 2) - μ^2)

  marginal(wv, μ, σ, T)

end
function marginal{T <: ContinuousUnivariateDistribution}(wv::weights_values, μ::Float64, σ::Float64, ::Type{T})
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
