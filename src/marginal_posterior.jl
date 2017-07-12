

struct marginal{T <: InterpolateIntegral}
  wv::weights_values
  μ::Float64
  σ::Float64
  itp::T
end


function weights_values{p, q, P}(jp::JointPosterior{p, q, P}, f::Function)
  values_positive = similar(jp.weights_positive)
  values_negative = similar(jp.weights_negative)
  for i ∈ eachindex(values_positive)
    values_positive[i] = f(transform!(jp.M.Θ, jp.M.Grid.nodes_positive[:,i], jp.Θ_hat, jp.L, q))
  end
  for i ∈ eachindex(values_negative)
    values_negative[i] = f(transform!(jp.M.Θ, jp.M.Grid.nodes_negative[:,i], jp.Θ_hat, jp.L, q))
  end
  weights_values(jp.weights_positive, jp.weights_negative, values_positive, values_negative)
end

function marginal{T}(jp::JointPosterior, f::Function, interpolate::Type{T} = Grid)
  wv = weights_values(jp, f)
  μ = dot(wv.weights_positive, wv.values_positive) + dot(wv.weights_negative, wv.values_negative)
  σ = sqrt(dot(wv.weights_positive, wv.values_positive .^ 2) + dot(wv.weights_negative, wv.values_negative .^ 2) - μ^2)

  marginal(wv, μ, σ, interpolate)

end
function marginal{T <: ContinuousUnivariateDistribution}(wv::weights_values, μ::Float64, σ::Float64, interpolate::Type{T})
  marginal(wv, μ, σ, GLM(wv, T))
end
function marginal(wv::weights_values, μ::Float64, σ::Float64, interpolate::Type{Grid})
  marginal(wv, μ, σ, Grid(wv))
end
function marginal(wv::weights_values, μ::Float64, σ::Float64, interpolate::Type{GLM})
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
