

struct marginal
  weights_positive::Array{Float64,1}
  weights_negative::Array{Float64,1}
  values_positive::Array{Float64,1}
  values_negative::Array{Float64,1}
  itp::InterpolateIntegral
  μ::Float64
  σ::Float64

end

function marginal{p, q, P}(jp::JointPosterior{p, q, P}, f::Function)

  values_positive = Array{Float64}(length(jp.weights_positive))
  values_negative = Array{Float64}(length(jp.weights_negative))
  for i ∈ eachindex(values_positive)
    values_positive[i] = f(transform!(jp.M.Θ, jp.M.Grid.nodes_positive[:,i], jp.Θ_hat, jp.L, q))
  end
  for i ∈ eachindex(values_negative)
    values_negative[i] = f(transform!(jp.M.Θ, jp.M.Grid.nodes_negative[:,i], jp.Θ_hat, jp.L, q))
  end
  μ = dot(jp.weights_positive, values_positive) + dot(jp.weights_negative, values_negative)
  σ = sqrt(dot(jp.weights_positive, values_positive .^ 2) + dot(jp.weights_negative, values_negative .^ 2) - μ^2)

  itp = InterpolateIntegral(jp.weights_positive, jp.weights_negative, values_positive, values_negative)

  marginal(jp.weights_positive, jp.weights_negative, values_positive, values_negative, itp, μ, σ)
end


function cdf(m::marginal, x)
  cdf(m.itp, x)
end
function Base.quantile(m::marginal, x)
  quantile(m.itp, x)
end

function Base.show(io::IO, m::marginal)
  println("Marginal parameter")
  println("μ: ", m.μ)
  println("σ: ", m.σ)
  println("Quantiles: ", quantile.(m, [.025 .25 .5 .75 .975]))
end
