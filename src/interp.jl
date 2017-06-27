

struct InterpolateIntegral
  weights::Array{Float64,1}
  values::Array{Float64,1}

end


function generate_value_and_weight_nodes(values::Array{Float64,1}, weights::Array{Float64,1})
  sort_indices = sortperm(values)
  value_nodes = Array{Float64,1}(length(values)+1)
  weight_nodes = Array{Float64,1}(length(values)+1)
  weight_nodes[1] = 0.0
  value_nodes[1] = 2values[sort_indices[1]] - values[sort_indices[2]]
  value_nodes[end] = 2values[sort_indices[end]] - values[sort_indices[end-1]]
  for i ∈ 2:length(values)
    value_nodes[i] = (values[sort_indices[i]] + values[sort_indices[i-1]]) / 2
    weight_nodes[i] = weight_nodes[i-1] + weights[sort_indices[i-1]]
  end
  weight_nodes[end] = weight_nodes[end-1] + weights[sort_indices[end]]
  value_nodes, weight_nodes
end


function interpolate_weights(values::Vector, weights::Vector)
  v, w = generate_value_and_weight_nodes(values, weights)
  interpolate((v,), w, Gridded(Linear()))
end


function InterpolateIntegral(weights_positive::Array{Float64,1}, weights_negative::Array{Float64,1}, values_positive::Array{Float64,1}, values_negative::Array{Float64,1})

  negative_itp = interpolate_weights(values_negative, weights_negative)

  value_nodes, weight_nodes = generate_value_and_weight_nodes(values_positive, weights_positive)

  for i ∈ eachindex(value_nodes)
    weight_nodes[i] += negative_itp[value_nodes[i]]
  end

  InterpolateIntegral(weight_nodes, value_nodes)
end
function cdf(itp::InterpolateIntegral, x::Real)
  if x < itp.values[1]
    return 0.0
  elseif x > itp.values[end]
    return 1.0
  else
    return linear_interp(itp.values, itp.weights, searchsortedfirst(itp.values, x), x)
  end
end
function Base.quantile(itp::InterpolateIntegral, x::Real)
  if x <= 0
    return - Inf
  elseif x >= 1
    return Inf
  elseif x < 0.5
    i = searchsortedfirst(itp.weights, x)
  else
    i = searchsortedlast(itp.weights, x) + 1
  end
  linear_interp(itp.weights, itp.values, i, x)
end
function linear_interp(x::Array{Float64,1}, y::Array{Float64,1}, i::Int, z::Real)
  y[i-1] + (z - x[i-1]) * (y[i] - y[i-1])/(x[i]-x[i-1])
end
