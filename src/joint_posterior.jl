struct JointPosterior{p, q, P}
  M::Model
  weights_positive::Array{Float64,1}
  weights_negative::Array{Float64,1}
  Θ_hat::Array{Float64,1}
  L::LowerTriangular{Float64,Array{Float64,2}}
end
function lower_chol{T <: Real}(A::Array{T, 2})
  try
    return inv(chol(Symmetric(A), :L))
  catch x
    ef = eigfact(A)
    p = size(A,1)
    λ = ef.values
    z = isapprox.(λ, 0.0, atol = 1e-10)
    warn("x"*"\nModel is probably nearly unidentifiable.\nModel has $p parameters but is of approximate rank $(p-sum(z)).")
#    for i ∈ 1:p
#      if z[i]
#        λ[i] = 1.0
#      else
#        λ[i] = d[i]^-0.5
#      end
#    end
    warn("Temporary hack-fix, because currently only lower cholesky outputs are supported.")
    return lower_chol(A + 0.001I)
  end
end

function negative_log_density!(::Type{GenzKeister}, x::Vector, Θ::parameters, data::Data, Θ_hat::Vector, L::LowerTriangular)
  Θ.x .= Θ_hat .+ L * x
  update!(Θ)
  negative_log_density!(Θ, data)
end

sigmoid_jacobian(x::Vector) = -sum(log, 1 .- x .^ 2)

function negative_log_density!(::Type{KronrodPatterson}, x::Vector, Θ::parameters, data::Data, Θ_hat::Vector, L::LowerTriangular)
  l_jac = sigmoid_jacobian(x)
  Θ.x .= Θ_hat .+ L * sigmoid.(x)
  update!(Θ)
  negative_log_density!(Θ, data) - l_jac
end

function transform!(Θ::parameters, x::Vector, Θ_hat::Vector, L::LowerTriangular, ::Type{KronrodPatterson})
  Θ.x .= Θ_hat .+ ( L * sigmoid.(x) )
  update!(Θ)
  Θ
end

function transform!(Θ::parameters, x::Vector, Θ_hat::Vector, L::LowerTriangular, ::Type{GenzKeister})
  Θ.x .= Θ_hat .+ ( L * x )
  update!(Θ)
  Θ
end


function JointPosterior{p, q <: QuadratureRule, P <: parameters}( M::Model{p, q, P} , data::Data )

  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(x -> negative_log_density(x, M.UA, data), M.Θ.x, autodiff = :forward), method = NewtonTrustRegion()))#LBFGS; NewtonTrustRegion
  L = lower_chol(hessian(x -> negative_log_density(x, M.UA, data), Θ_hat) * 2)

#  Θ = construct(P{Float64})
  weights_positive = copy(M.Grid.baseline_weights_positive)
  weights_negative = copy(M.Grid.baseline_weights_negative)

  for i ∈ eachindex(weights_positive)
    weights_positive[i] += negative_log_density!(q, M.Grid.nodes_positive[:,i], M.Θ, data, Θ_hat, L)
  end
  for i ∈ eachindex(weights_negative)
    weights_negative[i] += negative_log_density!(q, M.Grid.nodes_negative[:,i], M.Θ, data, Θ_hat, L)
  end
  min_density = min(minimum(weights_positive), minimum(weights_negative))
  weights_positive .= exp.(min_density .- weights_positive) .* M.Grid.weights_positive
  weights_negative .= exp.(min_density .- weights_negative) .* M.Grid.weights_negative
  weights_positive .*= M.Grid.weight_sum[1] ./ sum(weights_positive)
  weights_negative .*= M.Grid.weight_sum[2] ./ sum(weights_negative)

  JointPosterior{p, q, P}(M, weights_positive, weights_negative, Θ_hat, L)

end
