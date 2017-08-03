struct JointPosterior{P}
  Θ::Vector{P}
  density::Vector{Float64}
  Θ_hat::Vector{Float64}
  U::Array{Float64,2}
end
struct JointPosteriorConstrained{p, q, P}
  M::Model{p, q, P}
  grid::FlattenedGrid{q}
  Θ_hat::Vector{Float64}
  U::Array{Float64,2}
end


function chol!(U::AbstractArray{<:Real,2}, Σ::AbstractArray{<:Real,2})
  @inbounds for i ∈ 1:size(U,1)
    U[i,i] = Σ[i,i]
    for j ∈ 1:i-1
      U[j,i] = Σ[j,i]
      for k ∈ 1:j-1
        U[j,i] -= U[k,i] * U[k,j]
      end
      U[j,i] /= U[j,j]
      U[i,i] -= U[j,i]^2
    end
    U[i,i] > 0 ? U[i,i] = √U[i,i] : return false
  end
  true
end
function inv!(U_inverse::AbstractArray{<:Real,2}, U::AbstractArray{<:Real,2})
  for i ∈ 1:size(U,1)
    U_inverse[i,i] = 1 / U[i,i]
    for j ∈ i+1:size(U,1)
      U_inverse[i,j] = U[i,j] * U_inverse[i,i]
      for k ∈ i+1:j-1
        U_inverse[i,j] += U[k,j] * U_inverse[i,k]
      end
      U_inverse[i,j] /= -U[j,j]
    end
  end
end
function inv!(U::AbstractArray{<:Real,2})
  for i ∈ 1:size(U,1)
    U[i,i] = 1 / U[i,i]
    for j ∈ i+1:size(U,1)
      U[i,j] = U[i,j] * U[i,i]
      for k ∈ i+1:j-1
        U[i,j] += U[k,j] * U[i,k]
      end
      U[i,j] /= -U[j,j]
    end
  end
  true
end
function inv_chol!(U, H)
  chol!(U, H) ? inv!(U) : false
end
function reduce_dimensions!(M::Model, H::Array{Float64,2})
  ef = eigfact!(Symmetric(H))
  v0 = ef.values .> 1e-13
  r = sum(v0)
  out = SparseQuadratureGrids.mats(M.Grid, r)
  g0 = 0
  for i ∈ eachindex(ef.values)
    if v0[i]
      g0 += 1
      out[:,g0] .= ef.vectors[:,i] ./ √ef.values[i]
    end
  end
  out
end

function deduce_scale!(M::Model, H::Array{Float64,2})
  inv_chol!(M.Grid.U, H) ? M.Grid.U : reduce_dimensions!(M, H)
end

sigmoid_jacobian(x::AbstractArray{<:Real,1}) = -sum(log, 1 .- x .^ 2)

function negative_log_density!(::Type{GenzKeister}, x::AbstractArray{<:Real,1}, Θ::parameters, data::Data, Θ_hat::Vector, U::AbstractArray{<:Real,2})
  Θ.x .= Θ_hat .+ U * x
  update!(Θ)
  negative_log_density!(Θ, data)
end

function negative_log_density!(::Type{KronrodPatterson}, x::AbstractArray{<:Real,1}, Θ::parameters, data::Data, Θ_hat::Vector, U::AbstractArray{<:Real,2})
  l_jac = sigmoid_jacobian(x)
  Θ.x .= Θ_hat .+ U * sigmoid.(x)
  update!(Θ)
  negative_log_density!(Θ, data) - l_jac
end
function negative_log_density_cache!(::Type{GenzKeister}, x::AbstractArray{<:Real,1}, Θ::parameters, data::Data, Θ_hat::Vector, U::AbstractArray{<:Real,2})
  Θ.x .= Θ_hat .+ U * x
  update!(Θ)
  negative_log_density!(Θ, data), Θ
end

function negative_log_density_cache!(::Type{KronrodPatterson}, x::AbstractArray{<:Real,1}, Θ::parameters, data::Data, Θ_hat::Vector, U::AbstractArray{<:Real,2})
  l_jac = sigmoid_jacobian(x)
  Θ.x .= Θ_hat .+ U * sigmoid.(x)
  update!(Θ)
  negative_log_density!(Θ, data) - l_jac, Θ
end

function transform!(Θ::parameters, x::AbstractArray{<:Real,1}, Θ_hat::Vector, U::AbstractArray{<:Real,2}, ::Type{KronrodPatterson})
  Θ.x .= Θ_hat .+ ( U * sigmoid.(x) )
  update!(Θ)
  Θ
end

function transform!(Θ::parameters, x::AbstractArray{<:Real,1}, Θ_hat::Vector, U::AbstractArray{<:Real,2}, ::Type{GenzKeister})
  Θ.x .= Θ_hat .+ ( U * x )
  update!(Θ)
  Θ
end

sample_size_order(::Data) = 1
function negative_log_density{T, P <: parameters}(Θ::Vector{T}, ::Type{P}, data::Data)
  param = construct(P{T}, Θ)
  nld = -log_jacobian!(param)
  nld - Main.log_density(param, data)
end
function evaluate(M::Model{p, q, P, aPrioriBuild{q}}, Θ_hat::Vector{Float64}, U::Array{Float64,2}, data::Data) where {p,q,P}

end

function evaluate(M::Model{p, q, P, AdaptiveBuild{q}}, Θ_hat::Vector{Float64}, U::Array{Float64,2}, data::Data) where {p,q,P}
  ab = AdaptiveBuilderCache(p, x -> negative_log_density_cache!(q, x, Θ::parameters, data, Θ_hat, U) , P, n = 10_000, q, l = 6)
end
#_constrained return constrained grid.
function evaluate_constrained(M::Model{p, q, P, aPrioriBuild{q}}, Θ_hat::Vector{Float64}, U::Array{Float64,2}, data::Data) where {p,q,P}

end

function evaluate_constrained(M::Model{p, q, P, AdaptiveBuild{q}}, Θ_hat::Vector{Float64}, U::Array{Float64,2}, data::Data) where {p,q,P}
  ab = AdaptiveBuilder(p, x -> negative_log_density!(q, x, M.Θ, data, Θ_hat, U), n = 10_000, q, l::Int = 6)
end

function get_grid()

end

function get_grid_constrained()

end


function JointPosterior( M::Model{p, q, P} , data::Data ) where {p, q <: QuadratureRule, P <: parameters}

  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(x -> negative_log_density(x, M.UA, data), M.Θ.x, autodiff = :forward), method = NewtonTrustRegion()))#LBFGS; NewtonTrustRegion
  U = deduce_scale!(M, 2hessian(x -> negative_log_density(x, M.UA, data), Θ_hat))


  g = M.Grid[size(U, 2), sample_size_order(data)]

  @inbounds for i ∈ eachindex(g.density)
    g.density[i] = g.baseline_weights[i] + negative_log_density!(q, ( @views g.nodes[:,i] ), M.Θ, data, Θ_hat, U)
  end

  g.density .= exp.(minimum(g.density) .- g.density) .* g.weights
  g.density ./= sum(g.density)

  JointPosterior{p, q, P}(M, g, Θ_hat, U)

  Θ, density = evaluate(M, Θ_hat, U, data)
  JointPosterior{P}(Θ, density, Θ_hat, U)
end
function JointPosteriorConstrained( M::Model{p, q, P} , data::Data, build  ) where {p, q <: QuadratureRule, P <: parameters}

  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(x -> negative_log_density(x, M.UA, data), M.Θ.x, autodiff = :forward), method = NewtonTrustRegion()))#LBFGS; NewtonTrustRegion
  U = deduce_scale!(M, 2hessian(x -> negative_log_density(x, M.UA, data), Θ_hat))

  g = M.Grid[size(U, 2)]

  @inbounds for i ∈ eachindex(g.density)
    g.density[i] = g.baseline_weights[i] + negative_log_density!(q, ( @views g.nodes[:,i] ), M.Θ, data, Θ_hat, U)
  end

  g.density .= exp.(minimum(g.density) .- g.density) .* g.weights
  g.density ./= sum(g.density)



  JointPosterior{p, q, P}(M, g, Θ_hat, U)

end
