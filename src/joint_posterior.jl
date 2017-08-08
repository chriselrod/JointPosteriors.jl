struct JointPosterior{P}
  Θ::Vector{P}
  density::Vector{Float64}
  Θ_hat::Vector{Float64}
  U::Array{Float64,2}
end
struct JointPosteriorRaw{p, q, P}
  Θ::P
  grid::FlatGrid{p, q}
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
  v0 = ef.values .> 1e-11
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
function negative_log_density_cache!(::Type{GenzKeister}, x::AbstractArray{<:Real,1}, ::Type{P}, data::Data, Θ_hat::Vector, U::AbstractArray{<:Real,2})
  Θ = construct(P, Θ_hat .+ U * x)
#  update!(Θ)
  negative_log_density!(Θ, data), Θ
end

function negative_log_density_cache!(::Type{KronrodPatterson}, x::AbstractArray{<:Real,1}, ::Type{P}, data::Data, Θ_hat::Vector, U::AbstractArray{<:Real,2})
  Θ = construct(P, Θ_hat .+ U * sigmoid.(x))
#  update!(Θ)
  negative_log_density!(Θ, data) - sigmoid_jacobian(x), Θ
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

sample_size_order(::Any) = 1

function negative_log_density{T, P <: parameters}(Θ::Vector{T}, ::Type{P}, data::Data)
  param = construct(P{T}, Θ)
  nld = -log_jacobian!(param)
  nld - Main.log_density(param, data)
end

index_gen(U::Array{Float64,2},::Type{<:DynamicRank}, data, n::Int) = size(U,2),sample_size_order(data), n
index_gen(U::Array{Float64,2},::Type{<:StaticRank}, data, n::Int) = sample_size_order(data), n
function fit( M::Model{G, PF, P, R}, data, n::Int = 10_000 ) where {q, G <: GridVessel{q}, PF <: parameters, P <: parameters, R <: ModelRank}

  nld(x::Vector) = negative_log_density(x, P, data)

  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(nld, M.Θ.x, autodiff = :forward), method = NewtonTrustRegion()))#LBFGS; NewtonTrustRegion
  U = deduce_scale!(M, 2hessian(nld, Θ_hat))

  snld(x::AbstractArray) = negative_log_density_cache!(q, x, PF, data, Θ_hat, U)

  density, Θ = get!(M, index_gen(U, data, n), snld)

  JointPosterior{P}(Θ, density, Θ_hat, U)
end
function rawFit( M::Model{G, PF, P, R}, data, seq::Vector{Int} = SparseQuadratureGrids.default(q) ) where {q, p, G <: GridVessel{q}, PF <: parameters, P <: parameters, R <: StaticRank{p}}
  nld(x::Vector) = negative_log_density(x, P, data)

  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(nld, M.Θ.x, autodiff = :forward), method = NewtonTrustRegion()))#LBFGS; NewtonTrustRegion
  U = deduce_scale!(M, 2hessian(nld, Θ_hat))

  snld(x::AbstractArray) = negative_log_density!(q, x, M.Θ, data, Θ_hat, U)

  JointPosteriorRaw{p, q, P}(M.Θ, return_grid!(M, (sample_size_order(data), seq), snld), Θ_hat, U)
end
function rawFit( M::Model{G, PF, P, R}, data, seq::Vector{Int} = SparseQuadratureGrids.default(q) ) where {q, G <: GridVessel{q}, PF <: parameters, P <: parameters, R <: DynamicRank}
  nld(x::Vector) = negative_log_density(x, P, data)

  Θ_hat = Optim.minimizer(optimize(OnceDifferentiable(nld, M.Θ.x, autodiff = :forward), method = NewtonTrustRegion()))#LBFGS; NewtonTrustRegion
  U = deduce_scale!(M, 2hessian(nld, Θ_hat))

  snld(x::AbstractArray) = negative_log_density!(q, x, M.Θ, data, Θ_hat, U)

  JointPosteriorRaw( M.Θ, return_grid!(M.Θ, (sample_size_order(data), seq), snld), Θ_hat, U )
end
JointPosteriorRaw( Θ::P, Grid::FlatGrid{p, q}, Θ_hat::Vector{Float64}, U::Array{Float64,2} ) where {p, q, P} = JointPosteriorRaw{p, q, P}( Θ, Grid, Θ_hat, U )

###These are the two functions called by the JointPosteriors package.
###The first, return_grid!, is for the raw version, and it simply returns a flattened grid of the raw unconstrained values.
###The second returns a vector of parameter objects. More costly, but should be cheaper to compute marginals on thanks to cacheing the transformations.
function return_grid!( M::Model{G, PF, P, R} where {G <: GridVessel, PF <: parameters, P <: parameters, R <: ModelRank}, i, f::Function )
  haskey(GV.grids, i) ? eval_grid(GV.grids[i], f) : calc_grid!(GV, i, f)
end
function get!( M::Model{G, PF, P, R} where {G <: GridVessel, P <: parameters, R <: ModelRank}, i, f::Function ) where {PF <: parameters}
  haskey(GV.grids, i) ? eval_grid(GV.grids[i], f, PF) : calc_grid!(GV, i, f, PF)
end
