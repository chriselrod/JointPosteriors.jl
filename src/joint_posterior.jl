abstract type JointDist{P} end
struct JointPosterior{P} <: JointDist{P}
  Θ::Vector{P}
  density::Vector{Float64}
  Θ_hat::Vector{Float64}
  U::Array{Float64,2}
end
struct JointPosteriorRaw{p, q, P} <: JointDist{P}
  Θ::P
  grid::FlatGrid{p, q}
  Θ_hat::Vector{Float64}
  U::Array{Float64,2}
end


function try_chol!(U::AbstractArray{<:Real,2}, Σ::AbstractArray{<:Real,2})
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
    U[i,i] = √U[i,i]
  end
end
function inv!(U_inverse::AbstractArray{<:Real,2}, U::AbstractArray{<:Real,2})
  @inbounds for i ∈ 1:size(U,1)
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
  @inbounds for i ∈ 1:size(U,1)
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
function safe_inv_chol!(U, H)
  try_chol!(U, H) ? try_inv!(U) : false
end
function inv_chol!(U, H)
  chol!(U, H)
  inv!(U)
  U
end

function count(v::Vector, ::Type{LDR{g}}) where g
  @assert 0.0 < g < 1.0
  inv_v = 1 ./ v
  inadmissable = sum(inv_v .>= 1e11)
  ind_start = inadmissable + 1
  @inbounds for i ∈ ind_start:length(v)
    total_energy += inv_v[i]
  end
  limit = g*total_energy
  p = 0
  cumulative_energy = 0.0
  @inbounds while cumulative_energy < limit
    p += 1
    cumulative_energy += inv_v[ind_start]
    ind_start += 1
  end
  p, inadmissable
end

#reduce_dimensions assumes the eigenvalues are sorted from smallest to largest.
function reduce_dimensions!(M::Model, H::Array{Float64,2})
  ef = eigfact!(Symmetric(H))
  out = SparseQuadratureGrids.mats(M.Grid, sum(ef.values .> 1e-11))
  g0 = 0
  for (i, v) ∈ enumerate(ef.values)
    if v < 1e-11
      continue
    end
    g0 += 1
    out[:,g0] .= ef.vectors[:,i] ./ √ef.values[i]
  end
  out
end
function reduce_dimensions!(M::Model, H::Array{Float64,2}, ::Type{R}) where {g, R <: LDR{g}}
  ef = eigfact!(Symmetric(H))
  p, inadmissable = count(ef.values, R)
  out = SparseQuadratureGrids.mats(M.Grid, p)
  for (g0, i) ∈ enumerate(inadmissable+(1:p))
    out[:,g0] .= ef.vectors[:,i] ./ √ef.values[i]
  end
  out
end
function reduce_dimensions!(M::Model, H::Array{Float64,2}, ::Type{FixedRank{p}}) where p
  ef = eigfact!(Symmetric(H))
  out = SparseQuadratureGrids.mats(M.Grid, p)
  g0 = 0
  for (i, v) ∈ enumerate(ef.values)
    if v < 1e-11
      continue
    elseif g0 >= p
      break
    end
    g0 += 1
    out[:,g0] .= ef.vectors[:,i] ./ √ef.values[i]
  end
  out
end

function deduce_scale!(M::Model, H::Array{Float64,2}, ::Type{Dynamic})
  safe_inv_chol!(M.Grid.U, H) ? M.Grid.U : reduce_dimensions!(M, H)
end
function deduce_scale!(M::Model, H::Array{Float64,2}, ::Type{Full{p}}) where p
  inv_chol!(M.Grid.U, H)
end
function deduce_scale!(M::Model, H::Array{Float64,2}, ::Type{R}) where R
  reduce_dimensions!(M, H, R)
end

sigmoid_jacobian(x::AbstractArray{<:Real,1}) = -sum(log, 1 .- x .^ 2)

function log_density!(::Type{GenzKeister}, x::AbstractArray{<:Real,1}, Θ::parameters, data, Θ_hat::Vector, U::AbstractArray{<:Real,2}, neg_min::Float64 = 0.0)
  Θ.x .= Θ_hat .+ U * x
  update!(Θ)
  log_density!(Θ, data) + neg_min
end

function log_density!(::Type{KronrodPatterson}, x::AbstractArray{<:Real,1}, Θ::parameters, data, Θ_hat::Vector, U::AbstractArray{<:Real,2}, neg_min::Float64 = 0.0)
  l_jac = sigmoid_jacobian(x)
  Θ.x .= Θ_hat .+ U * sigmoid.(x)
  update!(Θ)
  log_density!(Θ, data) + neg_min + l_jac
end
function log_density_cache!(::Type{GenzKeister}, x::AbstractArray{<:Real,1}, ::Type{P}, data, Θ_hat::Vector, U::AbstractArray{<:Real,2}, neg_min::Float64 = 0.0) where P
  Θ = construct(P, Θ_hat .+ U * x)
  update!(Θ)
  log_density!(Θ, data) + neg_min, Θ
end

function log_density_cache!(::Type{KronrodPatterson}, x::AbstractArray{<:Real,1}, ::Type{P}, data, Θ_hat::Vector, U::AbstractArray{<:Real,2}, neg_min::Float64 = 0.0) where P
  Θ = construct(P, Θ_hat .+ U * sigmoid.(x))
  update!(Θ)
  log_density!(Θ, data) + neg_min + sigmoid_jacobian(x), Θ
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


index_gen(U::Array{Float64,2},::Type{<:DynamicRank}, data, n::Int) = size(U,2),sample_size_order(data), n
index_gen(U::Array{Float64,2},::Type{<:StaticRank}, data, n::Int) = sample_size_order(data), n
index_gen(U::Array{Float64,2},::Type{<:DynamicRank}, data, seq::Vector{Int}) = size(U,2), seq
index_gen(U::Array{Float64,2},::Type{<:StaticRank}, data, seq::Vector{Int}) = seq

function mode(M::Model{G, PF, P, R} where {G, PF}, data) where {P, R}
    nld(x::Vector) =  - log_density(x, P, data)
    optimum = optimize(OnceDifferentiable(nld, M.Θ.x, autodiff = :forward), method = NewtonTrustRegion())#LBFGS; NewtonTrustRegion
    Θ_hat = Optim.minimizer(optimum)
    Θ_hat, deduce_scale!(M, 2hessian(nld, Θ_hat), R), Optim.minimum(optimum)
end


function fit( M::Model{G, PF, P, R}, data, n::Int = 1_000 ) where {q, B <: Adaptive, G <: GridVessel{q, B}, PF <: parameters, P <: parameters, R <: ModelRank}
    Θ_hat, U, neg_min = mode( M, data )
    snld(x::AbstractArray) = log_density_cache!(q, x, PF, data, Θ_hat, U, neg_min)
    density, Θ = get!(M, index_gen(U, R, data, n), snld)
    JointPosterior{P}(Θ, density, Θ_hat, U)
end
function fit( M::Model{G, PF, P, R}, data, seq::Vector{Int} = SparseQuadratureGrids.default(q) ) where {q, B <: aPrioriBuild, G <: GridVessel{q, B}, PF <: parameters, P <: parameters, R <: ModelRank}
    Θ_hat, U, neg_min = mode( M, data )
    snld(x::AbstractArray) = log_density_cache!(q, x, PF, data, Θ_hat, U, neg_min)
    density, Θ = get!(M, index_gen(U, R, data, seq), snld)
    JointPosterior{P}(Θ, density, Θ_hat, U)
end
function fit( M::Model{G, PF, P, R}, data, seq::Vector{Int} = SparseQuadratureGrids.default(q) ) where {q, B <: RawBuild, G <: GridVessel{q, B}, PF <: parameters, P <: parameters, R <: ModelRank}
    Θ_hat, U, neg_min = mode( M, data )
    snld(x::AbstractArray) = log_density!(q, x, M.Θ, data, Θ_hat, U, neg_min)
    JointPosteriorRaw( M.Θ, return_grid!(M, index_gen(U, R, data, seq), snld), Θ_hat, U )
end
function fit( M::Model{G, PF, P, R}, data, n::Int = 1_000 ) where {q, B <: AdaptiveRaw, G <: GridVessel{q, B}, PF <: parameters, P <: parameters, R <: ModelRank}
    Θ_hat, U, neg_min = mode( M, data )
    snld( x::AbstractArray ) = log_density!( q, x, M.Θ, data, Θ_hat, U, neg_min)
    JointPosteriorRaw( M.Θ, return_grid!(M, index_gen(U, R, data, n), snld), Θ_hat, U )
end

###These are the two functions called by the JointPosteriors package.
###The first, return_grid!, is for the raw version, and it simply returns a flattened grid of the raw unconstrained values.
###The second returns a vector of parameter objects. More costly, but should be cheaper to compute marginals on thanks to cacheing the transformations.
function return_grid!( M::Model{G, PF, P, R} where {G <: GridVessel, PF <: parameters, P <: parameters, R <: ModelRank}, i, f::Function )
  haskey(M.Grid.grids, i) ? eval_grid(M.Grid.grids[i], f) : calc_grid!(M.Grid, i, f)
end
function get!( M::Model{G, PF, P, R} where {G <: GridVessel, P <: parameters, R <: ModelRank}, i, f::Function ) where {PF <: parameters}
  haskey(M.Grid.grids, i) ? eval_grid(M.Grid.grids[i], f, PF) : calc_grid!(M.Grid, i, f, PF)
end
