abstract type JointDist{𝛭} end
struct JointPosterior{P, 𝛭} <: JointDist{𝛭}
    M::𝛭
    Θ::Vector{P}
    density::Vector{Float64}
    μ_hat::Vector{Float64}
    U::Matrix{Float64}
end
struct JointPosteriorRaw{G, 𝛭} <: JointDist{𝛭}
    M::𝛭
    grid::G
    μ_hat::Vector{Float64}
    U::Matrix{Float64}
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


@inline function log_density!(Θ::ModelParam, data, neg_min::Float64 = 0.0)
  update!(Θ)
  log_density(Θ, data) + neg_min
end
@inline function log_density_cache(x::AbstractVector{<:Real}, θ::Tuple, data, neg_min::Float64 = 0.0)
    param = construct(x, θ)
    log_density(param, data) + neg_min, param
end


sample_size_order(::Any) = 1

index(U::Array{Float64,2},::Type{<:DynamicRank}, data, n::Int) = size(U,2),sample_size_order(data), n
index(U::Array{Float64,2},::Type{<:StaticRank}, data, n::Int) = sample_size_order(data), n
index(U::Array{Float64,2},::Type{<:DynamicRank}, data, seq::Vector{Int}) = size(U,2), seq
index(U::Array{Float64,2},::Type{<:StaticRank}, data, seq::Vector{Int}) = seq

function mode(M::Model{G, MP, P, R} where {G, MP, P}, data, f::Function = Main.log_density) where R
	optimize!(ModelDiff(M.diff_buffer, f, data))
	M.diff_buffer.state.x, deduce_scale!(M, 2M.diff_buffer.dr.derivs[2], R), M.diff_buffer.dr.value
end


function SlidingVecFun( Θ::ModelParam, data, neg_min::Float64 )
    ld = () -> log_density!( Θ, data, neg_min )
    SlidingVecFun(ld, Θ.v)
end


function fit( M::Model{G, MP, P, R} where P, data, n = default(B) ) where {q, B <: CacheBuild, G <: GridVessel{q, B}, MP, R}
    μ_hat, U, neg_min = mode( M, data )
    ldc(x::AbstractVector) = log_density_cache(x, M.ϕ, data, neg_min)
    Θ, density = eval_grid!( M.Grid, ldc, μ_hat, U, index(U, R, data, n), MP )
    JointPosterior( M, Θ, density, μ_hat, U )
end
function fit( M::Model{G, MP, P, R} where {MP, P}, data, n = default(B) ) where {q, B <: RawBuild, G <: GridVessel{q, B}, R}
    μ_hat, U, neg_min = mode( M, data )
    svf = SlidingVecFun( M.Θ, data, neg_min )
    g = eval_grid!( M.Grid, svf, μ_hat, U, index(U, R, data, n) )
    JointPosteriorRaw( M, g, μ_hat, U )
end
