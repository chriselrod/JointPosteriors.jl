module JointPosteriors

using Optim
using SparseQuadratureGrids
using ConstrainedParameters
using LogDensities
using Interpolations
using Distributions

import  Base: show, quantile
import  ForwardDiff: hessian
import  LogDensities: log_density!

export  JointPosterior,
        JointPosteriorRaw,
        fit,
        marginal,
        Data,
        parameter,
        CovarianceMatrix,
        PositiveVector,
        ProbabilityVector,
        RealVector,
        Simplex,
        Model,
        construct,
        log_density,
        log_density!,
        log_jacobian!,
        quad_form,
        inv_det,
        inv_root_det,
        root_det,
        log_root_det,
        trace_inverse,
        lpdf_InverseWishart,
        lpdf_normal,
        Normal,
        Adaptive,
        AdaptiveRaw,
        Smolyak



include("joint_posterior.jl")
include("interp.jl")
include("marginal_posterior.jl")

end # module
