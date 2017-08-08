module JointPosteriors

using Optim
using SparseQuadratureGrids
using ConstrainedParameters
using LogDensities
using Interpolations
using Distributions

import  Base: show, quantile
import  ForwardDiff: hessian
import  LogDensities: negative_log_density!

export  JointPosterior,
        JointPosteriorRaw,
        fit,
        marginal,
        Data,
        parameters,
        CovarianceMatrix,
        PositiveVector,
        ProbabilityVector,
        RealVector,
        Simplex,
        Model,
        construct,
        negative_log_density,
        negative_log_density!,
        log_jacobian!,
        quad_form,
        inv_det,
        inv_root_det,
        root_det,
        log_root_det,
        trace_inverse,
        lpdf_InverseWishart,
        lpdf_normal,
        Normal



include("joint_posterior.jl")
include("interp.jl")
include("marginal_posterior.jl")

end # module
