module JointPosteriors

using Optim
using SparseQuadratureGrids
using LogDensities
using Interpolations

import  Base.show,
        Base.quantile
import  ForwardDiff: hessian
import  LogDensities: negative_log_density!

export  JointPosterior,
        marginal,
        Data,
        parameters,
        CovarianceMatrix,
        PositiveVector,
        ProbabilityVector,
        RealVector,
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
        lpdf_normal



include("joint_posterior.jl")
include("interp.jl")
include("marginal_posterior.jl")

end # module
