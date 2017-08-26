
using Plots




function Plots.plot(m::marginal_posterior{Ω, <: SmoothCDF} where Ω)
  plot(x -> pdf(m, x), quantile(m, .01), quantile(m, 0.99))
end
