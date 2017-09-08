
using Plots




function Plots.plot(m::marginal{Ω, <: SmoothCDF} where Ω)
  plot(x -> pdf(m, x), quantile(m, .01), quantile(m, 0.99))
end
