module ClapeyronDirectQhullExt

using Clapeyron, DirectQhull

function Clapeyron.cem_qhull(points::Matrix{Float64})
    hull = DirectQhull.ConvexHull(points)
    return hull.simplices,hull.neighbors,hull.equations
end

end
