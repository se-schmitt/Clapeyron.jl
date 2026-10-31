#=
Convex envelope method (CEM)

Göttl, Rosen, Burger (2025), Comput. Chem. Eng. 203, 109326. doi:10.1016/j.compchemeng.2025.109326

The reduced Gibbs energy of every aggregate state is evaluated on a lattice of the composition simplex
(step 1/δ). The phase split of a feed z is the facet of the lower convex envelope of those points above z:

    min Σₖ λₖ gₖ   s.t.  Σₖ λₖ xₖ = z,  λ ≥ 0

The lower convex envelope is obtained from the convex hull of the points (x₁ … xₙ₋₁, g), calculated with
Qhull (DirectQhull.jl extension). The facets with a downward normal form a triangulation of the composition
simplex. The facet that contains a feed is found by a visibility walk over neighbouring facets (the
triangulation is regular, so the walk always terminates).

Points of the optimal basis belong to the same phase when they have the same aggregate state (fluid points
with almost equal volumes are also merged: near-critical region) and are lattice neighbours, or are connected
through the stable single phase region (see cem_samephase).
=#

const CEM_LIQUID = Int8(1)
const CEM_VAPOUR = Int8(2)
const CEM_SOLID = Int8(3)
const CEM_λTOL = 1e-12 #phase amount tolerance
const CEM_VOL_RTOL = 0.1 #fluid points with this relative volume difference are the same phase

cem_state_symbol(s) = s == CEM_LIQUID ? :liquid : (s == CEM_VAPOUR ? :vapour : :solid)

cem_default_δ(N) = N == 2 ? 256 : (N == 3 ? 128 : (N == 4 ? 64 : 32))

#grid + lower convex hull
struct CEMEnvelope
    δ::Int
    lattice::Matrix{Int32} #N × npoints, lattice coordinates (x = k/δ)
    x::Matrix{Float64} #N × ncolumns
    g::Vector{Float64} #reduced gibbs energy, minus the linear function x⋅gpure
    gpure::Vector{Float64}
    point::Vector{Int32} #lattice point of each column
    state::Vector{Int8}
    volume::Vector{Float64}
    point_column::Vector{Int32} #column with the lowest g at each lattice point
    vertex::Vector{Int32} #point_column of the pure components
    facets::Matrix{Int32} #N × nfacets, columns of the lower hull facets
    neighbors::Matrix{Int32} #N × nfacets, facet opposite to each vertex (0: none)
end

Base.length(env::CEMEnvelope) = size(env.x,1)

function cem_lattice(N,δ)
    K = Matrix{Int32}(undef,N,binomial(δ + N - 1,N - 1))
    k = zeros(Int32,N)
    _cem_lattice!(K,k,1,δ,Ref(0))
    return K
end

function _cem_lattice!(K,k,i,rem,j)
    if i == length(k)
        k[i] = rem
        j[] += 1
        K[:,j[]] .= k
        return nothing
    end
    for ki in rem:-1:0
        k[i] = ki
        _cem_lattice!(K,k,i + 1,rem - ki,j)
    end
    return nothing
end

#model used to evaluate the fluid phases, solid model (or nothing)
function cem_models(model,p,T,z,equilibrium)
    solid = nothing
    fluid = model
    if model isa CompositeModel && model.solid !== nothing
        solid = model.solid
        fluid = model.fluid
        if !(solid isa SolidHfusModel)
            throw(ArgumentError("CEM: only `SolidHfus` solid models are supported (pure solid phases), got $(typeof(solid))"))
        end
    end
    cached_fluid = __tpflash_cache_model(fluid,p,T,z,is_lle(equilibrium) ? :lle : :vle) #PTFlashWrapper does not know :auto
    return fluid,cached_fluid,solid
end

#Δμᵢ/RT (pure solid - pure liquid) of each component, NaN if the component does not form a solid
function cem_solid_Δμ(model,solid,p,T)
    N = length(model)
    Δμ = fill(NaN,N)
    solid === nothing && return Δμ
    solid_comps = component_list(solid)
    μ = chemical_potential(solid,p,T,ones(length(solid)))
    fluid_comps = component_list(model)
    for i in 1:N
        j = findfirst(isequal(fluid_comps[i]),solid_comps)
        j === nothing || (Δμ[i] = μ[j]/(Rgas(solid)*T))
    end
    return Δμ
end

#returns (gl,vl,gv,vv), NaN g if the column does not exist (lle, or single EoS root of the other state)
function cem_eval_point(fluid,p,T,x,lle)
    nan = NaN
    if fluid isa PTFlashWrapper
        gl,vl = modified_gibbs(fluid,p,T,x,:liquid,nan)
        lle && return gl,vl,nan,nan
        gv,vv = modified_gibbs(fluid,p,T,x,:vapour,nan)
        return gl,vl,gv,vv
    end
    vl = volume(fluid,p,T,x,phase = :liquid)
    gl,_ = modified_gibbs(fluid,p,T,x,:liquid,vl)
    lle && return gl,vl,nan,nan
    vv = volume(fluid,p,T,x,phase = :vapour)
    if isapprox(vl,vv,rtol = 1e-8)
        #only one root: store it with the state it actually has
        if is_vapour(identify_phase(fluid,p,T,x,vol = vl))
            return nan,nan,gl,vl
        else
            return gl,vl,nan,nan
        end
    end
    gv,_ = modified_gibbs(fluid,p,T,x,:vapour,vv)
    return gl,vl,gv,vv
end

function cem_chunks(n,threaded)
    #more chunks than threads: the cost per lattice point is very uneven (load balancing)
    nt = threaded ? 8*Threads.nthreads() : 1
    nt = max(1,min(nt,n ÷ 256))
    #contiguous blocks: consecutive lattice points are neighbours (warm starts)
    bounds = round.(Int,range(0,n,length = nt + 1))
    return [(bounds[i] + 1):bounds[i + 1] for i in 1:nt]
end

function cem_foreach_chunk(f,n,threaded)
    chunks = cem_chunks(n,threaded)
    if length(chunks) == 1
        return [f(chunks[1])]
    else
        tasks = [StableTasks.@spawn f(chunk) for chunk in chunks]
        return fetch.(tasks)
    end
end

function cem_eval_chunk!(gl,vl,gv,vv,fluid,p,T,K,δ,lle,idx)
    x = zeros(size(K,1))
    for k in idx
        x .= @view(K[:,k]) ./ δ
        gl[k],vl[k],gv[k],vv[k] = cem_eval_point(fluid,p,T,x,lle)
    end
    return nothing
end

"""
    cem_envelope(model,p,T,δ,equilibrium = :auto,threaded = true)

Evaluates the Gibbs energy of all aggregate states on the composition lattice and calculates the lower convex hull.
"""
function cem_envelope(model,p,T,δ,equilibrium = :auto,threaded = true)
    N = length(model)
    δ < 1 && throw(DomainError(δ,"CEM: δ must be a positive integer"))
    z0 = fill(1/N,N)
    _,fluid,solid = cem_models(model,p,T,z0,equilibrium)
    lle = is_lle(equilibrium)
    K = cem_lattice(N,δ)
    M = size(K,2)
    gl,vl,gv,vv = zeros(M),zeros(M),zeros(M),zeros(M)
    cem_foreach_chunk(idx -> cem_eval_chunk!(gl,vl,gv,vv,fluid,p,T,K,δ,lle,idx),M,threaded)
    Δμs = cem_solid_Δμ(model,solid,p,T)
    #compact into columns
    Xv = Float64[]
    G,P,S,V = Float64[],Int32[],Int8[],Float64[]
    function addcol!(k,g,s,v)
        isfinite(g) || return nothing
        append!(Xv,@view(K[:,k]))
        push!(G,g);push!(P,k);push!(S,s);push!(V,v)
        return nothing
    end
    for k in 1:M
        addcol!(k,gl[k],CEM_LIQUID,vl[k])
        addcol!(k,gv[k],CEM_VAPOUR,vv[k])
        i = findfirst(isequal(δ),@view(K[:,k]))
        if i !== nothing && isfinite(Δμs[i])
            #pure solid, relative to the pure liquid (or the only fluid root)
            addcol!(k,(isfinite(gl[k]) ? gl[k] : gv[k]) + Δμs[i],CEM_SOLID,NaN)
        end
    end
    X = reshape(Xv,N,length(G)) ./ δ
    #remove the linear part (affine transformations do not change the envelope)
    gpure = fill(Inf,N)
    for c in eachindex(G)
        i = findfirst(isequal(δ),@view(K[:,P[c]]))
        i !== nothing && (gpure[i] = min(gpure[i],G[c]))
    end
    all(isfinite,gpure) || throw(error("CEM: Gibbs energy evaluation failed for a pure component at p = $p, T = $T"))
    for c in eachindex(G)
        G[c] -= dot(@view(X[:,c]),gpure)
    end
    return cem_build_envelope(δ,K,X,G,gpure,P,S,V)
end

## lower convex hull

"""
    cem_qhull(points::Matrix{Float64})

Convex hull of the columns of `points` (one point per column). Returns `(simplices, neighbors, equations)` as in `DirectQhull.ConvexHull`.
Defined in the DirectQhull.jl extension.
"""
function cem_qhull(points)
    throw(error("The convex envelope method requires DirectQhull.jl. Load it with `using DirectQhull`."))
end

function cem_build_envelope(δ,K,X,G,gpure,P,S,V)
    N,nc = size(X)
    M = size(K,2)
    point_column = zeros(Int32,M)
    for c in 1:nc
        k = P[c]
        c0 = point_column[k]
        if c0 == 0 || G[c] < G[c0]
            point_column[k] = c
        end
    end
    vertex = zeros(Int32,N)
    for k in 1:M
        i = findfirst(isequal(δ),@view(K[:,k]))
        i !== nothing && (vertex[i] = point_column[k])
    end
    #columns above the plane of the pure components (g > 0) can not touch the lower envelope. Removing them also
    #protects Qhull from huge values (failed volume solutions), that destroy its precision.
    hull_cols = findall(<=(1e-12),G)
    simplices,neighbors,equations = cem_qhull(vcat(X[1:(N - 1),hull_cols],permutedims(G[hull_cols])))
    #lower facets: the outward normal points to negative g (vertical facets are the boundary of the composition simplex)
    lower = [equations[N,f] < -1e-10 for f in 1:size(simplices,2)]
    id = cumsum(lower)
    facets = Int32.(hull_cols[simplices[:,lower]])
    #the lower facets must tile the composition simplex: Σ|det(xᵢ)| = 1 (Qhull only prints its precision errors)
    tiled = sum(f -> abs(det(X[:,facets[:,f]])),axes(facets,2);init = 0.0)
    isapprox(tiled,1.0,atol = 1e-6) || throw(error("CEM: Qhull returned an invalid convex hull (the lower facets cover $tiled of the composition space)"))
    nb = zeros(Int32,N,size(facets,2))
    for f in findall(lower), j in 1:N
        fn = neighbors[j,f]
        lower[fn] && (nb[j,id[f]] = id[fn])
    end
    return CEMEnvelope(δ,K,X,G,gpure,P,S,V,point_column,vertex,facets,nb)
end

## facet location

struct CEMWorkspace
    B::Vector{Int32}
    Bm::Matrix{Float64}
    λ::Vector{Float64}
    facet::Base.RefValue{Int}
end

CEMWorkspace(env::CEMEnvelope) = CEMWorkspace(env.facets[:,1],zeros(length(env),length(env)),zeros(length(env)),Ref(1))

#barycentric coordinates λ of z in facet f (λ = -1 for a degenerate facet)
function cem_barycentric!(ws::CEMWorkspace,env::CEMEnvelope,f,z)
    ws.facet[] = f
    for j in eachindex(ws.B)
        c = env.facets[j,f]
        ws.B[j] = c
        ws.Bm[:,j] .= @view(env.x[:,c])
    end
    F = lu!(ws.Bm,check = false)
    if issuccess(F)
        ldiv!(ws.λ,F,z)
    else
        ws.λ .= -1
    end
    return ws
end

#lower hull facet that contains z: visibility walk, starting from the current facet of ws
function cem_locate!(ws::CEMWorkspace,env::CEMEnvelope,z)
    nf = size(env.facets,2)
    f = ws.facet[]
    for _ in 1:nf
        cem_barycentric!(ws,env,f,z)
        λmin,j = findmin(ws.λ)
        if λmin >= -1e-12
            ws.λ .= max.(ws.λ,0.0)
            return ws
        end
        f = env.neighbors[j,f]
        f == 0 && break
    end
    #fallback (numerical problems only): search all facets
    for f in 1:nf
        cem_barycentric!(ws,env,f,z)
        if minimum(ws.λ) >= -1e-9
            ws.λ .= max.(ws.λ,0.0)
            return ws
        end
    end
    throw(error("CEM: composition $z is not inside the convex envelope"))
end

#index of a lattice point (in the order of cem_lattice)
function cem_lattice_rank(k,δ)
    N = length(k)
    R = δ
    idx = 1
    for i in 1:(N - 1)
        d = R - k[i]
        d > 0 && (idx += binomial(d - 1 + N - i,N - i))
        R -= k[i]
    end
    return idx
end

#tangent plane (μᵢ/RT, in the reference of the model) of the facet with columns cols
cem_tangent_plane(env::CEMEnvelope,cols) = transpose(env.x[:,cols]) \ env.g[cols] .+ env.gpure

## phases

#columns a and b can belong to the same phase (aggregate state)
function cem_compatible(env::CEMEnvelope,a,b)
    sa,sb = env.state[a],env.state[b]
    (sa == CEM_SOLID || sb == CEM_SOLID) && return a == b
    sa == sb && return true
    return isapprox(env.volume[a],env.volume[b],rtol = CEM_VOL_RTOL) #near critical: the roots merge
end

#stability (single phase) of lattice points, from a diagram or lazily from the facet that contains each point
struct CEMStability
    env::CEMEnvelope
    point_facet::Union{Nothing,Vector{Int32}}
    ws::CEMWorkspace
    memo::Dict{Int,Bool}
end

CEMStability(env::CEMEnvelope,point_facet = nothing) = CEMStability(env,point_facet,CEMWorkspace(env),Dict{Int,Bool}())

function (stable::CEMStability)(k)
    stable.point_facet !== nothing && return iszero(stable.point_facet[k])
    return get!(stable.memo,k) do
        env,ws = stable.env,stable.ws
        cem_locate!(ws,env,env.x[:,env.point_column[k]])
        cem_numphases(env,ws) == 1
    end
end

#lattice point next to x (in lattice units, sum(x) = δ)
function cem_round_lattice!(k,x,δ)
    k .= floor.(Int32,x)
    for _ in 1:(δ - sum(k))
        _,i = findmax(i -> x[i] - k[i],eachindex(x))
        k[i] += 1
    end
    return k
end

#=
Two hull vertices of the same aggregate state are the same phase if they are lattice neighbours [1].
At grid resolution, a binodal that runs almost parallel to a lattice direction (e.g. close to an edge) has hull vertices that are
several lattice steps apart. They are also the same phase if the segment between them stays in the stable single phase region,
i.e. a stable lattice point of the same state lies on or next to every lattice step of the segment. For a real phase split, the
segment crosses the miscibility gap.
=#
function cem_samephase(env::CEMEnvelope,a,b,stable = nothing)
    cem_compatible(env,a,b) || return false
    K = env.lattice
    pa,pb = env.point[a],env.point[b]
    dist,n = 0,0
    for i in 1:size(K,1)
        dk = abs(K[i,pa] - K[i,pb])
        dist += dk
        n = max(n,dk)
    end
    dist <= 2 && return true
    (stable === nothing || n > max(4,env.δ ÷ 8)) && return false
    N = size(K,1)
    x = zeros(N)
    kq = zeros(Int32,N)
    kn = zeros(Int32,N)
    for j in 1:(n - 1)
        for i in 1:N
            x[i] = K[i,pa] + (j/n)*(K[i,pb] - K[i,pa])
        end
        cem_round_lattice!(kq,x,env.δ)
        ok = cem_stable_compatible(env,cem_lattice_rank(kq,env.δ),a,stable)
        for jj in 1:N, ii in 1:N
            (ok || ii == jj || kq[jj] == 0) && continue
            kn .= kq
            kn[jj] -= 1
            kn[ii] += 1
            ok = cem_stable_compatible(env,cem_lattice_rank(kn,env.δ),a,stable)
        end
        ok || return false
    end
    return true
end

cem_stable_compatible(env,q,a,stable) = stable(q) && cem_compatible(env,env.point_column[q],a)

#cluster index of each column in cols
function cem_clusters(env::CEMEnvelope,cols,stable = nothing)
    n = length(cols)
    parent = collect(1:n)
    root(i) = (while parent[i] != i; i = parent[i]; end; i)
    for a in 1:n, b in (a + 1):n
        root(a) == root(b) && continue
        if cem_samephase(env,cols[a],cols[b],stable)
            parent[root(b)] = root(a)
        end
    end
    roots = unique(root.(1:n))
    return [findfirst(isequal(root(i)),roots) for i in 1:n],length(roots)
end

function cem_numphases(env::CEMEnvelope,ws::CEMWorkspace,stable = nothing)
    cols = ws.B[ws.λ .> CEM_λTOL]
    length(cols) <= 1 && return 1
    return last(cem_clusters(env,cols,stable))
end

#phases (sorted by volume, solids last) of the current basis with weights λ
function cem_phases(env::CEMEnvelope,fluid,p,T,B,λ,stable = nothing)
    support = findall(>(CEM_λTOL),λ)
    cols = B[support]
    w = λ[support]
    labels,np = cem_clusters(env,cols,stable)
    N = length(env)
    comps = [zeros(N) for _ in 1:np]
    β = zeros(np)
    states = fill(CEM_LIQUID,np)
    wmax = zeros(np)
    for (j,c) in enumerate(cols)
        l = labels[j]
        β[l] += w[j]
        comps[l] .+= w[j] .* @view(env.x[:,c])
        if w[j] > wmax[l]
            wmax[l] = w[j]
            states[l] = env.state[c]
        end
    end
    vols = zeros(np)
    for l in 1:np
        comps[l] ./= β[l]
        s = states[l]
        vols[l] = s == CEM_SOLID ? NaN : volume(fluid,p,T,comps[l],phase = cem_state_symbol(s))
    end
    idx = sortperm(vols) #NaN (solids) last
    return comps[idx],β[idx],vols[idx],states[idx]
end

function cem_flash_result(env::CEMEnvelope,fluid,p,T,z,ws::CEMWorkspace,stable = nothing)
    comps,β,vols,states = cem_phases(env,fluid,p,T,ws.B,ws.λ,stable)
    g = dot(ws.λ,@view(env.g[ws.B])) + dot(z,env.gpure)
    vapour_idx = -1
    for i in eachindex(states)
        states[i] == CEM_VAPOUR && (vapour_idx = i) #largest volume
    end
    return FlashResult(comps,β,vols,FlashData(p,T,g,vapour_idx))
end

## phase diagram

"""
    CEMPhaseDiagram(;δ = nothing, equilibrium = :auto, polish = nothing, threaded = true)

Method to calculate the complete phase diagram at constant pressure and temperature with the convex envelope method (CEM) [1].

The reduced Gibbs energy of every aggregate state (liquid, vapour and, for `CompositeModel` with a `SolidHfus` solid model, pure solids) is evaluated on a lattice of the composition simplex with step `1/δ`.
The lower convex envelope of those points is calculated with Qhull. Its facets that connect points of different phases are the multiphase regions (tie-lines, tie-triangles, …).
Requires DirectQhull.jl (`using DirectQhull`).

## Keyword arguments
- `δ`: number of divisions of the composition axes. By default, `256`, `128` and `64` for binary, ternary and quaternary mixtures (as in [1]), and `32` for more components. The number of grid points is `binomial(δ + N - 1, N - 1)`.
- `equilibrium`: `:auto` considers liquid and vapour phases, `:lle` only liquid phases.
- `polish`: `nothing` (default) or a `TPFlashMethod` (e.g. `MichelsenTPFlash()` or `MultiPhaseTPFlash()`). If given, the phases of every multiphase facet are refined with that method, starting from the CEM phases. The refined result is only accepted if it lowers the Gibbs energy. Facets with solid phases are not refined.
- `threaded`: evaluate the grid and classify the grid points in parallel (uses `Threads.nthreads()` tasks).

## References
1. Göttl, Q., Rosen, N., & Burger, J. (2025). Convex envelope method for T, p flash calculations for mixtures with an arbitrary number of components and arbitrary aggregate states. Computers & Chemical Engineering, 203, 109326. [doi:10.1016/j.compchemeng.2025.109326](https://doi.org/10.1016/j.compchemeng.2025.109326)
"""
Base.@kwdef struct CEMPhaseDiagram <: ThermodynamicMethod
    δ::Union{Int,Nothing} = nothing
    equilibrium::Symbol = :auto
    polish::Union{Nothing,TPFlashMethod} = nothing
    threaded::Bool = true
end

"""
    CEMFacet

Multiphase facet of the convex envelope. Fields:
- `columns`: grid columns (indices into the envelope) of the facet vertices.
- `μ`: tangent plane of the facet, `μᵢ/RT` in the reference state of the model.
- `compositions`, `volumes`, `states`: phases in equilibrium (sorted by volume, solids last, with `NaN` volume). Without polishing, each phase composition is the mean of its facet vertices.
"""
struct CEMFacet
    columns::Vector{Int32}
    μ::Vector{Float64}
    compositions::Vector{Vector{Float64}}
    volumes::Vector{Float64}
    states::Vector{Symbol}
end

"""
    CEMDiagram

Result of [`phase_diagram`](@ref). Fields:
- `components`, `p`, `T`: system and conditions.
- `envelope`: grid and lower convex hull. `envelope.lattice ./ envelope.δ` are the grid compositions (one column per point).
- `facets`: vector of [`Clapeyron.CEMFacet`](@ref), the multiphase facets of the convex envelope.
- `point_facet`: for each grid point, the index of the multiphase facet that contains it (`0` if the point is a stable single phase).
- `equilibrium`: equilibrium type used to build the diagram.

A `CEMDiagram` can be used for fast flash calculations at the same conditions via `CEMFlash(diagram = diagram)`.
"""
struct CEMDiagram
    components::Vector{String}
    p::Float64
    T::Float64
    equilibrium::Symbol
    envelope::CEMEnvelope
    facets::Vector{CEMFacet}
    point_facet::Vector{Int32}
end

Base.show(io::IO,env::CEMEnvelope) = print(io,"CEMEnvelope(δ = ",env.δ,", ",size(env.lattice,2)," grid points, ",length(env.g)," columns, ",size(env.facets,2)," facets)")
Base.show(io::IO,d::CEMDiagram) = print(io,"CEMDiagram(",length(d.components)," components, p = ",d.p," Pa, T = ",d.T," K, δ = ",d.envelope.δ,", ",length(d.facets)," multiphase facets)")

function Base.show(io::IO,::MIME"text/plain",d::CEMDiagram)
    env = d.envelope
    nps = [length(f.compositions) for f in d.facets]
    print(io,"CEMDiagram(",length(d.components)," components, p = ",d.p," Pa, T = ",d.T," K, δ = ",env.δ,", ",size(env.lattice,2)," grid points)")
    print(io,"\n multiphase facets: ",length(d.facets))
    for np in sort(unique(nps))
        print(io,"\n  ",np,"-phase: ",count(isequal(np),nps))
    end
    print(io,"\n single phase grid points: ",count(iszero,d.point_facet))
end

"""
    phase_diagram(model, p, T, method = CEMPhaseDiagram())
    phase_diagram(model, p, T; kwargs...)

Calculates the complete phase diagram of `model` at pressure `p` `[Pa]` and temperature `T` `[K]`, over the whole composition space.
Returns a [`Clapeyron.CEMDiagram`](@ref), containing all multiphase regions (as facets of the convex envelope of the Gibbs energy) and the phase classification of every grid point.

Currently, the only method available is [`CEMPhaseDiagram`](@ref). The keyword arguments are passed to it.

## Example
```julia
model = UNIFAC(["water","hexane","ethanol"])
diagram = phase_diagram(model, 1e5, 298.15, CEMPhaseDiagram(δ = 64))
diagram.facets[1].compositions #compositions of the phases of the first multiphase facet
tp_flash2(model, 1e5, 298.15, [0.4,0.4,0.2], CEMFlash(diagram = diagram)) #flash with the stored envelope
```
"""
function phase_diagram(model::EoSModel,p,T;kwargs...)
    return phase_diagram(model,p,T,CEMPhaseDiagram(;kwargs...))
end

function phase_diagram(model::EoSModel,_p,_T,method::CEMPhaseDiagram)
    p,T = ustrip(_p,pressure),ustrip(_T,temperature)
    p,T = float(primalval(p)),float(primalval(T))
    N = length(model)
    N < 2 && throw(ArgumentError("phase_diagram: at least two components are required"))
    δ = something(method.δ,cem_default_δ(N))
    env = cem_envelope(model,p,T,δ,method.equilibrium,method.threaded)
    fluid = cem_models(model,p,T,fill(1/N,N),method.equilibrium)[2]
    M = size(env.lattice,2)
    walk_facet = zeros(Int32,M)
    #each chunk locates its lattice points in order, starting the walk from the facet of the previous point
    results = cem_foreach_chunk(idx -> cem_walk!(walk_facet,env,idx),M,method.threaded)
    #merge the facets found by every chunk
    ids = Dict{Vector{Int32},Int}()
    walk_cols = Vector{Int32}[]
    for (idx,list) in results
        newid = map(list) do key
            get!(ids,key) do
                push!(walk_cols,key)
                length(walk_cols)
            end
        end
        for k in idx
            walk_facet[k] != 0 && (walk_facet[k] = newid[walk_facet[k]])
        end
    end
    facet_cols,point_facet = cem_merge_phases(env,walk_cols,walk_facet)
    stable = CEMStability(env,point_facet)
    ws = CEMWorkspace(env)
    w = fill(1/N,N)
    facets = map(facet_cols) do cols
        ws.B .= cols
        z = env.x[:,cols]*w
        μ = cem_tangent_plane(env,cols)
        comps,_,vols,states = cem_phases(env,fluid,p,T,cols,w,stable)
        f = CEMFacet(cols,μ,comps,vols,cem_state_symbol.(states))
        method.polish === nothing ? f : cem_polish_facet(model,p,T,z,f,env,fluid,ws,stable,method.polish)
    end
    return CEMDiagram(component_list(model),p,T,method.equilibrium,env,facets,point_facet)
end

#second pass over the multiphase facets found by the walk: vertices that are the same phase but not lattice neighbours
#(see cem_samephase) are merged, points and facets that become single phase are removed.
function cem_merge_phases(env::CEMEnvelope,facet_cols,point_facet)
    N = length(env)
    stable = CEMStability(env,copy(point_facet))
    labels = [first(cem_clusters(env,cols,stable)) for cols in facet_cols]
    Binv = [inv(env.x[:,cols]) for cols in facet_cols]
    λ = zeros(N)
    newpf = zeros(Int32,length(point_facet))
    used = falses(length(facet_cols))
    for k in eachindex(point_facet)
        f = point_facet[k]
        f == 0 && continue
        mul!(λ,Binv[f],@view(env.x[:,env.point_column[k]]))
        l = labels[f]
        if length(unique(l[i] for i in 1:N if λ[i] > CEM_λTOL)) > 1
            newpf[k] = f
            used[f] = true
        end
    end
    newid = cumsum(used)
    for k in eachindex(newpf)
        newpf[k] != 0 && (newpf[k] = newid[newpf[k]])
    end
    return facet_cols[used],newpf
end

#classifies the lattice points idx: facet that contains each point, starting the walk from the facet of the previous point
function cem_walk!(point_facet,env::CEMEnvelope,idx)
    ws = CEMWorkspace(env)
    keys = Dict{Vector{Int32},Int}()
    list = Vector{Int32}[]
    key = similar(ws.B)
    for k in idx
        cem_locate!(ws,env,@view(env.x[:,env.point_column[k]]))
        if cem_numphases(env,ws) > 1
            sort!(key .= ws.B)
            id = get(keys,key,0)
            if id == 0
                fkey = copy(key) #key is a reused buffer
                push!(list,fkey)
                id = keys[fkey] = length(list)
            end
            point_facet[k] = id
        end
    end
    return idx,list
end

function cem_polish_facet(model,p,T,z,f::CEMFacet,env,fluid,ws,stable,method)
    ws.λ .= 1/length(z)
    res = cem_flash_result(env,fluid,p,T,z,ws,stable)
    res2 = cem_polish(model,p,T,z,res,method)
    res2 === res && return f
    np = numphases(res2,true)
    np == length(f.compositions) || return f
    idx = [i for i in 1:numphases(res2) if is_active_phase(res2,i)]
    vapour_idx = res2.data.vapour_idx
    states = [i == vapour_idx ? :vapour : :liquid for i in idx]
    return CEMFacet(f.columns,f.μ,[collect(res2.compositions[i]) for i in idx],res2.volumes[idx],states)
end

include("tp_flash/CEMtp_flash.jl")

export phase_diagram, CEMPhaseDiagram
