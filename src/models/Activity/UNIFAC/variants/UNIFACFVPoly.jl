#a GC averaged UNIFAC.
struct UNIFACFVPolyCache <: EoSModel
    components::Vector{String}
    r::Vector{Float64}
    q::Vector{Float64}
    m::Vector{Float64}
    Mw::Vector{Float64}
end

UNIFACFVPolyCache(groups::GroupParam,params) = UNIFACFVPolyCache(groups,params.Q,params.R,params.Mw)

function UNIFACFVPolyCache(groups::GroupParam,Q,R,Mw)
    Mw = group_sum(groups,Mw.values)
    r = group_sum(groups,R.values) ./ Mw
    q = group_sum(groups,Q.values) ./ Mw
    m = group_sum(groups,nothing)
    
    return UNIFACFVPolyCache(groups.components,r,q,m,Mw)
end

struct UNIFACFVPolyParam <: EoSParam
    v::SingleParam{Float64}
    c::SingleParam{Float64}
    A::PairParam{Float64}
    R::SingleParam{Float64}
    Q::SingleParam{Float64}
    Mw::SingleParam{Float64}
end

abstract type UNIFACFVPolyModel <: ActivityModel end

struct UNIFACFVPoly{c<:EoSModel} <: UNIFACFVPolyModel
    components::Array{String,1}
    icomponents::UnitRange{Int}
    groups::GroupParam
    params::UNIFACFVPolyParam
    puremodel::EoSVectorParam{c}
    references::Array{String,1}
    UNIFACFVPoly_cache::UNIFACFVPolyCache
end

@registermodel UNIFACFVPoly
export UNIFACFVPoly

"""
    UNIFACFVPolyModel <: ActivityModel

    UNIFACFVPoly(components::Vector{String};
    puremodel = PR,
    userlocations = String[], 
    pure_userlocations = String[],
    verbose = false)

## Input parameters
- `v`: Single Parameter (`Float64`)  - specific volume of species
- `R`: Single Parameter (`Float64`)  - Normalized group Van der Vals volume
- `Q`: Single Parameter (`Float64`) - Normalized group Surface Area
- `A`: Pair Parameter (`Float64`, asymetrical, defaults to `0`) - Binary group Interaction Energy Parameter
- `Mw`: Single Parameter (`Float64`) - Molecular weight of groups

## Input models
- `puremodel`: model to calculate pure pressure-dependent properties

## Description
UNIFACFVPoly (UNIFAC Free Volume) activity model.

The Combinatorial part corresponds to an GC-averaged modified [`UNIQUAC`](@ref) model. The residual part iterates over groups instead of components.

```
Gᴱ = nRT(gᴱ(comb) + gᴱ(res) + gᴱ(FV))
```

Combinatorial part:
```
gᴱ(comb) = ∑[xᵢlog(Φᵢ) + 5qᵢxᵢlog(θᵢ/Φᵢ)]
θᵢ = qᵢxᵢ/∑qᵢxᵢ
Φᵢ = rᵢxᵢ/∑rᵢxᵢ
rᵢ = ∑Rₖνᵢₖ for k ∈ groups
qᵢ = ∑Qₖνᵢₖ for k ∈ groups
```
Residual Part:
```
gᴱ(residual) = -v̄∑XₖQₖlog(∑ΘₘΨₘₖ)
v̄ = ∑∑xᵢνᵢₖ for k ∈ groups,  for i ∈ components
Xₖ = (∑xᵢνᵢₖ)/v̄ for i ∈ components 
Θₖ = QₖXₖ/∑QₖXₖ
Ψₖₘ = exp(-(Aₖₘ + BₖₘT + CₖₘT²)/T)
```
Free-volume Part:
```
gᴱ(FV) = -v̄∑XₖQₖlog(∑ΘₘΨₘₖ)
v̄ = ∑∑xᵢνᵢₖ for k ∈ groups,  for i ∈ components
Xₖ = (∑xᵢνᵢₖ)/v̄ for i ∈ components 
Θₖ = QₖXₖ/∑QₖXₖ
Ψₖₘ = exp(-(Aₖₘ + BₖₘT + CₖₘT²)/T)
```

## References

"""
UNIFACFVPoly

function UNIFACFVPoly(components::Vector{String};
    puremodel = PR,
    userlocations = String[], 
    pure_userlocations = String[],
    verbose = false)
    
    params_species = getparams(components, ["Activity/UNIFAC/UNIFACFV/UNIFACFV_like.csv"]; userlocations=userlocations, verbose=verbose)
    
    groups = GroupParam(components, ["Activity/UNIFAC/UNIFACFV/UNIFACFV_groups.csv"]; verbose=verbose)

    params = getparams(groups, ["Activity/UNIFAC/ogUNIFAC/ogUNIFAC_like.csv", "Activity/UNIFAC/ogUNIFAC/ogUNIFAC_unlike.csv"]; userlocations=userlocations, asymmetricparams=["A"], ignore_missing_singleparams=["A"], verbose=verbose)
    A  = params["A"]
    R  = params["R"]
    Q  = params["Q"]
    Mw = params["Mw"]
    v  = params_species["volume"]
    c  = params_species["c"]
    icomponents = 1:length(components)
    _puremodel = init_puremodel(puremodel,components,pure_userlocations,verbose)
    packagedparams = UNIFACFVPolyParam(v,c,A,R,Q,Mw)
    references = String["10.1021/i260064a004"]
    cache = UNIFACFVPolyCache(groups,packagedparams)
    model = UNIFACFVPoly(components,icomponents,groups,packagedparams,_puremodel,references,cache)
    return model
end

function activity_coefficient(model::UNIFACFVPolyModel,V,T,z)
    _data=@f(data)
    return exp.(@f(lnγ_comb,_data)+ @f(lnγ_res,_data)+ @f(lnγ_FV,_data))
end

function data(model::UNIFACFVPolyModel,V,T,z)
    Mw = model.UNIFACFVPoly_cache.Mw
    x = z ./ sum(z)
    w = z.*Mw / sum(z.*Mw)
    return w,x
end

function lnγ_comb(model::UNIFACFVPolyModel,V,T,z,_data=@f(data))
    w,x = _data
    Mw = model.UNIFACFVPoly_cache.Mw
    r =model.UNIFACFVPoly_cache.r
    q =model.UNIFACFVPoly_cache.q
    Φ = w.*r.^(3/4)/dot(w,r.^(3/4))
    lnγ_comb = @. log(Φ/x)+(1-Φ)
    return lnγ_comb
end

function lnγ_res(model::UNIFACFVPolyModel,V,T,z,_data=@f(data))
    v  = model.groups.n_flattenedgroups
    _Ψ = @f(Ψ)
    lnΓ_ = @f(lnΓ,_Ψ)
    lnΓi_ = @f(lnΓi,_Ψ)
    lnγ_res_ =  [sum(v[i][k].*(lnΓ_[k].-lnΓi_[i][k]) for k ∈ @groups) for i ∈ @comps]
    return lnγ_res_
end

function lnΓ(model::UNIFACFVPolyModel,V,T,z,_Ψ = @f(Ψ))
    Mw = model.params.Mw.values
    Q = model.params.Q.values ./ Mw
    v  = model.groups.n_flattenedgroups
    x = z ./ sum(z)
    W = sum(v[i][:].*Mw*x[i] for i ∈ @comps) ./ sum(sum(v[i][k]*Mw[k]*x[i] for k ∈ @groups) for i ∈ @comps)
    θ = W.*Q / dot(W,Q)
    lnΓ_ = Mw.*Q.*(1 .-log.(sum(θ[m]*_Ψ[m,:] for m ∈ @groups)) .- sum(θ[m]*_Ψ[:,m]./sum(θ[n]*_Ψ[n,m] for n ∈ @groups) for m ∈ @groups))
    return lnΓ_
end

function lnΓi(model::UNIFACFVPolyModel,V,T,z,_Ψ = @f(Ψ))
    Mw = model.params.Mw.values
    Q = model.params.Q.values ./ Mw
    v  = model.groups.n_flattenedgroups
    W = [v[i][:].*Mw ./ sum(v[i][k]*Mw[k] for k ∈ @groups) for i ∈ @comps]
    θ = [W[i][:].*Q ./ sum(W[i][n]*Q[n] for n ∈ @groups) for i ∈ @comps]
    lnΓi_ = [Mw.*Q.*(1 .-log.(sum(θ[i][m]*_Ψ[m,:] for m ∈ @groups)) .- sum(θ[i][m]*_Ψ[:,m]./sum(θ[i][n]*_Ψ[n,m] for n ∈ @groups) for m ∈ @groups)) for i ∈ @comps]
    return lnΓi_
end

function Ψ(model::UNIFACFVPolyModel,V,T,z)
    A = model.params.A.values
    return @. exp(-A/T)
end

function lnγ_FV(model::UNIFACFVPolyModel,V,T,z,_data=@f(data))
    w,x = _data
    c = model.params.c.values
    b = 1.28
    v = model.params.v.values
    r = model.UNIFACFVPoly_cache.r

    v̄  = @. v/(15.17*b*r)
    v̄ₘ = sum(v.*w)/(15.17*b*sum(r.*w))

    return @. 3*c*log((v̄^(1/3)-1)/(v̄ₘ^(1/3)-1))-c*((v̄/v̄ₘ-1)/(1-v̄^(-1/3)))
end

function excess_g_SG(model::UNIFACFVPolyModel,p,T,z)
    lnγ = lnγ_SG(model,p,T,z)
    return sum(z[i]*R̄*T*lnγ[i] for i ∈ @comps)
end

function excess_g_res(model::UNIFACFVPolyModel,p,T,z)
    lnγ = lnγ_res(model,p,T,z)
    return sum(z[i]*R̄*T*lnγ[i] for i ∈ @comps)
end