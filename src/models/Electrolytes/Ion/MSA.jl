struct MSAParam <: EoSParam
    sigma::SingleParam{Float64}
    charge::SingleParam{Float64}
end

abstract type MSAModel <: IonModel end

struct MSA{ϵ} <: MSAModel
    components::Array{String,1}
    icomponents::UnitRange{Int}
    params::MSAParam
    RSPmodel::ϵ
    references::Array{String,1}
end

@registermodel MSA

export MSA
"""
    MSA(solvents::Array{String,1},
        ions::Array{String,1};
        RSPmodel = ConstRSP,
        userlocations = String[],
        RSPmodel_userlocations = String[],
        verbose = false)

## Input parameters
- `sigma`: Single Parameter (`Float64`) - Hard-sphere diameter `[m]`
- `charge`: Single Parameter (`Float64`) - Charge `[-]`

## Input models
- `RSPmodel`: Relative Static Permittivity Model

## Description
This function is used to create a Mean Spherical Approximation model. The MSA term gives the excess Helmholtz energy to account for the electrostatic interactions between ions in solution.

## References
1. Blum, L. (1974). Solution of a model for the solvent‐electrolyte interactions in the mean spherical approximation, 61, 2129–2133.
"""
function MSA(solvents,ions; RSPmodel=ConstRSP, userlocations=String[], RSPmodel_userlocations=String[], verbose=false)
    components = deepcopy(ions)
    prepend!(components,solvents)
    icomponents = 1:length(components)
    params = getparams(components, ["Electrolytes/properties/charges.csv","properties/molarmass.csv"]; userlocations=userlocations,ignore_missing_singleparams=["sigma_born","charge"], verbose=verbose)
    if any(keys(params).=="b")
        params["b"].values .*= 3/2/N_A/π*1e-3
        params["b"].values .^= 1/3
        sigma = SingleParam("sigma",components,params["b"].values)
    else
        params["sigma"].values .*= 1E-10
        sigma = params["sigma"]
    end

    charge = params["charge"]

    packagedparams = MSAParam(sigma,charge)

    references = String[]

    init_RSPmodel = @initmodel RSPmodel(solvents,ions,userlocations = RSPmodel_userlocations, verbose = verbose)

    model = MSA(components, icomponents, packagedparams, init_RSPmodel, references)
    return model
end

function data(model::MSAModel, V, T, z)
    return dielectric_constant(model.RSPmodel, V, T, z)
end

function a_res(model::MSAModel, V, T, z, _data=@f(data))
    return a_ion(model, V, T, z, _data)
end

function a_ion(model::MSAModel, V, T, z, _data=@f(data))
    ϵ_r = _data
    σ = model.params.sigma.values
    Z = model.params.charge.values
    iions = model.icomponents[Z.!=0]

    if length(iions) == 0
        return zero(V+T+first(z))
    end
    σ = model.params.sigma.values
    Z = model.params.charge.values
    ϵ_r = _data
    ∑z = sum(z)
    ρ = N_A*sum(z)/V
    Γ = @f(screening_length,ϵ_r)
    Δ = 1-π*ρ/6*sum(z[i]*σ[i]^3 for i ∈ iions)/∑z
    Ω = 1+π*ρ/(2*Δ)*sum(z[i]*σ[i]^3/(1+Γ*σ[i]) for i ∈ iions)/∑z
    Pn = ρ/Ω*sum(z[i]*σ[i]*Z[i]/(1+Γ*σ[i]) for i ∈ iions)/∑z

    U_MSA = -e_c^2*V/(4π*ϵ_0*ϵ_r)*(Γ*ρ*sum(z[i]*Z[i]^2/(1+Γ*σ[i]) for i ∈ iions)/∑z + π/(2Δ)*Ω*Pn^2)
    return (U_MSA+Γ^3*k_B*T*V/(3π))/(N_A*k_B*T*sum(z))
end

function screening_length(model::MSAModel,V,T,z,ϵ_r = @f(data))
    σ = model.params.sigma.values
    Z = model.params.charge.values
    iions = model.icomponents[Z.!=0]

    ∑z = sum(z)
    ρ = N_A*sum(z)/V
    Δ = 1-π*ρ/6*sum(z[i]*σ[i]^3 for i ∈ @comps)/∑z
    yyy = 4π*e_c^2/(4π*ϵ_0*ϵ_r*k_B*T)*ρ
    
    Γold = sqrt(4π*e_c^2/(4π*ϵ_0*ϵ_r*k_B*T)*ρ) * sqrt(sum(z[i]*Z[i]^2 for i ∈ iions)/∑z)
    _0 = zero(Γold)
    iszero(Γold) && return _0
    Γnew = _0
    tol  = one(_0)
    iter = 1
    k1 = sqrt(π*e_c^2*ρ/(4π*ϵ_0*ϵ_r*k_B*T))
    while tol>1e-12 && iter < 100
        Ω = 1+π*ρ/(2*Δ)*sum(z[i]*σ[i]^3/(1+Γold*σ[i]) for i ∈ iions)/∑z
        Pn = ρ/Ω*sum(z[i]*σ[i]*Z[i]/(1+Γold*σ[i]) for i ∈ iions)/∑z
        #Q = @. (Z-σ^2*Pn*(π/(2Δ)))./(1+Γold*σ)
        ∑Q2x = _0
        for i ∈ iions
            Qi = (Z[i]-σ[i]^2*Pn*(π/(2Δ)))/(1+Γold*σ[i])
            ∑Q2x += z[i]*Qi^2
        end
        Γnew = k1*sqrt(∑Q2x/∑z)
        tol = abs(1-Γnew/Γold)
        Γold = Γnew
        iter += 1
    end

    return Γnew
end