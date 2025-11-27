abstract type PitzerModel <: IonModel end

struct Pitzer{ϵ} <: PitzerModel
    components::Array{String,1}
    RSPmodel::ϵ
    references::Array{String,1}
end

"""
    Pitzer(solvents::Array{String,1},
        ions::Array{String,1};
        RSPmodel = ConstRSP,
        userlocations = String[],
        RSPmodel_userlocations = String[],
        verbose = false)

## Input models
- `RSPmodel`: Relative Static Permittivity Model

## Description
This function is used to create a Pitzer model. The Debye-Hückel term gives the excess Helmholtz energy to account for the electrostatic interactions between ions in solution.

## References
1. Pitzer, K. S.: Thermodynamics of Electrolytes. I. Theoretical Basis and General Equations, J. Phys. Chem. 77 (1973) 268–277, DOI: [10.1021/j100621a026](https://doi.org/10.1021/j100621a026).
"""
Pitzer

export Pitzer
function Pitzer(solvents, ions; RSPmodel=ConstRSP, userlocations=String[], RSPmodel_userlocations=String[], verbose=false)

    components = deepcopy(ions)
    prepend!(components,solvents)

    references = String[]
    init_RSPmodel = @initmodel RSPmodel(solvents,ions,userlocations = RSPmodel_userlocations, verbose = verbose)

    model = Pitzer(components, init_RSPmodel,references)
    return model
end

function a_res(model::PitzerModel, V, T, z, iondata)    # excess Helmholtz energy
    return a_pitzer
end