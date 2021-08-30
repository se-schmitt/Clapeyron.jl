abstract type MHV1RuleModel <: MixingRule end

struct MHV1Rule{γ} <: MHV1RuleModel
    components::Array{String,1}
    activity::γ
end

has_sites(::Type{<:MHV1RuleModel}) = false
has_groups(::Type{<:MHV1RuleModel}) = false
built_by_macro(::Type{<:MHV1RuleModel}) = false

function Base.show(io::IO, mime::MIME"text/plain", model::MHV1Rule)
    return eosshow(io, mime, model)
end

function Base.show(io::IO, model::MHV1Rule)
    return eosshow(io, model)
end

export MHV1Rule
function MHV1Rule(components::Vector{String}; activity = Wilson, userlocations::Vector{String}=String[],activity_userlocations::Vector{String}=String[], verbose::Bool=false)
    init_activity = activity(components;userlocations = activity_userlocations,verbose)
    
    model = MHV1Rule(components, init_activity)
    return model
end

function mixing_rule(model::RKModel,V,T,z,mixing_model::MHV1RuleModel,α,a,b)
    n = sum(z)
    x = z./n
    invn2 = (one(n)/n)^2
    g_E = excess_gibbs_free_energy(mixing_model.activity,1e5,T,z) / n
    b̄ = dot(z,Symmetric(b),z) * invn2
    ā = b̄*R̄*T*(sum(x[i]*a[i,i]*α[i]/b[i,i]/(R̄*T) for i ∈ @comps)-1/0.593*(g_E/(R̄*T)+sum(x[i]*log(b̄/b[i,i]) for i ∈ @comps)))
    return ā,b̄
end

function mixing_rule(model::PRModel,V,T,z,mixing_model::MHV1RuleModel,α,a,b)
    n = sum(z)
    x = z./n
    invn2 = (one(n)/n)^2
    g_E = excess_gibbs_free_energy(mixing_model.activity,1e5,T,z) / n
    b̄ = dot(z,Symmetric(b),z) * invn2
    ā = b̄*R̄*T*(sum(x[i]*a[i,i]*α[i]/b[i,i]/(R̄*T) for i ∈ @comps)-1/0.53*(g_E/(R̄*T)+sum(x[i]*log(b̄/b[i,i]) for i ∈ @comps)))
    return ā,b̄
end

is_splittable(::MHV1Rule) = true