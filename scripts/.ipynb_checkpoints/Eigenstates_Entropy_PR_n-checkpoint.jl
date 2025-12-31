t = time()
using LinearAlgebra
using Pkg
Pkg.add(["QuantumToolbox"])
Pkg.add(["StatsBase"])
Pkg.add(["NLsolve"])
Pkg.add(["BenchmarkTools"])
using QuantumToolbox
using JLD
using DelimitedFiles
using StatsBase
Pkg.add("MKL")
using MKL
BLAS.set_num_threads(16)   # or try 16, depending on memory bandwidth
println("BLAS threads: ", BLAS.get_num_threads())


include("Closed_2KPOs_QTB_functions.jl")
using .Coupled_KPOs_QTB_functions

parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = Tuple(parse.(Float64,ARGS[2:end-2]))
Neff = parse(Float64,ARGS[end-1])
job =  parse(Int,ARGS[1])
#Definitions
N = parse(Int,ARGS[end])

println("Neff = $(Neff), N = $(N)")


function Entanglement_entropy_fock(ψ, N,ismixed = false)
    if ismixed
        ρ_A = zeros(N,N)
        for i in 0:N-1
            for j in 0:N-1
                for k in 0:N-1
                    el1 = (i*N+1) + k
                    el2 = (j*N+1) + k
                    ρ_A[i+1,j+1] += ψ[el1] * (ψ[el2]') 
                end
            end
        end
    else
        ψ_r = reshape(ψ, N, N)
        ρ_A = ψ_r * ψ_r' 
    end
    λ, v = eigen(ρ_A, sortby=real)
    λ = abs.(λ) 
    λ = λ[λ .> 1e-14];
    return -sum(λ .* log.(λ))
end


#Definitions
N = 100
Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
Neff = 3 
ps = [0.1, 1., 5.]
N_states=4000
Entropies = zeros(N_states,4)
n1s = zeros(N_states,4)
PR_fock = zeros(N_states,4)
div_len = 40
mean_Ss = zeros(div_len-1,4)
Es = zeros(N_states,4)
intv_Es = zeros(div_len-1,4)
n1_ = repeat(0:N-1, inner=N)

t = time()
for k in 1:length(ps)
    γ = ps[k]
    p = Δ1*Neff, K1, ξ11/sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12/sqrt(Neff), ξ22*Neff, γ*Neff
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);       

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
    
    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies);
    E = all_energies[sorted_indices];
    ψ = all_states[:, sorted_indices];
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E = E[1:N_states] ./ Neff^2;
    ψ = ψ[:,1:N_states];
    for i in 1:N_states
        Entropies[i,k] = Entanglement_entropy_fock(ψ[:,i],N)
        n1s[i,k] = sum(abs2.(ψ[:,i]) .* n1_)
        PR_fock[i,k] = ( 1 / sum(abs.(ψ[:,i]).^4, dims=1))[1] 
    end
    mean_S = Float64[]
    intv_E =  range(minimum(E),maximum(E)+1,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[:,k][index]))
    end
    mean_Ss[:,k] = mean_S
    Es[:,k] = E #.- E[1]
    intv_Es[:,k] = intv_E[1:end-1] 
end 

save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "Entropies", Entropies)
save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "Es", Es)
save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/n_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "n1s", n1s)
save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/PRfock_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "PR_fock", PR_fock)