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


function r_mean(E)
    S = E[2:end] - E[1:end-1]
    r = S[2:end] ./ S[1:end-1]
    r_inv = S[1:end-1] ./ S[2:end]
    r_n = min.(r, r_inv)
    return mean(r_n)
end

E_thr = 0
N_p = N_Δ = 50
N_states = 1000
r_means_b0, r_means_a0, r_means_all = zeros(N_p, N_p), zeros(N_p, N_p), zeros(N_p, N_p)
ps = range(0.01, 10.0, length=N_p)
Δs = range(0.01, 10.0, length=N_Δ)
#Δs = range(3.0, 7.0, length=N_Δ)
t = time()


for k in (1*(job-1)+1):(1*job)
    for j in 1:N_p
        p = Δs[k]*Neff, K1, ξ11*sqrt(Neff), ξ21*Neff, Δs[k]*Neff, K2, ξ12*sqrt(Neff), ξ22*Neff, ps[j]*Neff
        #p = Δ1*Neff, K1, ξ11*sqrt(Neff), Δs[k]*Neff, Δ2*Neff, K2, ξ12*sqrt(Neff), Δs[k]*Neff, ps[j]*Neff

        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);
        E_S_even, E_A_even, E_S_odd, E_A_odd = real.(E_S_even)[1:N_states], real.(E_A_even)[1:N_states], real.(E_S_odd)[1:N_states], real.(E_A_odd)[1:N_states]

        E_S_even_b0 =  E_S_even[E_S_even/Neff^2 .< E_thr]
        E_A_even_b0 =  E_A_even[E_A_even/Neff^2 .< E_thr]
        E_S_odd_b0 =  E_S_odd[E_S_odd/Neff^2 .< E_thr]
        E_A_odd_b0 =  E_A_odd[E_A_odd/Neff^2 .< E_thr]
        E_S_even_a0 =  E_S_even[E_S_even/Neff^2 .> E_thr]
        E_A_even_a0 =  E_A_even[E_A_even/Neff^2 .> E_thr]
        E_S_odd_a0 =  E_S_odd[E_S_odd/Neff^2 .> E_thr]
        E_A_odd_a0 =  E_A_odd[E_A_odd/Neff^2 .> E_thr]


        r1_b0, r2_b0 = r_mean(E_S_even_b0), r_mean(E_A_even_b0)
        r3_b0, r4_b0 = r_mean(E_S_odd_b0), r_mean(E_A_odd_b0)
        r_means_b0[k,j] = mean([r1_b0,r2_b0,r3_b0,r4_b0])
        r1_a0, r2_a0 = r_mean(E_S_even_a0), r_mean(E_A_even_a0)
        r3_a0, r4_a0 = r_mean(E_S_odd_a0), r_mean(E_A_odd_a0)
        r_means_a0[k,j] = mean([r1_a0,r2_a0,r3_a0,r4_a0])
        r1, r2 = r_mean(E_S_even), r_mean(E_A_even)
        r3, r4 = r_mean(E_S_odd), r_mean(E_A_odd)
        r_means_all[k,j] = mean([r1,r2,r3,r4])
    end
end


save("data/r_means/r_means_p_$(parameters)_N_$(N)_Neff_$(Neff)_job_$(job)_Δ.jld", "r_means_all", r_means_all)
save("data/r_means/r_means_a0_p_$(parameters)_N_$(N)_Neff_$(Neff)_job_$(job)_Δ.jld", "r_means_a0", r_means_a0)
save("data/r_means/r_means_b0_p_$(parameters)_N_$(N)_Neff_$(Neff)_job_$(job)_Δ.jld", "r_means_b0", r_means_b0)
