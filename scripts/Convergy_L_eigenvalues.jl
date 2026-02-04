"""
    Julia code for Convergency of the Lindbladian eigenvalues
"""



####################################################################
println("Loading libraries")
using LinearAlgebra
using QuantumToolbox
using DelimitedFiles, NLsolve
using Random, Distributions, Dates, Polynomials, StatsBase 
using JLD, BenchmarkTools
println("Done")
####################################################################

println("")
println("")

####################################################################
println("Loading internal modules")
using Pkg
include("src/Open_2KPOs_QTB_functions.jl")
using .Coupled_KPOs_QTB_functions
println("Done")
####################################################################

println("")
println("")

####################################################################
println("Definitions")
N1, N2 = 15, 10 
# Paramters
p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ =  0., 1., 0., 5., 0., 1., 0., 5., 1.
Neff = 1.
κ1, κ2  =  0.025, 0.025
n_th1, n_th2 = 0.01, 0.01;
println("N1, N2 = $(N1), $(N2)")
println("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
println("κ1, κ2  = $(κ1), $(κ2)")
println("n_th1, n_th2  = $(n_th1), $(n_th2)")
println("Done")
####################################################################


println("Defining H and dissipators for N1 and N2")
#Full closed Hamiltonian - No symmetries
N = N1
H_N1 = H_full(p, N);

a1, ad1, a2, ad2 = tensor(destroy(N), qeye(N)), tensor(create(N), qeye(N)), tensor(qeye(N), destroy(N)), tensor(qeye(N), create(N));
n1, n2 = tensor(num(N), qeye(N)), tensor(qeye(N), num(N));
#Defines the disipators for the master equation as $√Γ_i L_i$
c_ops_N1 = [sqrt(κ1 * (n_th1 + 1)) * a1, sqrt(κ1 * n_th1) * ad1, sqrt(κ2 * (n_th2 + 1)) * a2, sqrt(κ2 * n_th2) * ad2];


N = N2
#Full closed Hamiltonian - No symmetries
H_N2 = H_full(p, N2);

a1, ad1, a2, ad2 = tensor(destroy(N), qeye(N)), tensor(create(N), qeye(N)), tensor(qeye(N), destroy(N)), tensor(qeye(N), create(N));
n1, n2 = tensor(num(N), qeye(N)), tensor(qeye(N), num(N));
#Defines the disipators for thvals_N2, vecs_N2 = eigen(Matrix(L_N2))e master equation as $√Γ_i L_i$
c_ops_N2 = [sqrt(κ1 * (n_th1 + 1)) * a1, sqrt(κ1 * n_th1) * ad1, sqrt(κ2 * (n_th2 + 1)) * a2, sqrt(κ2 * n_th2) * ad2];
println("Done")






