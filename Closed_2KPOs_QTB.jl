using LinearAlgebra
using PyPlot
using PyCall
using BenchmarkTools, Roots
using LaTeXStrings
using QuantumToolbox
using DelimitedFiles, NLsolve
using Random, Distributions, Dates, Polynomials, StatsBase 
using JLD


include("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Closed_2KPOs_QTB_functions.jl")
using .Coupled_KPOs_QTB_functions
pygui(true)


# -------------------------------- Energies ---------------------------------------------------
    #Definitions
    N = 10
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0.04068266906056494, 1., 0., 1.571424199455252, 0.04068266906056494, 1., 0., 3.571424199455252, 0.40825486855662213;
   
    E_even, ψ_even, E_odd, ψ_odd  = Coupled_kerr_equiv(p, N);
   
    #ψ = (ψ_S_even[:,1] - ψ_S_odd[:,1])/sqrt(2)
    N_Q = 70 #dimension of the Q function N_Q^2
    q1vals, p1vals, q2vals, p2vals = range(-5,5, length=N_Q),range(-5,5, length=N_Q),range(-5,5, length=N_Q),range(-5,5, length=N_Q)
    
    #Individual Husimi plots 
        t = time()
        Qgrid = Q_function_grid_q1q2_full(ψ[1], q1vals, q2vals, N);
        imshow(Qgrid,origin="lower",cmap="magma",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]));
        time() - t
        #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
        xticks([])
        yticks([])
        title("E = $(round(Es[S_n][k], digits=3))")
    #

    fig = figure(figsize=(12,12), layout= "constrained")
    gs = fig.add_gridspec(5,5)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    #Multiple Husimi
    count=1
    for i in 1:5
        for j in 1:5
                ax = fig.add_subplot(element(i-1,j-1))
                Qgrid = Q_function_grid_q1q2_full(ψ[count], q1vals, q2vals, N)
                #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                imshow(Qgrid,origin="lower",cmap="magma",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                if j ==1
                   ylabel(L"q_{2}", fontsize=15)
                else
                    yticks([])
                end
                if i == 5
                   xlabel(L"q_{1}", fontsize=15)
                else
                    xticks([])
                end
                title("E = $(round(real(E[count]), digits=4))")
            count+=1 
        end
    end
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Qcomputing/Quantum/Husimis_N_$(N)_p_$(p).png")

    
###


# -------------------------------- Convergency of E and ψs ---------------------------------------------------


    function crit_energies2(parameters,n_g,x_g)
        # Root-finding function
        function find_roots_EqM(parameters, initial_guesses)
            unique_roots = []
            for k in 1:n_g
                f!(F, u) = (F .= EqM_2(u, parameters))
                result = nlsolve(f!, initial_guesses[k,:])
                root = result.zero
                
                # Check uniqueness
                if !any(x -> norm(x - root) < 1e-2, unique_roots)
                    push!(unique_roots, root)
                end
            end
            return unique_roots
        end
        # Generate multiple initial guesses with Float64 values
        Random.seed!(123)
        d = Uniform(-x_g, x_g)
        initial_guesses = rand(d,n_g,4)
        roots_ = find_roots_EqM(parameters, initial_guesses)
        #println("Unique Roots of EqM = 0: ", roots)
        R = length(roots_)
        E_cl = zeros(R)
        λs =  zeros(Complex,R, 4)
        s_λs = []
        for i in 1:R
            E_cl[i] = H_class(roots_[i], parameters)
            λ, v = eigen(Jacobian_qp(roots_[i], parameters))
            λs[i,:] = λ
            push!(s_λs, classify_fixed_point(λ)) 
        end
        
        sorted_indices = sortperm(real.(E_cl));  # Sort by absolute distance

        return roots_[sorted_indices],E_cl[sorted_indices], λs[sorted_indices,:], s_λs[sorted_indices]
    end

    #Convergency for EquivKPOs without linear drive
        N = 20;
        parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 1.0);
        Neff = 10
        p = Δ1*Neff, K1, ξ11/sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12/sqrt(Neff), ξ22*Neff, γ*Neff
        #p = (1.0, 1.0, 0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 1.0) - 1400 per S Energy / same for states
        #p = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 1.0) - 1400 per S Energy / same for states
        #p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 20.0) - 1200 per S Energy / same for states
        #p = (0.0, 1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 10.0, 1.0) - 1200 per S Energy / same for states
        #p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 5.0)Neff=5 - 400 per S Energy / same for states
        #p = (0.0, 1.0, 0.0, 10.0, 0.0, 1.0, 0.0, 10.0, 1.0)Neff=5 - 6 per S Energy / same for states
        #p = (10.0, 1.0, 0.0, 10.0, 10.0, 1.0, 0.0, 10.0, 10.0)Neff=5 - 6 per S Energy / same for states
        #p = (5.0, 1.0, 0.0, 5.0, 5.0, 1.0, 0.0, 5.0, 5.0)Neff=3 - 700 per S Energy / same for states

        #p = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 10.0) Neff=5 - 1800/1371 per S Energy / same for states N=200
        #p = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 10.0) Neff=10 -73 per S Energy / same for states N=200
        


        t = time()
        #E_conv, state_conv = Convergency_test(p, 100,30);
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, 10);
        E_S_even2, ψ_S_even2, E_A_even2, ψ_A_even2, E_S_odd2, ψ_S_odd2, E_A_odd2, ψ_A_odd2  = Coupled_kerr_equiv(p, 20);
        time() - t
        CE = crit_energies2(parameters, 100, 10)

        E_S_even, E_A_even, E_S_odd, E_A_odd = real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd)
        E_S_even2, E_A_even2, E_S_odd2, E_A_odd2 = real.(E_S_even2), real.(E_A_even2), real.(E_S_odd2), real.(E_A_odd2)

        E_S_even2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(300)_SE.dat")[:,1]
        E_A_even2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(300)_AE.dat")[:,1]
        E_S_odd2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(300)_SO.dat")[:,1]
        E_A_odd2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(300)_AO.dat")[:,1]
        
        E_S_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(200)_SE.dat")[:,1]
        E_A_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(200)_AE.dat")[:,1]
        E_S_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(200)_SO.dat")[:,1]
        E_A_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(200)_AO.dat")[:,1]
        
        se = abs.(E_S_even2[1:length(E_S_even)] - E_S_even)
        se
        n_se1 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1507
        se = abs.(E_A_even2[1:length(E_A_even)] - E_A_even)
        n_se2 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1445
        se = abs.(E_S_odd2[1:length(E_S_odd)] - E_S_odd)
        n_se3 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1475
        se = abs.(E_A_odd2[1:length(E_A_odd)] - E_A_odd)
        n_se4 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1475
        n_se = n_se1 + n_se2 + n_se3 + n_se4 # n_se = 5902

        ψ_S_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_SE.dat")
        ψ_A_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_AE.dat")
        ψ_S_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_SO.dat")
        ψ_A_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_AO.dat")

        ψ_S_even2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(300)_SE.dat")
        ψ_S_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_SE.dat")
        PR2 = vec( 1 ./ sum(abs.(ψ_S_even2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_S_even).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        
        
        ψ_A_even2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(300)_AE.dat")
        ψ_S_odd2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(300)_SO.dat")
        ψ_A_odd2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(300)_AO.dat")
     
        PR2 = vec( 1 ./ sum(abs.(ψ_S_even2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_S_even).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        bit_se = se .< 1e-1
        bit_se_inv = .!bit_se
        ψ_S_even_conv = ψ_S_even[:,bit_se]
        ψ_S_even_Nconv = ψ_S_even[:,bit_se_inv]
        E_S_even_conv = E_S_even[bit_se]
        E_S_even_Nconv = E_S_even[bit_se_inv]

        n_se1 = (findall(x -> x > 1e-1, se))  
        PR2 = vec( 1 ./ sum(abs.(ψ_A_even2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_A_even).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        n_se2 = (findall(x -> x > 1e-1, se))[1] - 1 
        PR2 = vec( 1 ./ sum(abs.(ψ_S_odd2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_S_odd).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        n_se3 = (findall(x -> x > 1e-1, se))[1] - 1 
        PR2 = vec( 1 ./ sum(abs.(ψ_A_odd2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_A_odd).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        n_se4 = (findall(x -> x > 1e-1, se))[1] - 1 
        n_se = n_se1 + n_se2 + n_se3 + n_se4 # n_se = 5902


        
        state_conv = n_se

    ###    
    
    #Convergency for Non-equiv KPOs without linear drive but
        N = 100;
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(1.0, 1.0, 0.0, 10.0, 1.0, 1.0, 0.0, 3.0, 1.0);
        #p = (1.0, 1.0, 0.0, 10.0, 1.0, 1.0, 0.0, 3.0, 1.0), 2500 per S Energy / same for states;
        
        t = time()
        E_even, ψ_even, E_odd, ψ_odd = Coupled_kerr(p, N);
        time() - t

        E_even[1]
        roots, cE, λs, s_λ = crit_energies(p);
        cE

        E_even2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_(10.0, 10.0, 50.0, 50.0, 10.0)_N_230_SE.dat")
        E_odd2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_(10.0, 10.0, 50.0, 50.0, 10.0)_N_230_SO.dat")
        E_even2[1]
        se = abs.(E_even2[1:length(E_even)] - E_even)
        n_se1 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1507
        se = abs.(E_odd2[1:length(E_odd)] - E_odd)
        n_se3 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1475
        n_se = n_se1 + n_se2  # n_se = 5902


        ψ_even2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_E.dat")
        ψ_odd2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200)_O.dat")
     
        PR2 = vec( 1 ./ sum(abs.(ψ_even2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_even).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        n_se1 = (findall(x -> x > 1e-3, se))[1] - 1 
        PR2 = vec( 1 ./ sum(abs.(ψ_odd2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ_odd).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        n_se4 = (findall(x -> x > 1e-3, se))[1] - 1 
        n_se = n_se1 + n_se2 # n_se = 5902

    ###

    #Convergency for KPOs with linear drive but
        N = 100;
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(0.0, 1.0, 10.0, 5.0, 0.0, 1.0, 10.0, 5.0, 1.0);
        
        t = time()
        E, ψ = Coupled_kerr(p, N);
        time() - t

        E[1]
        roots, cE, λs, s_λ = crit_energies(p);
        cE

        E2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(200).dat")

        E2[1]
        se = abs.(E2[:,1][1:length(E)] - E)
        n_se1 = (findall(x -> x > 1e-3, se))[1] - 1 #n = 1507

        
        ψ2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(200).dat")
  
        PR2 = vec( 1 ./ sum(abs.(ψ2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ).^4, dims=1) )
        se = abs.(PR2[1:length(PR)] - PR)
        n_se1 = (findall(x -> x > 1e-3, se))[1] - 1 

        
    ###
###

# -------------------------------- Statistics E  ---------------------------------------------------

    function crit_energies2(parameters,n_g,x_g)
        # Root-finding function
        function find_roots_EqM(parameters, initial_guesses)
            unique_roots = []
            for k in 1:n_g
                f!(F, u) = (F .= EqM_2(u, parameters))
                result = nlsolve(f!, initial_guesses[k,:])
                root = result.zero
                
                # Check uniqueness
                if !any(x -> norm(x - root) < 1e-2, unique_roots)
                    push!(unique_roots, root)
                end
            end
            return unique_roots
        end
        # Generate multiple initial guesses with Float64 values
        Random.seed!(123)
        d = Uniform(-x_g, x_g)
        initial_guesses = rand(d,n_g,4)
        roots_ = find_roots_EqM(parameters, initial_guesses)
        #println("Unique Roots of EqM = 0: ", roots)
        R = length(roots_)
        E_cl = zeros(R)
        λs =  zeros(Complex,R, 4)
        s_λs = []
        for i in 1:R
            E_cl[i] = H_class(roots_[i], parameters)
            λ, v = eigen(Jacobian_qp(roots_[i], parameters))
            λs[i,:] = λ
            push!(s_λs, classify_fixed_point(λ)) 
        end
        
        sorted_indices = sortperm(real.(E_cl));  # Sort by absolute distance

        return roots_[sorted_indices],E_cl[sorted_indices], λs[sorted_indices,:], s_λs[sorted_indices]
    end

    N = 10;
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 5.0);
    Neff = 1
    p = Δ1*Neff, K1, ξ11*sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12*sqrt(Neff), ξ22*Neff, γ*Neff
    #p = (1.0, 1.0, 0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 1.0) - 1400 per S Energy / same for states
    #p = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 1.0) - 1400 per S Energy / same for states
    #p = (1.0, 1.0, 0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 1.0) - 1400 per S Energy / same for states
    #p = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 1.0) - 1400 per S Energy / same for states
    t = time()
    #E_conv, state_conv = Convergency_test(p, 100,30);
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);
    
    
    time() - t
    E_S_even, E_A_even, E_S_odd, E_A_odd = real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd)
    E_S_even
    #Integrable Tail
    p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, 1.
    n_p = 100
    p_ = range(.01, 13., length = n_p)
    ns_job = 1:2:101
    λ_tail = 1e-1
    step = 0.1
    Es_tail = fill(NaN, 100)
    no_data = []
    for job in 1:length(ns_job)
        try 
            E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_γ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
            GS = zeros(2)
            count = 1
            #for i in ns_job[job]:(ns_job[job+1]-1)
            #    roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
            #    GS[count] = cE[1]
            #    count+=1
            #end
            Es_tail[ns_job[job]:ns_job[job+1]-1] = E_tail 
        catch
            println("Missing job $(job)")
            push!(no_data,job)
        end
    end
    p_[77]
    p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
    CE = crit_energies(p)  
    
    E_thr = -0#CE[2][1] + Es_tail[77]
    E_S_even2 =  E_S_even[E_S_even/Neff^2 .< E_thr]
    E_A_even2 =  E_A_even[E_A_even/Neff^2 .< E_thr]
    E_S_odd2 =  E_S_odd[E_S_odd/Neff^2 .< E_thr]
    E_A_odd2 =  E_A_odd[E_A_odd/Neff^2 .< E_thr]

    #Spacing Distribution
        function WD(s)
        return (π*s/2)*exp(-π*s^2/4)
        end
        function Brody(s,β)
        b = (gamma.((β.+2)./(β.+1))).^(β.+1)
        return (β.+1).*b.*(s.^β).*exp.(-b.*s.^(β.+1))
        end
        function general_polynomial_model(x, p)
        sum(p[i] .* x.^(i-1) for i in 1:length(p))
        end
        function unfolded(E)
            S = abs.(E[2:end] - E[1:end-1])
            g_n = 10 #number of E per group
            N_g = Int(floor(size(S)[1] / g_n)) # Number of group/s
            ξ = zeros(length(S))
            for i in 1:N_g
                i_n = g_n*(i-1)+1
                f_n = g_n*(i)
                s_mean = mean(S[i_n:f_n])
                ξ[i_n:f_n] = S[i_n:f_n]/s_mean
            end
            return ξ
        end
        ξ1 = unfolded(E_S_even)
        ξ2 = unfolded(E_A_even)
        ξ3 = unfolded(E_S_odd)
        ξ4 = unfolded(E_A_odd)
        ξ = vcat(ξ1,ξ2,ξ3,ξ4)
        hist(ξ, bins=20,density=true)
        x = range(0,6,length=100)
        plot(x,exp.(-x))
        plot(x,WD.(x))
    ###


    ##Average ratio
        function r_mean(E)
            S = E[2:end] - E[1:end-1]
            r = S[2:end] ./ S[1:end-1]
            r_inv = S[1:end-1] ./ S[2:end]
            r_n = min.(r, r_inv)
            return mean(r_n)
        end
        r1 = r_mean(E_S_even)
        r2 = r_mean(E_A_even)
        r3 = r_mean(E_S_odd)
        r4 = r_mean(E_A_odd)
        r = mean([r1,r2,r3,r4])
        println(r)
        #0.39 P and 0.54 WD
        #GAMMA 5, DELTA=0  <0 0.41   > 0 .46 
    ###
###

# -------------------------------- Statistics E vs γ ---------------------------------------------------

    ## CODE
        function r_mean(E)
            S = E[2:end] - E[1:end-1]
            r = S[2:end] ./ S[1:end-1]
            r_inv = S[1:end-1] ./ S[2:end]
            r_n = min.(r, r_inv)
            return mean(r_n)
        end
        N = 100;
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0);
        Neff = 4
        E_thr = -0#CE[2][1] + Es_tail[77]
        N_p = 100
        N_states = 2000
        r_means= zeros(N_p)
        ps = range(11.01, 21.0, length=N_p)

        for j in 1:N_p
            p = Δ1*Neff, K1, ξ11*sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12*sqrt(Neff), ξ22*Neff, ps[j]*Neff
            #E_conv, state_conv = Convergency_test(p, 100,30);
            E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);
            E_S_even, E_A_even, E_S_odd, E_A_odd = real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd)    
            #E_S_even2 =  E_S_even[E_S_even/Neff^2 .< E_thr]
            #E_A_even2 =  E_A_even[E_A_even/Neff^2 .< E_thr]
            #E_S_odd2 =  E_S_odd[E_S_odd/Neff^2 .< E_thr]
            #E_A_odd2 =  E_A_odd[E_A_odd/Neff^2 .< E_thr]
            E_S_even2 =  E_S_even[1:N_states]
            E_A_even2 =  E_A_even[1:N_states]
            E_S_odd2 =  E_S_odd[1:N_states]
            E_A_odd2 =  E_A_odd[1:N_states]
            r1 = r_mean(E_S_even2)
            r2 = r_mean(E_A_even2)
            r3 = r_mean(E_S_odd2)
            r4 = r_mean(E_A_odd2)
            r_means[j] = mean([r1,r2,r3,r4])   
        end
        fig = figure(figsize=(7,7), layout= "constrained");
        gs = fig.add_gridspec(1,1);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);

        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

        ax = fig.add_subplot(element(0,0))
        plot(ps, r_means, marker = markers_[1], color=colors_[1])
        ylabel(L"r", fontsize=20);
        xlabel(L"γ", fontsize=20);    
        fig.suptitle("Neff, Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Neff, Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);

        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_$(N)_γ_Neff_$(Neff)_$(ps[1])_whole_spectrum.png")
    ###


    ### PLots γ x Δ
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ  = 0., 1., 0., 5., 0., 1., 0., 5., 0. # For γ
        N_r = 50
        N=100
        Neff=2.0
        r_means = zeros(N_r,N_r)
        r_means_a = zeros(N_r,N_r)
        r_means_b = zeros(N_r,N_r)
        for job in 1:50
            try
                r_means += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job)_Δ.jld")["r_means_all"];
                r_means_a += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_a0_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job)_Δ.jld")["r_means_a0"];
                r_means_b += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_b0_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job)_Δ.jld")["r_means_b0"];

            catch
                println("$(job)")
            end
        end

        im = imshow(r_means,origin="lower",cmap="OrRd",extent=(0.01, 10.0, 0.01,10.0),vmin = 0.39, vmax=0.53)
        im = imshow(r_means_a,origin="lower",cmap="OrRd",extent=(0.01, 10.0, 0.01,10.0),vmin = 0.39, vmax=0.53)
        im = imshow(r_means_b,origin="lower",cmap="OrRd",extent=(0.01, 10.0, 0.01,10.0),vmin = 0.39, vmax=0.53)
        #cbar = colorbar(im)

        cbar = colorbar(im,ticks=[0.39, (0.53 - 0.38)/2 + 0.38,0.53], shrink=0.9)
        cbar.set_label("⟨r⟩", fontsize=20)
        xlabel("γ", fontsize=20)
        ylabel("Δ", fontsize=20)
        xticks([0,1,2,3,4,5],fontsize=15)
        yticks([0,1,2,3,4,5],fontsize=15)
        cbar.ax.tick_params(axis="y", labelsize=15)
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_spectrum_high.png")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_spectrum_low.png")
    ###

    
    ### PLots γ x ξ2
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ  = 0., 1., 0., 5., 0., 1., 0., 5., 0. # For γ
        N_r = 50
        N=100
        Neff=3.0
        r_means = zeros(N_r,N_r)
        r_means_a = zeros(N_r,N_r)
        r_means_b = zeros(N_r,N_r)
        for job in 1:50
            try
                r_means += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job).jld")["r_means_all"];
                r_means_a += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_a0_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job).jld")["r_means_a0"];
                r_means_b += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_b0_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job).jld")["r_means_b0"];

            catch
                println("$(job)")
            end
        end

        im = imshow(r_means,origin="lower",cmap="OrRd",extent=(0.01, 5.0, 2.,7.0),vmin = 0.39, vmax=0.53)
        im = imshow(r_means_a,origin="lower",cmap="OrRd",extent=(0.01, 5.0, 2.,7.0),vmin = 0.39, vmax=0.53)
        im = imshow(r_means_b,origin="lower",cmap="OrRd",extent=(0.01, 5.0, 2.,7.0),vmin = 0.39, vmax=0.53)

        #cbar = colorbar(im)

        cbar = colorbar(im,ticks=[0.39, (0.53 - 0.38)/2 + 0.38,0.53])#, shrink=0.9)
        cbar.set_label("⟨r⟩", fontsize=20)
        xlabel("γ", fontsize=20)
        ylabel(L"ξ_{2}", fontsize=20)
        xticks([0,1,2,3,4,5],fontsize=15)
        yticks([0,1,2,3,4,5],fontsize=15)
        cbar.ax.tick_params(axis="y", labelsize=15)
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_spectrum_high_ξ2.png")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_spectrum_low_ξ2.png")
    ###

    
    ### PLots γ 
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ  = 0., 1., 0., 5., 0., 1., 0., 5., 0. # For γ
        N_r = 50
        N=200
        Neff=5.0
        r_means = zeros(N_r)
        r_means_a = zeros(N_r)
        r_means_b = zeros(N_r)
        for job in 1:25
            try
                r_means += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job).jld")["r_means_all"];
                r_means_a += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_a0_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job).jld")["r_means_a0"];
                r_means_b += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/r_means/r_means_b0_p_$(p)_N_$(N)_Neff_$(Neff)_job_$(job).jld")["r_means_b0"];

            catch
                println("$(job)")
            end
        end
        fig = figure(figsize=(10,7), layout= "constrained");
        title("Neff, Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Neff, Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);
    
        plot(range(0.01, 10.0, length=N_r), r_means, "o-", label = "All States")
        plot(range(0.01, 10.0, length=N_r), r_means_a, "o-",label = "Above E = 0")
        plot(range(0.01, 10.0, length=N_r), r_means_b, "o-",label = "Below E = 0")
        plot(range(0.01, 10.0, length=N_r), range(0.39, 0.39, length=N_r), color="black")
        plot(range(0.01, 10.0, length=N_r), range(0.53, 0.53, length=N_r), color="black")


        xlabel("γ", fontsize=20)
        ylabel("⟨r⟩", fontsize=20)
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_spectrum_γ.png")
    ###

###

# -------------------------------- Statistics E vs γ vs Δ ---------------------------------------------------

    
    function r_mean(E)
        S = E[2:end] - E[1:end-1]
        r = S[2:end] ./ S[1:end-1]
        r_inv = S[1:end-1] ./ S[2:end]
        r_n = min.(r, r_inv)
        return mean(r_n)
    end
    N = 10;
    K1, ξ11, ξ21, K2, ξ12, ξ22 = (1.0, 0.0, 5.0, 1.0, 0.0, 5.0);
    Neff = 3
    E_thr = 0#CE[2][1] + Es_tail[77]
    N_p = N_Δ = 100
    N_states = 1000
    r_means_b0, r_means_a0, r_means_all = zeros(N_p, N_p), zeros(N_p, N_p), zeros(N_p, N_p) 
    ps = range(0.01, 5.0, length=N_p)
    Δs = range(0.01, 5.0, length=N_Δ)
    t = time()
    #for k in 1:N_Δ
        #for j in 1:N_p
            p = Δs[k]*Neff, K1, ξ11*sqrt(Neff), ξ21*Neff, Δs[k]*Neff, K2, ξ12*sqrt(Neff), ξ22*Neff, ps[j]*Neff

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
        #end
    #end
    #6228652K,Ph
    #43311020K Con
    time()-t
    fig = figure(figsize=(7,7), layout= "constrained");
    gs = fig.add_gridspec(1,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

    ax = fig.add_subplot(element(0,0))
    plot(ps, r_means, marker = markers_[1], color=colors_[1])
    ylabel(L"r", fontsize=20);
    xlabel(L"γ", fontsize=20);    
    fig.suptitle("Neff, Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Neff, Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/r_statistics_E_$(N)_γ_Neff_$(Neff)_$(ps[1])_whole_spectrum.png")
    
###

# -------------------------------- state Husimis ---------------------------------------------------
    #Definitions
    N = 100
    p = Δ1, ξ21, K1, Δ2, ξ22, K2, γ = 0., 5., 1., 0., 5., 1., 1.0;
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    ψ = (ψ_S_even[:,1] - ψ_S_odd[:,1])/sqrt(2)
    N_Q = 70 #dimension of the Q function N_Q^2
    q1vals, p1vals, q2vals, p2vals = range(-7,7, length=N_Q),range(-7,7, length=N_Q),range(-7,7, length=N_Q),range(-7,7, length=N_Q)
    
    Qgrid = Q_function_grid_q1q2_full(Qobj(ψ, dims=(N,N)), q1vals, q2vals, N)
    imshow(Qgrid,origin="lower",cmap="magma",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
    xticks([])
    yticks([])
    title("E = $(round(Es[S_n][k], digits=3))")
###

#------------------------------------- PR coherent states Fig1 -------------------------------------
    #CODE
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 5.0);
        
        N = 30;
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);

        x_lim = 7
        N_Q = 5 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 10 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];

        #Multiple Husimi
        t = time()
        Qgrid = zeros(N_Q,N_Q)
        
        IPR_coh = zeros(Float64, length(q1vals), length(q2vals))
        for (j, q1) in enumerate(q1vals)
            for (i, q2) in enumerate(q2vals)
                α1 = (1/sqrt(2))*(q1+ (0.)*im)
                α2 = (1/sqrt(2))*(q2+ (0.)*im)
                for k in 1:N_states
                    IPR_coh[i, j] = (((π^2)*Q_function_full(QuantumObject(ψ[:,k], dims=(N,N)),α1, α2, N))^2)
                end
            end
        end
        PR_coh = 1 ./ IPR_coh

        im = imshow(PR_coh,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.01)
        colorbar()
        time() - t
    ###


    ###Reading Cluster
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 0.01);
        
        N = 100;
        N_Q = 100
        IPR = zeros(N_Q,N_Q)
        for job in 1:10
            IPR += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/PR_coh/IPR_p_$(p)_N_$(N)_N_Q_$(N_Q)_job_$(job).jld")["IPR_coh"];
        end
        PR = 1 ./ IPR 

        xx= x_lim = 6
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        
        im = imshow(PR,origin="lower",cmap="summer",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]),vmax = 25)
        x = range(-xx, xx, length=1000);
        y = range(-xx, xx, length=1000);
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        E_Contours = H_class([q1, p1, q2, p2],p);
        roots, E_cl, λs, s_λ = crit_energies(p)
        contour(coordinates_x, coordinates_y, E_Contours, range(E_cl[1], 10, length = 5), colors="black"); # Only draw contour line for E = 0
        title(" Δ1, ξ21, γ = $(p[[1,4,9]])", fontsize=20)
        xlabel(L"q_1", fontsize=20)
        ylabel(L"q_2", fontsize=20)
        cbar = colorbar(im,label="PR")
        
        time() - t
    ###


###

#------------------------------------- Husimis with Classical E surfaces-------------------------------------
    #EquivKPOs without linear drive
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 5.0);
        
        N = 100;
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);

        roots,E_cl, λs, s_λs = crit_energies(p)
        
        x_lim = 7
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.

        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        
        # Select the slowest 25 energies and their states
        slowest_25_indices = sorted_indices[1:25]
        slowest_25_energies = all_energies[slowest_25_indices]
        slowest_25_states = all_states[:, slowest_25_indices]

        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(5,5)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(5,5)

        #Multiple Husimi
        c =1
        t = time()
        for i in 1:5
            for j in 1:5
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(slowest_25_states[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.05)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == 5
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("E = $(round(real(slowest_25_energies[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(slowest_25_states[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == 5
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("E = $(round(real(slowest_25_energies[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis/Husimis_q1_q2_p_$(p).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis/Husimis_q1_q2_p_$(p)_2.png")
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_Delmar.png")    
    ###

    #Non-Equi KPOs without linear drive9k
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(10.0, 1.0, 0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 1.0);
        
        N = 100;
        E_even, ψ_even, E_odd, ψ_odd  = Coupled_kerr(p, N);

        roots,E_cl, λs, s_λs = crit_energies(p)
        
        x_lim = 6
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.

        #Organizing 
        all_energies = vcat(real.(E_even),  real.(E_odd))
        all_states = hcat(ψ_even, ψ_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        
        # Select the slowest 25 energies and their states
        slowest_25_indices = sorted_indices[1:25]
        slowest_25_energies = all_energies[slowest_25_indices]
        slowest_25_states = all_states[:, slowest_25_indices]


        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(5,5)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(5,5)

        #Multiple Husimi
        c =1
        t = time()
        for i in 1:5
            for j in 1:5
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(slowest_25_states[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.05)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == 5
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("E = $(round(real(slowest_25_energies[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(slowest_25_states[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == 5
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("E = $(round(real(slowest_25_energies[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_p_$(p).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_p_$(p)_2.png")
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_Delmar.png")    
    ###

    #KPOs with linear drive
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(0.0, 1.0, 10.0, 5.0, 0.0, 1.0, 1.0, 5.0, 1.0);
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
        
        N = 100;
        E, ψ  = Coupled_kerr(p, N);

        roots,E_cl, λs, s_λs = crit_energies(p)
        
        x_lim = 6
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.

        
        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(5,5)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(5,5)

        #Multiple Husimi
        c =1
        t = time()
        for i in 1:5
            for j in 1:5
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(ψ[c], q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.05)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == 5
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("E = $(round(real(E[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(ψ[c], q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == 5
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("E = $(round(real(E[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis_q1_q2_p_$(p).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis_q1_q2_p_$(p)_2.png")
    ###

###

#------------------------------------- Average Husimis -------------------------------------
    #EquivKPOs without linear drive
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 0.01);
        
        N = 100;
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);

        x_lim = 7
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];

        #Multiple Husimi
        t = time()
        Qgrid = zeros(N_Q,N_Q)
        N_H = 100
        for i in 1:N_H
            Qgrid += Q_function_grid_q1q2_full(QuantumObject(ψ[:,i], dims=(N,N)), q1vals, q2vals, N)
        end
        Qgrid /= norm(Qgrid)
        im = imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.01)
        colorbar()
        time() - t
        ###

    #Non-Equi KPOs without linear drive9k
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(10.0, 1.0, 0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 1.0);
        
        N = 100;
        E_even, ψ_even, E_odd, ψ_odd  = Coupled_kerr(p, N);

        roots,E_cl, λs, s_λs = crit_energies(p)
        
        x_lim = 6
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.

        #Organizing 
        all_energies = vcat(real.(E_even),  real.(E_odd))
        all_states = hcat(ψ_even, ψ_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        
        # Select the slowest 25 energies and their states
        slowest_25_indices = sorted_indices[1:25]
        slowest_25_energies = all_energies[slowest_25_indices]
        slowest_25_states = all_states[:, slowest_25_indices]


        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(5,5)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(5,5)

        #Multiple Husimi
        c =1
        t = time()
        for i in 1:5
            for j in 1:5
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(slowest_25_states[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.05)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == 5
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("E = $(round(real(slowest_25_energies[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(slowest_25_states[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == 5
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("E = $(round(real(slowest_25_energies[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_p_$(p).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_p_$(p)_2.png")
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_Delmar.png")    
    ###

    #KPOs with linear drive
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(0.0, 1.0, 10.0, 5.0, 0.0, 1.0, 1.0, 5.0, 1.0);
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
        
        N = 100;
        E, ψ  = Coupled_kerr(p, N);

        roots,E_cl, λs, s_λs = crit_energies(p)
        
        x_lim = 6
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.

        
        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(5,5)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(5,5)

        #Multiple Husimi
        c =1
        t = time()
        for i in 1:5
            for j in 1:5
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(ψ[c], q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.05)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == 5
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("E = $(round(real(E[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", linewidht=1.0, zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(ψ[c], q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == 5
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("E = $(round(real(E[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis_q1_q2_p_$(p).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis_q1_q2_p_$(p)_2.png")
    ###

###

# -------------------------------- Multiple plot Entanglement Entropy for Equivalent KPOs withour linear drive  ---------------------------------------------------
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
    div_len = 40
    mean_Ss = zeros(div_len-1,4)
    Es = zeros(N_states,4)
    intv_Es = zeros(div_len-1,4)
    CE = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22,10.))
    for k in 1:length(ps)
        γ = ps[k]
        p = Δ1*Neff, K1, ξ11/sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12/sqrt(Neff), ξ22*Neff, γ*Neff
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
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

    function plotting_entropy()
        #Individual plot
        fig = figure(figsize=(6,6), layout= "constrained")
        gs = fig.add_gridspec(4,6)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        ax = fig.add_subplot(element(slice(1,4),slice(0,6)))
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

        for k in 1:3
            plot(Es[:,k].- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], label="γ = $(ps[k])")
        end
        xlim(-5, 120)
        ylim(0.3,4)
        legend( fontsize=15, shadow=true, loc = "upper left")
        #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
        #legend(fontsize=12, shadow=true, loc = "upper left")
        xlabel("E", fontsize=20)
        ylabel(L"S", fontsize=20)
        xticks([0,60,120],fontsize=15)
        yticks([0,2,4],fontsize=15)
        
        ax = fig.add_subplot(element(0,slice(0,2)))
        for k in 1:1
            plot(Es[:,k], Entropies[:,k], "o", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        xlim(-5, 600)
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        ylim(0.3,4)
        ylabel(L"S", fontsize=20)
        xlabel("E", fontsize=20,labelpad=-15)
        xticks([0,600],fontsize=15)
        yticks([0,2,4],fontsize=15)
        ax = fig.add_subplot(element(0,slice(2,4)))
        for k in 2:2
            plot(Es[:,k] .- Es[:,k][1] , Entropies[:,k], "o", color=colors_[k], label="γ = $(ps[k])", alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1] , mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        xlim(-5, 600)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20,labelpad=-15)
        xticks([0,600],fontsize=15)
        ax = fig.add_subplot(element(0,slice(4,6)))
        for k in 3:3
            plot(Es[:,k].- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], label="γ = $(ps[k])", alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        xlim(-5, 600)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20,labelpad=-15)
        xticks([0,600],fontsize=15)
        
    end
    plotting_entropy()
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_γ_Neff_$(Neff).png")
    
    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    #Average Entropy
    div_len = 15
    mean_S = Float64[]
    
    intv_E =  range(minimum(E), 200.,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    
    plot(E_S_even[1:N_states], Entropies[1:N_states], "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")
###

# -------------------------------- Fig1. Quantum  ---------------------------------------------------
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
    #K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
    #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
    Neff = 3
    ps = [0.1, 1., 5.]
    #ps = [2., 5., 10.]
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
        println("$(k)/3")
        γ = ps[k]
        #Δ1 = Δ2 = ps[k]
        #ξ21 = ξ22 = ps[k]
        p = Δ1*Neff, K1, ξ11/sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12/sqrt(Neff), ξ22*Neff, γ*Neff
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd  = Coupled_kerr_equiv(p, N);       
        #E_S_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(N)_SE.dat")[:,1]
        #ψ_S_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(N)_SE.dat")
        println("1/4")
        println(time() - t)
        #E_A_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(N)_AE.dat")[:,1]
        #ψ_A_even = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(N)_AE.dat")
        println("2/4")
        println(time() - t)
        #E_S_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(N)_SO.dat")[:,1]
        #ψ_S_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(N)_SO.dat")
        println("3/4")
        println(time() - t)
        #E_A_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/Energy_$(p)_N_$(N)_AO.dat")[:,1]
        #ψ_A_odd = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Convergency/State_$(p)_N_$(N)_AO.dat")
        println("4/4")
        println(time() - t)
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
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ ))_Δ.jld", "Entropies", Entropies)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ ))_Δ.jld", "Es", Es)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ ))_Δ.jld", "intv_Es", intv_Es)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12, γ))_Δ.jld", "mean_Ss", mean_Ss)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "Entropies", Entropies)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "Es", Es)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/n_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "n1s", n1s)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/PRfock_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld", "PR_fock", PR_fock)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld", "intv_Es", intv_Es)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld", "mean_Ss", mean_Ss)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld", "Entropies", Entropies)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld", "Es", Es)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld", "intv_Es", intv_Es)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld", "mean_Ss", mean_Ss)


    
    #Load 
    N = 200
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
    #K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
    #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
    Neff = 5
    ps = [0.1, 1., 5.]
    #ps = [2., 5., 10.]
    N_states=4000
    Entropies = zeros(N_states,4)
    div_len = 40
    mean_Ss = zeros(div_len-1,4)
    Es = zeros(N_states,4)
    intv_Es = zeros(div_len-1,4)
    
    #Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ))_Δ.jld")["Entropies"]
    #Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ ))_Δ.jld")["Es"]
    #intv_Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ))_Δ.jld")["intv_Es"]
    #mean_Ss = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((Δ1, K1, ξ11, Δ2, K2, ξ12,γ))_Δ.jld")["mean_Ss"]

    Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["Entropies"]
    Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["Es"]
    intv_Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["intv_Es"]
    mean_Ss = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["mean_Ss"]
    
    #Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld")["Entropies"]
    #Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld")["Es"]
    #intv_Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld")["intv_Es"]
    #mean_Ss = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ ))_Δ.jld")["mean_Ss"]

    function old_plotting_entropy()
        #Individual plot
        fig = figure(figsize=(6,6), layout= "constrained")
        gs = fig.add_gridspec(4,6)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        ax = fig.add_subplot(element(slice(1,4),slice(0,6)))
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        xx_lim = 200

        for k in 1:3
            plot(Es[:,k].- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], label="γ = $(ps[k])")
        end
        xlim(-5, 120)
        ylim(0.3,4)
        legend( fontsize=15, shadow=true, loc = "upper left")
        #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
        #legend(fontsize=12, shadow=true, loc = "upper left")
        xlabel("E", fontsize=20)
        ylabel(L"S", fontsize=20)
        xticks([0,60,120],fontsize=15)
        yticks([0,2,4],fontsize=15)
        
        ax = fig.add_subplot(element(0,slice(0,2)))
        for k in 1:1
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        xlim(-5, xx_lim)
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        ylim(0.3,4)
        ylabel(L"S", fontsize=20)
        xlabel("E", fontsize=20,labelpad=-15)
        xticks([0,xx_lim],fontsize=15)
        yticks([0,2,4],fontsize=15)
        ax = fig.add_subplot(element(0,slice(2,4)))
        for k in 2:2
            plot(Es[:,k] .- Es[:,k][1] , Entropies[:,k], "o", color=colors_[k], label="γ = $(ps[k])", alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1] , mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        xlim(-5, xx_lim)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20,labelpad=-15)
        xticks([0,xx_lim],fontsize=15)
        ax = fig.add_subplot(element(0,slice(4,6)))
        for k in 3:3
            plot(Es[:,k].- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], label="Δ = $(ps[k])", alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        xlim(-5, xx_lim)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20,labelpad=-15)
        xticks([0,xx_lim],fontsize=15)
        
    end
    old_plotting_entropy()
    
    function plotting_entropy_γ()
        #Individual plot
        fig = figure(figsize=(18,6), layout= "constrained")
        gs = fig.add_gridspec(1,3)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        xx_lim = 200
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
        ps = [0.1, 1., 5.]
        #ps = [2. , 5. ,10.]
        
    
        ax = fig.add_subplot(element(0,0))
        for k in 1:1
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"γ = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        #xlim(-5, xx_lim)
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        ylim(0.3,4)
        ylabel(L"S", fontsize=20)
        xlabel("E", fontsize=20)#,labelpad=-15)
        #xticks([0,xx_lim],fontsize=15)
        yticks([0,1,2,3,4],fontsize=15)
        
        ax = fig.add_subplot(element(0,1))
        for k in 2:2
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
             #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"γ = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        #xlim(-5, xx_lim)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20)#,labelpad=-15)
        #xticks([0,xx_lim],fontsize=15)

        ax = fig.add_subplot(element(0,2))
        for k in 3:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"γ = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        #xlim(-5, xx_lim)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20)#,labelpad=-15)
        #xticks([0,xx_lim],fontsize=15)
        
    end
    function plotting_entropy_Δ()
        #Individual plot
        fig = figure(figsize=(18,6), layout= "constrained")
        gs = fig.add_gridspec(1,3)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        xx_lim = 200
        #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
        K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
        ps = [0.1, 1., 5.]
        #ps = [2. , 5. ,10.]
        
    
        ax = fig.add_subplot(element(0,0))
        for k in 1:1
            #γ = ps[k]
            Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"Δ = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        #xlim(-5, xx_lim)
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        ylim(0.3,4)
        ylabel(L"S", fontsize=20)
        xlabel("E", fontsize=20)#,labelpad=-15)
        #xticks([0,xx_lim],fontsize=15)
        yticks([0,1,2,3,4],fontsize=15)
        
        ax = fig.add_subplot(element(0,1))
        for k in 2:2
            γ = ps[k]
            Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
             #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"Δ = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        #xlim(-5, xx_lim)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20)#,labelpad=-15)
        #xticks([0,xx_lim],fontsize=15)

        ax = fig.add_subplot(element(0,2))
        for k in 3:3
            #γ = ps[k]
            Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"Δ = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(intv_Es[:,k] .- Es[:,k][1], mean_Ss[:,k], color="black", label="⟨S⟩")
        end
        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
        #xlim(-5, xx_lim)
        ylim(0.3,4)
        yticks([])
        xlabel("E", fontsize=20)#,labelpad=-15)
        #xticks([0,xx_lim],fontsize=15)
        
    end
    plotting_entropy_γ()
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_γ_Neff_$(Neff).png")
    
    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    #Average Entropy
    div_len = 15
    mean_S = Float64[]
    
    intv_E =  range(minimum(E), 200.,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    
    plot(E_S_even[1:N_states], Entropies[1:N_states], "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")
###

# -------------------------------- Fig1. Quantum <n> and PR_fock  ---------------------------------------------------


  

    
    #Load 
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
    #K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
    #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
    Neff = 3
    ps = [0.1, 1., 5.]
    #ps = [2., 5., 10.]
    N_states=4000
    div_len = 40
    ICs= 1000
    
    #Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["Entropies"]
    #Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22)).jld")["Es"]
    #n1s = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/n_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22)).jld")["n1s"]
    #PR_fock = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/PRfock_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22)).jld")["PR_fock"]

    Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld")["Entropies"]
    Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld")["Es"]
    n1s = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/n_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld")["n1s"]
    PR_fock = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/PRfock_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Neff_$(Neff).jld")["PR_fock"]

    
    function plotting_γ()
        #Individual plot
        fig = figure(figsize=(30,18), layout= "constrained")
        gs = fig.add_gridspec(4,3)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        xx_lim = 500
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
        ps = [0.1, 1., 5.]
        #ps = [2. , 5. ,10.]
        
        #Lyapunov exponent
        #Lyapunov exponent
        for j in 1:3
            ax = fig.add_subplot(element(0,j-1))
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];  

            roots, cE, λs_p31, s_λ = crit_energies(p);
            n_E = 500
            Es1 = range(cE[1],600+cE[1], length=n_E)
            println( cE[1] )
            Energies = Es1 .- cE[1] 
            λs = [Float64[] for i in 1:n_E]
            λ_mean = zeros(n_E)
            data_miss= []
            for job in 1:100
                try
                    λ_mean[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov/Mean_Lyapunov_Energies_$(p)_job_$(job)__ICs_$(ICs).jld")["λ_mean"][(5*(job-1) + 1):5*job]
                    λs[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov/Lyapunov_Energies_$(p)_job_$(job)_ICs_$(ICs).jld")["λs"][(5*(job-1) + 1):5*job]
                catch
                    #println("Missing job $(job)")
                    push!(data_miss, job)
                end    
            end
            println("γ = $(ps[j])")
            println("data_miss = $(data_miss)")
            #println(length(data_miss))
            #pltos
            plot(Energies, λ_mean, "-", color="red", markersize=5, label = L"⟨λ⟩");
            for i in 1:length(λs)
                scatter(range(Energies[i], Energies[i], length=length(λs[i])), λs[i], color="black", alpha=0.2,s=1);
            end
            title(L"γ = %$(ps[j])", fontsize=15)
            #ax.text(.03, 0.85, L"γ = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
            if j ==1
                legend(fontsize=20, shadow=true, loc = "upper right");
            end
            xticks([])
            if j ==1
                ylabel("λ", fontsize = 20)
                yticks([0, 4, 8], fontsize=15)
                #xlim(0, 130)
                #ylim(-.05, 3.5)
            else
                yticks([])
            end
            xlim(-0.05, xx_lim)
            ylim(-.05, 8.0)
        end

        #Entropy
        for k in 1:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            ax = fig.add_subplot(element(1,k-1))
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], alpha = 0.2)
            #title(L"γ = %$(ps[k])", fontsize=15)
            mean_S = Float64[]
            intv_E =  range(minimum(Es[:,k]),maximum(Es[:,k])+1,length=div_len)
            for i in 1: (div_len-1)
                index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], Es[:,k])
                push!(mean_S, mean(Entropies[:,k][index]))
            end
            plot(intv_E[1:end-1] .- Es[:,k][1], mean_S, color="black", label="⟨S⟩")
            xlim(-5, xx_lim)
            if k == 1
                ylabel(L"S", fontsize=20)
                yticks([0,1,2,3],fontsize=15)
            else
                yticks([])
            end
            #legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
            ylim(0.3,4)
            #ylabel(L"S", fontsize=20)
            #xlabel("E", fontsize=20)#,labelpad=-15)
            xticks([])
            #yticks([0,1,2,3,4],fontsize=15)
        end

        #Photon number
        max_num = maximum(n1s) + 0.5
        for k in 1:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            ax = fig.add_subplot(element(2,k-1))
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,max_num, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], n1s[:,k], "o", color=colors_[k], alpha = 0.2)
            if k == 1
                ylabel(L"⟨n_{1}⟩", fontsize=20)
                yticks([10,30,50],fontsize=15)
            else
                yticks([])
            end
            #xlabel("E", fontsize=20)#,labelpad=-15)
            xlim(-5, xx_lim)
            #legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
            ylim(7,max_num)
            xticks([])
            #yticks([0,1,2,3,4],fontsize=15)
        end


        #PR_fock
        max_num = maximum(PR_fock) + 0.5
        for k in 1:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            ax = fig.add_subplot(element(3,k-1))
            plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,max_num, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], PR_fock[:,k], "o", color=colors_[k], alpha = 0.2)

            if k == 1
                ylabel(L"PR_{Fock}", fontsize=20)
                yticks([0,400,800],fontsize=15)
            else
                yticks([])
            end
            xlabel("E", fontsize=20)#,labelpad=-15)
            xlim(-5, xx_lim)
            #legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
            ylim(0,max_num)
            #xticks([0,xx_lim],fontsize=15)
            #yticks([0,1,2,3,4],fontsize=15)
        end
    end
    plotting_γ()
    a
    #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_γ_Neff_$(Neff)_Fig1.png")
    savefig("C:/Users/edson/Desktop/Entanglement_Entropy_N_$(N)_γ_Neff_$(Neff)_Fig1.png")


    function plotting_γ_below_well()
        #Individual plot
        fig = figure(figsize=(30,20), layout= "constrained")
        gs = fig.add_gridspec(4,3)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ =  0., 1., 0., 0., 1., 0., 1.
        ps = [0.1, 1., 5.]
        
        #ps = [2. , 5. ,10.]
        ICs= 1000
        
        #Lyapunov exponent
        #Lyapunov exponent
        for j in 1:3
            ax = fig.add_subplot(element(0,j-1))
            title(L"γ = %$(ps[j])", fontsize=15)
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];  

            roots, cE, λs_p31, s_λ = crit_energies(p);
            n_E = 500
            Es1 = range(cE[1],600+cE[1], length=n_E)
            println( cE[1] )
            Energies = Es1 .- cE[1] 
            λs = [Float64[] for i in 1:n_E]
            λ_mean = zeros(n_E)
            data_miss= []
            for job in 1:100
                try
                    λ_mean[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov/Mean_Lyapunov_Energies_$(p)_job_$(job)__ICs_$(ICs).jld")["λ_mean"][(5*(job-1) + 1):5*job]
                    λs[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov/Lyapunov_Energies_$(p)_job_$(job)_ICs_$(ICs).jld")["λs"][(5*(job-1) + 1):5*job]
                catch
                    #println("Missing job $(job)")
                    push!(data_miss, job)
                end    
            end
            println("γ = $(ps[j])")
            println("data_miss = $(data_miss)")
            #println(length(data_miss))
            #pltos
            plot(Energies, λ_mean, "-", color="red", markersize=5, label = L"⟨λ⟩");
            for i in 1:length(λs)
                scatter(range(Energies[i], Energies[i], length=length(λs[i])), λs[i], color="black", alpha=0.2,s=1);
            end
            #ax.text(.03, 0.85, L"γ = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
            if j ==1
                legend(fontsize=20, shadow=true, loc = "upper right");
            end
            xticks([])
            if j ==1
                ylabel("λ", fontsize = 20)
                yticks([0, 1,2,3], fontsize=15)
                #xlim(0, 130)
                #ylim(-.05, 3.5)
            else
                yticks([])
            end
            xx_lim = -cE[1]
            function unique_indices(x; tol::Float64=1e-3)
                unique_inds = Int[]
                seen = Float64[]

                for (i, val) in enumerate(x)
                    if all(abs(val - s) > tol for s in seen)
                        push!(seen, val)
                        push!(unique_inds, i)
                    end
                end
                return unique_inds
            end
            E_cl = cE
            indices_E = unique_indices(E_cl)
            min_S = 0
            max_S = 4
            for i in indices_E
                plot(range(E_cl[i].- E_cl[1],E_cl[i].- E_cl[1], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λ[i] * "= $(round(E_cl[i] .- E_cl[1], digits=3))")
            end
            xlim(-5, xx_lim)
            ylim(-.05, 3.5)
        end


        #Entropy
        for k in 1:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            ax = fig.add_subplot(element(1,k-1))
            xx_lim = -Es[:,k][1]
            roots, E_cl, λs, s_λs = crit_energies(p)
            min_S = minimum(Entropies)
            max_S = maximum(Entropies)
            function unique_indices(x; tol::Float64=1e-3)
                unique_inds = Int[]
                seen = Float64[]

                for (i, val) in enumerate(x)
                    if all(abs(val - s) > tol for s in seen)
                        push!(seen, val)
                        push!(unique_inds, i)
                    end
                end
                return unique_inds
            end
            indices_E = unique_indices(E_cl)
            for i in indices_E
                n =sum(real.(Es[:,k])  .< E_cl[i])
                plot(range(E_cl[i].- Es[:,k][1],E_cl[i].- Es[:,k][1], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i] .- Es[:,k][1], digits=3)), nb = $(n)")
            end
            #plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,4, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", color=colors_[k], alpha = 0.9)
            mean_S = Float64[]
            intv_E =  range(minimum(Es[:,k]),maximum(Es[:,k])+1,length=div_len)
            for i in 1: (div_len-1)
                index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], Es[:,k])
                push!(mean_S, mean(Entropies[:,k][index]))
            end
            plot(intv_E[1:end-1] .- Es[:,k][1], mean_S, color="black", label="⟨S⟩")
            xlim(-5, xx_lim)
            legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
            ylim(0.3,4)
            if k == 1
                ylabel(L"S", fontsize=20)
                yticks([0,1,2,3],fontsize=15)
            else
                yticks([])
            end
            #ylabel(L"S", fontsize=20)
            #xlabel("E", fontsize=20)#,labelpad=-15)
            xticks([])
            #yticks([0,1,2,3,4],fontsize=15)
        end

        #Photon number
        max_num = 40
        for k in 1:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            ax = fig.add_subplot(element(2,k-1))
            xx_lim = -Es[:,k][1]
            roots, E_cl, λs, s_λs = crit_energies(p)
            min_S = minimum(n1s)
            max_S = maximum(n1s)
            function unique_indices(x; tol::Float64=1e-3)
                unique_inds = Int[]
                seen = Float64[]

                for (i, val) in enumerate(x)
                    if all(abs(val - s) > tol for s in seen)
                        push!(seen, val)
                        push!(unique_inds, i)
                    end
                end
                return unique_inds
            end
            indices_E = unique_indices(E_cl)
            
            for i in indices_E
                n =sum(real.(Es[:,k])  .< E_cl[i])
                plot(range(E_cl[i].- Es[:,k][1],E_cl[i].- Es[:,k][1], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i] .- Es[:,k][1], digits=3)), nb = $(n)")
            end
            #plot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,max_num, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], n1s[:,k], "o", color=colors_[k], alpha = 0.9)
            if k == 1
                ylabel(L"⟨n_{1}⟩", fontsize=20)
                yticks([10,25,35],fontsize=15)
            else
                yticks([])
            end
            #ylabel(L"⟨n_{1}⟩", fontsize=20)
            #xlabel("E", fontsize=20)#,labelpad=-15)
            xlim(-5, xx_lim)
            #legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
            ylim(10,max_num)
            xticks([])
            #yticks([0,1,2,3,4],fontsize=15)
        end


        #PR_fock
        max_num = 900
        for k in 1:3
            γ = ps[k]
            #Δ1 = Δ2 = ps[k]
            #ξ21 = ξ22 = ps[k]
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
            #CE = crit_energies(p)
            ax = fig.add_subplot(element(3,k-1))
            xx_lim = -Es[:,k][1]


            roots, E_cl, λs, s_λs = crit_energies(p)
            min_S = minimum(PR_fock)
            max_S = maximum(PR_fock)
            function unique_indices(x; tol::Float64=1e-3)
                unique_inds = Int[]
                seen = Float64[]

                for (i, val) in enumerate(x)
                    if all(abs(val - s) > tol for s in seen)
                        push!(seen, val)
                        push!(unique_inds, i)
                    end
                end
                return unique_inds
            end
            indices_E = unique_indices(E_cl)
            
            for i in indices_E
                n =sum(real.(Es[:,k])  .< E_cl[i])
                plot(range(E_cl[i].- Es[:,k][1],E_cl[i].- Es[:,k][1], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i] .- Es[:,k][1], digits=3)), nb = $(n)")
            end
            #lot(range(-Es[:,k][1], -Es[:,k][1], length = 2), range(0,max_num, length=2), lw = 4, color="black")
            #plot(Es[:,k] .- Es[:,k][1], Entropies[:,k], "o", label=L"ξ_{2} = %$(ps[k])", color=colors_[k], alpha = 0.2)
            plot(Es[:,k] .- Es[:,k][1], PR_fock[:,k], "o", color=colors_[k], alpha = 0.9)
            if k == 1
                ylabel(L"PR_{Fock}", fontsize=20)
                yticks([0,400,800],fontsize=15)
            else
                yticks([])
            end
            xlabel("E", fontsize=20)#,labelpad=-15)
            xlim(-5, xx_lim)
            #legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
            ylim(0,800)
            #xticks([0,xx_lim],fontsize=15)
            #yticks([0,1,2,3,4],fontsize=15)
        end
    end
    plotting_γ_below_well()
    
    #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_γ_Neff_$(Neff)_Fig1.png")
    savefig("C:/Users/edson/Desktop/Entanglement_Entropy_N_$(N)_γ_Neff_$(Neff)_Fig4.png")
###


# -------------------------------- Entanglement Entropy for Equivalent KPOs withour linear drive  ---------------------------------------------------
    #Definitions
    N = 100
    parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ =  0., 1., 0., 5., 0., 1., 0., 5., 5.0
    Neff = 3
    p = Δ1*Neff, K1, ξ11/sqrt(Neff), ξ21*Neff, Δ2*Neff, K2, ξ12/sqrt(Neff), ξ22*Neff, γ*Neff
    

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
    
    #=
        S_rand = 0
        for i in 1:10
            ψ_rand = randn(N^2);
            ψ_rand /= norm(ψ_rand);
            S_rand += Entanglement_entropy_fock(ψ_rand, N)
        end
        S_rand /= 10
    =#

    #=
        ψ_rand =  randn(ComplexF64, N^2);
        ψ_rand /= norm(ψ_rand);
        x_lim= 6
        N_Q = 100 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_rand, dims=(N,N)), q1vals, q2vals, N)
        im = imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        colorbar(label="H")
        xlabel("q1")
        ylabel("q2")
        title("Random State of size "*L"N^{2}")
    =#

    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
    CE = crit_energies(parameters)
    CE[2][1], E_S_even[1]/Neff^2


    N_states = 1000 #States per Symmetry sector
    Entropies = zeros(4*N_states);
    for i in 1:N_states
        #Entropies[i] = Entanglement_entropy_fock(ψ[i].data,N)
        Entropies[i] = Entanglement_entropy_fock(ψ_S_even[:,i],N)
        Entropies[N_states + i] = Entanglement_entropy_fock(ψ_A_even[:,i],N)
        Entropies[2*N_states + i] = Entanglement_entropy_fock(ψ_S_odd[:,i],N)
        Entropies[3*N_states + i] = Entanglement_entropy_fock(ψ_A_odd[:,i],N)
    end
    
    #Average Entropy
    div_len = 100
    mean_S = Float64[]
    E = real.(vcat(E_S_even[1:N_states], E_A_even[1:N_states], E_S_odd[1:N_states], E_A_odd[1:N_states])) 
    intv_E =  range(minimum(E),maximum(E)+1,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end

    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$(parameters)_Neff_$(Neff).jld", "Entropies", Entropies)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$(parameters)_Neff_$(Neff).jld", "E", E)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$(parameters)_Neff_$(Neff).jld", "intv_E", intv_E)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$(parameters)_Neff_$(Neff).jld", "mean_S", mean_S)


    #Individual plot
    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_S_even[1:N_states], Entropies[1:N_states], alpha = 0.3, "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], alpha = 0.3, "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], alpha = 0.3, "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], alpha = 0.3, "o", color="black")
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")

    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    #Average Entropy
    div_len = 15
    mean_S = Float64[]
    
    intv_E =  range(minimum(E), 200.,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    
    plot(E_S_even[1:N_states], Entropies[1:N_states], "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 3.5)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")
###

# -------------------------------- Entanglement Entropy for Nonequiv-KPOs ---------------------------------------------------
    #Definitions
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 3., 0., 1., 0., 5., 1.
   
    t = time();
    E_even, ψ_even, E_odd, ψ_odd = Coupled_kerr(p, N);
    time() - t 

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
    
    ψ_rand = rand(N^2);
    ψ_rand /= norm(ψ_rand);
    S_rand = Entanglement_entropy_fock(ψ_rand, N)
    N_states = 2000 #States per Symmetry sector
    Entropies = zeros(2*N_states);
    
    for i in 1:N_states
        #Entropies[i] = Entanglement_entropy_fock(ψ[i].data,N)

        Entropies[i] = Entanglement_entropy_fock(ψ_even[:,i],N)
        Entropies[N_states + i] = Entanglement_entropy_fock(ψ_odd[:,i],N)
    end
     
    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_even[1:N_states], Entropies[1:N_states], alpha = 0.3, "o", color="black")
    plot(E_odd[1:N_states], Entropies[N_states+1:2*N_states], alpha = 0.3, "o", color="black")
    plot(E_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    

    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    plot(E_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    E_cl'
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    for i in [1, 3, 5, 9]
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end


    plot(E_even[1:N_states], Entropies[1:N_states], alpha = 0.3, "o", color="black")
    plot(E_odd[1:N_states], Entropies[N_states+1:2*N_states], alpha = 0.3, "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    



    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Entanglement_Entropy_N_$(N)_p_$(p).png")
###


# -------------------------------- Entanglement Entropy for KPOs with linear drive ---------------------------------------------------
    #Definitions
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 1., 5., 0., 1., 1., 5., 1.
   
    t = time();
    E, ψ = Coupled_kerr(p, N);
    time() - t 

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
    
    #=
        ψ_rand = randn(N^2);
        ψ_rand /= norm(ψ_rand);
        S_rand = Entanglement_entropy_fock(ψ_rand, N)
    =#

    N_states = 4000 #States per Symmetry sector
    Entropies = zeros(N_states);
    
    for i in 1:N_states
        Entropies[i] = Entanglement_entropy_fock(ψ[i].data,N)
    end
    
    fig = figure(figsize=(10,5), layout= "constrained");
    gs = fig.add_gridspec(1,2);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    ax = fig.add_subplot(element(0,0));
    plot(E[1:N_states], Entropies, alpha = 0.3, "o", color="black");
    #plot(E[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state");

    legend(fontsize=12, shadow=true, loc = "upper left");
    xlabel("E", fontsize=15);
    ylabel(L"S", fontsize=15);
    

    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");

    
    ax = fig.add_subplot(element(0,1));
    #plot(E[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3);
    roots, E_cl, λs, s_λs = crit_energies(p);
    E_cl'
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    plot(E[1:N_states], Entropies, "o", color="black")
    xlim(minimum(real.(E)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    



    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")

###


# -------------------------------- PR Eigenstates for Uncoupled basis  ---------------------------------------------------
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  5.0, 1., 0., 5., 5.0, 1., 0., 5., 1.;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies)
    E = all_energies[sorted_indices]
    ψ = all_states[:, sorted_indices]
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E0 = E[1:N_states];
    ψ0 = ψ[:,1:N_states];
    
    #Uncoupled KPOs
    E_unc, ψ_unc = H_un(p, N);
    PR = vec( 1 ./ sum(abs.(ψ_unc' * ψ0).^4, dims=1) ) 

    #Benchmarking values
    N=100
        rand_vec = randn(ComplexF64, N^2)
        rand_vec /= norm(rand_vec) 
        loc_ψ = zeros(N^2)
        loc_ψ[1] = 1.
        deloc_ψ = ones(N^2)
        deloc_ψ /= norm(deloc_ψ)
        PR_random_fock = 1/sum(abs.(ψ_unc' * rand_vec).^4)
        PR_random_unc = 1/sum(abs.(rand_vec).^4)
        PR_deloc_fock = 1/sum(abs.(ψ_unc' * deloc_ψ).^4)
        PR_deloc_unc = 1/sum(abs.(deloc_ψ).^4)
        PR_loc = 1/sum(abs.(loc_ψ).^4)
    ##

    #Average PR
    div_len = 200
    mean_PR = Float64[]
    intv_E =  range(minimum(E0),maximum(E0)+1,length=div_len)
    for i in 1:(div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E0)
        push!(mean_PR, mean(PR[index]))
    end

    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E0, PR, alpha = 0.3, "o", color="black")
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel("E", fontsize=15)
    ylabel(L"PR_{Unc}", fontsize=15)
    plot(intv_E[1:end-1], mean_PR, color = "red", label ="Mean")
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_PR = minimum(PR)
    max_PR = maximum(PR)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    n = 0
    for i in indices_E
        n = sum(real.(E0)  .< E_cl[i])
        plot(range(E_cl[i],E_cl[i], length=2), range(min_PR,max_PR, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    E_lim = 0
    plot(E0, PR, "o", color="black")
    xlim(minimum(real.(E0)) -5., E_lim+5)
    ylim(min_PR-0.05, maximum(PR[1:n])+5)
    legend(fontsize=10, shadow=true, loc = "upper left")
    xlabel("E", fontsize=15)
    ylabel(L"PR_{Unc}", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/PR/PR_unc_N_$(N)_p_$(p).png")

###


# -------------------------------- Entanglement Entropy and PR SCARS  ---------------------------------------------------
    #Definitions
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., 1.
   
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
    
    
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
    
    #=
        S_rand = 0
        for i in 1:10
            ψ_rand = randn(N^2);
            ψ_rand /= norm(ψ_rand);
            S_rand += Entanglement_entropy_fock(ψ_rand, N)
        end
        S_rand /= 10
    =#

    #=
        ψ_rand =  randn(ComplexF64, N^2);
        ψ_rand /= norm(ψ_rand);
        x_lim= 6
        N_Q = 100 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_rand, dims=(N,N)), q1vals, q2vals, N)
        im = imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        colorbar(label="H")
        xlabel("q1")
        ylabel("q2")
        title("Random State of size "*L"N^{2}")
    =#
    

    N_states = 1000 #States per Symmetry sector
    Entropies = zeros(4*N_states);
    for i in 1:N_states
        #Entropies[i] = Entanglement_entropy_fock(ψ[i].data,N)
        Entropies[i] = Entanglement_entropy_fock(ψ_S_even[:,i],N)
        Entropies[N_states + i] = Entanglement_entropy_fock(ψ_A_even[:,i],N)
        Entropies[2*N_states + i] = Entanglement_entropy_fock(ψ_S_odd[:,i],N)
        Entropies[3*N_states + i] = Entanglement_entropy_fock(ψ_A_odd[:,i],N)
    end
    Entanglement_entropy_fock(ψ_S_even[:,1],N,false)
    hist(Entropies)
    #Average Entropy
    div_len = 100
    mean_S = Float64[]
    E = real.(vcat(E_S_even[1:N_states], E_A_even[1:N_states], E_S_odd[1:N_states], E_A_odd[1:N_states])) 
    intv_E =  range(minimum(E),maximum(E)+1,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end

    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_S_even[1:N_states], Entropies[1:N_states], alpha = 0.3, "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], alpha = 0.3, "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], alpha = 0.3, "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], alpha = 0.3, "o", color="black")
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")

    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    #Average Entropy
    div_len = 15
    mean_S = Float64[]
    
    intv_E =  range(minimum(E), 200.,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    
    plot(E_S_even[1:N_states], Entropies[1:N_states], "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")

###

# -------------------------------- Statistics Entanglement Entropy for Equivalent KPOs withour linear drive  ---------------------------------------------------
    #Definitions
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  5., 1., 0., 5., 5., 1., 0., 5., 10.01
   
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

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
    
    #=
        S_rand = 0
        for i in 1:10
            ψ_rand = randn(N^2);
            ψ_rand /= norm(ψ_rand);
            S_rand += Entanglement_entropy_fock(ψ_rand, N)
        end
        S_rand /= 10
    =#

    #=
        ψ_rand =  randn(ComplexF64, N^2);
        ψ_rand /= norm(ψ_rand);
        x_lim= 6
        N_Q = 100 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_rand, dims=(N,N)), q1vals, q2vals, N)
        im = imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        colorbar(label="H")
        xlabel("q1")
        ylabel("q2")
        title("Random State of size "*L"N^{2}")
    =#
    

    N_states = 1000 #States per Symmetry sector
    Entropies = zeros(4*N_states);
    for i in 1:N_states
        #Entropies[i] = Entanglement_entropy_fock(ψ[i].data,N)
        Entropies[i] = Entanglement_entropy_fock(ψ_S_even[:,i],N)
        Entropies[N_states + i] = Entanglement_entropy_fock(ψ_A_even[:,i],N)
        Entropies[2*N_states + i] = Entanglement_entropy_fock(ψ_S_odd[:,i],N)
        Entropies[3*N_states + i] = Entanglement_entropy_fock(ψ_A_odd[:,i],N)
    end


    using StatsBase
    μ, σ, s, k = mean(Entropies), std(Entropies), skewness(Entropies), kurtosis(Entropies)
    #μ1, σ1, s1, k1 = μ, σ, s, k
    μ10, σ10, s10, k10 = μ, σ, s, k
    μ5, σ5, s5, k5 = μ, σ, s, k
    #Statistics
    #hist(Entropies, bins=30, density=true, alpha=0.6, color="g")
    
    #Average Entropy
    div_len = 100
    mean_S = Float64[]
    E = real.(vcat(E_S_even[1:N_states], E_A_even[1:N_states], E_S_odd[1:N_states], E_A_odd[1:N_states])) 
    intv_E =  range(minimum(E),maximum(E)+1,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    s_E = mean_S[end]

    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_S_even[1:N_states], Entropies[1:N_states], alpha = 0.3, "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], alpha = 0.3, "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], alpha = 0.3, "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], alpha = 0.3, "o", color="black")
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")

    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    #Average Entropy
    div_len = 15
    mean_S = Float64[]
    
    intv_E =  range(minimum(E), 200.,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    
    plot(E_S_even[1:N_states], Entropies[1:N_states], "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")

###

# -------------------------------- 2KPOs projection Uncoupled heatmap---------------------------------------------------
    n_states = 10;
    labels_states = labels_states_KPOs(n_states);

    labels_states_new = [];
    x_indices = [];
    for n in 1:n_states
        push!(labels_states_new, labels_states[(n-1)*n_states+1])
        push!(x_indices, (n-1)*n_states)
    end
    
    N = 100;
    function Proj_unc(p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        E = all_energies[sorted_indices]
        ψ = all_states[:, sorted_indices]
        
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E0 = E[real.(E) .< 0];
        ψ0 = ψ[:,1:length(E0)];
        
        #Uncoupled KPOsE_unc
        E_unc, ψ_unc = H_un(p, N, n_states = n_states);
    
        return abs2.(ψ0' * ψ_unc), E0        
    end 
    
    density_projs = [];
    Es = [];
    ps = [0.5, 1., 2., 5.];
    for p_ in ps
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., p_;
        density_proj, E0 = Proj_unc(p, N)
        push!(density_projs, density_proj)
        push!(Es, E0)
    end
    
    fig = figure(figsize=(12,6), layout= "constrained")
    gs = fig.add_gridspec(2,2);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    count = 1
    for j in 1:2
        for i in 1:2
            ax = fig.add_subplot(element(j-1,i-1));
            imshow(density_projs[count], origin="lower",cmap="OrRd",vmax= 0.5, aspect="auto")
            yticks(Int.(round.(range(0,length(Es[count])-1,length=5), digits=0)), round.(Es[count][Int.(round.(range(1,length(Es[count]),length=5), digits=0))], digits=2))
            ylabel("E")
            xticks(x_indices,labels_states_new, fontsize=8)
            title("γ = $(ps[count]), n = $(length(Es[count]))")
            cbar = colorbar(fraction=.025, pad=0.01)
            cbar.ax.set_xlabel(L"P_{un}", labelpad=10)
            cbar.ax.xaxis.set_label_position("bottom")  # move to bottom       
            #text(1.3, -0.1, L"P_{un}", fontsize=18, verticalalignment="top")
            #xlim(-0.5, 7*n_states)
            count+=1
        end
    end
    #text(10.3, -1.1, L"P_{un}", fontsize=18) 
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Projection_γ_$(ps).png")
###


# -------------------------------- Wehrl Entropy Eigenstates/chaos analysis  ---------------------------------------------------
    #Code
        t = time()
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  1., 1., 0., 5., 1., 1., 0., 5., 1.;
        N = 10
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        println("Time for Hamiltonian and Diagonalization: ", time() - t)
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        E = all_energies[sorted_indices]
        ψ = all_states[:, sorted_indices]
        
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E0 = E[real.(E) .< 0];
        ψ0 = ψ[:,1:length(E0)];
        E[1:50]

        #Benchmarking
        function Q_function_full2!(buffer, Ψ, α1::ComplexF64, α2::ComplexF64, N)
            # buffer should be preallocated with size N^2
            ψ_coh1 = coherent(N, α1)
            ψ_coh2 = coherent(N, α2)
            # Compute tensor product manually for speed
            @inbounds for i in 1:N
                for j in 1:N
                    buffer[(i-1)*N + j] = ψ_coh1[i] * ψ_coh2[j]
                end
            end
            return abs2(dot(buffer, Ψ)) / π^2
        end

        
        
        α1, α2 = 3.0 + 0im, 3.0 + 0im
        val1 = Q_function_full(Qobj(ψ0[:,1], dims=(N,N)),α1, α2,N)
        val1
        buffer = zeros(ComplexF64, N^2)
        val = Q_function_full2!(buffer, ψ0[:,1], α1, α2, N)
        Q_function_full2(Qobj(ψ0[:,1], dims=(N,N)),α1, α2,N) 
        

        qp_i, qp_f = -6,6
        N_Q = 5
        q1vals, p1vals, q2vals, p2vals = range(qp_i,qp_f, length=N_Q),range(qp_i,qp_f, length=N_Q),range(qp_i,qp_f, length=N_Q),range(qp_i,qp_f, length=N_Q)
        step(q1vals)
        length(E0) #States per Symmetry sector

        ##Rand vector
            ψ_rand =  randn(ComplexF64, N^2);
            ψ_rand /= norm(ψ_rand);
            Wehrl_entropy_q1q2(ψ_rand, q1vals, q2vals) # 0.12953404404076735
        ##
        
        function Wehrl_entropy_q1q2(ψ, q1vals, q2vals)
            N_Q = length(q1vals)
            Q = zeros(N_Q^2)
            N = Int(sqrt(length(ψ)))
            Δq1, Δq2 = step(q1vals), step(q2vals)
            count=1
            for q1 in q1vals, q2 in q2vals
                α1 = (1/sqrt(2))*(q1 + 0im)
                α2 = (1/sqrt(2))*(q2 + 0im)
                Q[count] = Q_function_full(Qobj(ψ, dims=(N,N)), α1, α2, N)
                count +=1
            end
            Q /= sum(Q) #normalization
            Q = Q[Q .> 1e-14];
            return -sum((Q .* log.(Q)))*Δq1*Δq2
        end

        t = time()
        Entropies = zeros(100)#zeros(N_states)
        for j in 1:5
            Entropies[j] =  Wehrl_entropy_q1q2(ψ0[:,j], q1vals, q2vals)
        end
        time() - t

        save("C:/Users/edson/Desktop/Wehrl_entropy/Wehrl_Entropies_$(p)_$(1).jld", "Entropies", Entropies)
        
    ###

    #Files from Cluster
        N= 100
        N_Q = 40
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., 11.0;
        ns_job = 1:2:101
        length(ns_job) 
        job =1
        Entropies = zeros(100)
        Entropies[ns_job[job]:ns_job[job+1]-1] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q)_$(ns_job[job])_$(ns_job[job+1]-1).jld")["Entropies"][ns_job[job]:ns_job[job+1]-1]
        Entropies
        no_data = []
        
        for job in 2:(length(ns_job)-1)
            try
                Entropies[ns_job[job]:ns_job[job+1]-1] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q)_$(ns_job[job])_$(ns_job[job+1]-1).jld")["Entropies"][ns_job[job]:ns_job[job+1]-1]
            catch
                println("No data $(ns_job[job]), $(ns_job[job+1]-1)")
                push!(no_data, job)
            end
        end
        println("no_data = ", no_data)
        length(no_data)
        
        
        #Missing jobs
            jobs = [] 
            for job in no_data
                push!(jobs, ns_job[job])
                push!(jobs, ns_job[job+1]-1)
            end
            println("jobs = ", jobs)
            length(jobs)

            for job in jobs
                try
                    Entropies[job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q)_$(job).jld")["Entropies"][job]
                catch
                    println("No data $(job)")
                    push!(no_data, job)
                end
            end
        #

        #Emerging data
            N= 100
            N_Q = 40
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., 11.0;
            ns_job = 1:2:101
            length(ns_job) 
            job =1 
            data = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q)_$(ns_job[job])_$(ns_job[job+1]-1).jld")["Entropies"][ns_job[job]:ns_job[job+1]-1]
            no_data = []
            
            Entropies = zeros(200)
            Entropies[1:100] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q).jld")["Entropies"]
            for job in 2:(length(ns_job)-1)
                try
                    Entropies[100 + ns_job[job]:100 + ns_job[job+1]-1] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q)_$(ns_job[job])_$(ns_job[job+1]-1).jld")["Entropies"][ns_job[job]:ns_job[job+1]-1]
                catch
                    println("No data $(ns_job[job]), $(ns_job[job+1]-1)")
                    push!(no_data, job)
                end
            end
            Entropies[Entropies .< 0]
            save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q).jld" , "Entropies", Entropies)
            
        #
    ##


    #PLots from Files
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., 0.2;
        N = 100
        N_Q = 40
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
    
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        E = all_energies[sorted_indices]
        ψ = all_states[:, sorted_indices]
        
        N_E = length(E[E .< 149])
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        
        Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/WE/WE4D_p_$(p)_N_$(N)_N_Q_$(N_Q).jld")["Entropies"]
        E0 = E[1:N_E]

        ploting()
        function ploting()
            fig = figure(figsize=(7,5), layout= "constrained");
            gs = fig.add_gridspec(1,1);
            element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
            ax = fig.add_subplot(element(0,0));
        

            xlabel("E", fontsize=15);
            ylabel(L"S_{W}", fontsize=15);
            

            fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");

            roots, E_cl, λs, s_λs = crit_energies(p);
            
            min_y, max_y = 6, 9.2
            ylim(min_y, max_y) 
            function unique_indices(x; tol::Float64=1e-3)
                unique_inds = Int[]
                seen = Float64[]

                for (i, val) in enumerate(x)
                    if all(abs(val - s) > tol for s in seen)
                        push!(seen, val)
                        push!(unique_inds, i)
                    end
                end
                return unique_inds
            end
            indices_E = unique_indices(E_cl)
            
            for i in indices_E
                n =sum(real.(E0)  .< E_cl[i]) 
                plot(range(E_cl[i],E_cl[i], length=2), range(min_y,max_y, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
            end
            plot(E0,Entropies[1:N_E], "o", color="black")
            legend(fontsize=10, shadow=true, loc = "upper left")

            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Wehrl_Entropy/Wehrl_Entropy4D_N_$(N)_p_$(p).png")
    
        end

    ###
###


# -------------------------------- GS as coherentstates  ---------------------------------------------------
    #Code
        N = 100
        N_p = 100
        ps = collect(range( 0.01,  13., length = N_p))
        P_coh = zeros(N_p)
        #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. # For γ
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1. # For ξ2
        K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1. # For Δ
        
        for j in 1:N_p
            #parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            #parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
            parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 

            E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(parameters, N);
            #Organizing 
            all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
            all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

            # Get sorting indices for energies (ascending order)
            sorted_indices = sortperm(all_energies)
            E = all_energies[sorted_indices]
            ψ = all_states[:, sorted_indices]
            

            roots, cE, λs, s_λ = crit_energies(parameters);
            q1, p1, q2, p2 = roots[1]
            α1 = (1/sqrt(2))*(q1 + p1*1im)
            α2 = (1/sqrt(2))*(q2 + p2*1im)
            P = Q_function_full(QuantumObject(ψ[:,1], dims = (N,N)), α1, α2,N)*π^2
            q1, p1, q2, p2 = roots[2]
            α1 = (1/sqrt(2))*(q1 + p1*1im)
            α2 = (1/sqrt(2))*(q2 + p2*1im)
            P += Q_function_full(QuantumObject(ψ[:,1], dims = (N,N)), α1, α2,N)*π^2
            P_coh[j] = P
        end

        ploting()
        function ploting()
            fig = figure(figsize=(7,5), layout= "constrained");
            gs = fig.add_gridspec(1,1);
            element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
            ax = fig.add_subplot(element(0,0));
        

            #xlabel("γ", fontsize=15);
            xlabel("Δ", fontsize=15);
            ylabel(L"P_{coh}", fontsize=15);
            

            #fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))");
            fig.suptitle("K1, ξ11, ξ21, K2, ξ12, ξ22, γ = $((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))");

            
            plot(ps,P_coh, "o", color="black")

            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/BellState_N_$(N)_γ.png")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/BellState_N_$(N)_Δ.png")
    
        end
        


    ###

    #Code2
        N = 100
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ  = 0., 1., 0., 5., 0., 1., 0., 5., 5. # For γ
        
        parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ 
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(parameters, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        E = all_energies[sorted_indices]
        ψ = all_states[:, sorted_indices]
            

        roots, cE, λs, s_λ = crit_energies(parameters);
        q1, p1, q2, p2 = roots[1]
        α1 = (1/sqrt(2))*(q1 + p1*1im)
        α2 = (1/sqrt(2))*(q2 + p2*1im)
        ψ_coh1 = kron(coherent(N, α1), coherent(N, α2))
        q1, p1, q2, p2 = roots[2]
        α3 = (1/sqrt(2))*(q1 + p1*1im)
        α4 = (1/sqrt(2))*(q2 + p2*1im)
        ψ_coh2 = kron(coherent(N, α3), coherent(N, α4))
        coherent(N, α3)' * coherent(N, α1)
        coherent(N, α4)' * coherent(N, α2)

        ψ_S = ψ_coh2 + ψ_coh1
        ψ_S /= norm(ψ_S)
        ψ_A = ψ_coh2 - ψ_coh1
        ψ_A /= norm(ψ_A)

        QuantumObject(ψ[:,2], dims = (N,N))' * ψ_S

        ploting()
        function ploting()
            fig = figure(figsize=(7,5), layout= "constrained");
            gs = fig.add_gridspec(1,1);
            element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
            ax = fig.add_subplot(element(0,0));
        

            #xlabel("γ", fontsize=15);
            xlabel("Δ", fontsize=15);
            ylabel(L"P_{coh}", fontsize=15);
            

            #fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))");
            fig.suptitle("K1, ξ11, ξ21, K2, ξ12, ξ22, γ = $((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))");

            
            plot(ps,P_coh, "o", color="black")

            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/BellState_N_$(N)_γ.png")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/BellState_N_$(N)_Δ.png")
    
        end
        


    ###
###


# -------------------------------- # of protected states for Equivalent KPOs withour linear drive  ---------------------------------------------------
    #Definitions
    
    N = 30

    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
    n_p = 100
    #p_ = range(0.01,5., length=n_p)
    p_ = range(1.,10., length=n_p)

    #Getting the classical data
        ns_job = 1:4:101
        λ_tail = 1e-1
        step_ = 0.1
        Es = fill(NaN, 100)
        for job in 1:25
            try 
                E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/E_tail/E_tail_p_$(p)_ξ2_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step_).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                Es[ns_job[job]:ns_job[job+1]-1] = E_tail
            catch
                println("Missing job $(job)")
            end
        end
        plot(p_, Es, "o", label = L"λ_{t} = %$(λ_tail)")
        xlabel(L"ξ_{2}", fontsize=15)
        ylabel(L"ΔE_{tail}", fontsize=15)
        title("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.")
        legend(fontsize=10, shadow=true, loc = "upper left");
    ##

    ns = fill(NaN, n_p)
    for i in 1:n_p
        p = Δ1, K1, ξ11, p_[i], Δ1, K2, ξ12, p_[i], γ
        if isnan(Es[i])
            continue
        end
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies)
        E = all_energies[sorted_indices]
        ψ = all_states[:, sorted_indices]
        
        ns[i] = sum(E .< Es[i])
    end    
    
    fig = figure(figsize=(7,5), layout= "constrained")
    gs = fig.add_gridspec(1,1)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    
    ax = fig.add_subplot(element(0,0))
    plot(p_, ns, "o", color="black")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel(L"ξ_{2}", fontsize=15)
    ylabel("Number of protected states", fontsize=15)
    #fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(p[1:8])")
    fig.suptitle("Δ1, K1, ξ11, Δ2, K2, ξ12, γ = $(p[[1,2,3,6,7,9]])")
    fig.suptitle("K1, ξ11, ξ21, K2, ξ12, ξ22, γ = $(p[[2,3,4,6,7,8,9]])")
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/n_protected_states_N_$(N)_p_$(p)_ξ2.png")


    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Entropies)
    max_S = maximum(Entropies)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    for i in indices_E
        n_SE =sum(real.(E_S_even)  .< E_cl[i])
        n_AE =sum(real.(E_A_even)  .< E_cl[i])
        n_SO =sum(real.(E_S_odd)  .< E_cl[i])
        n_AO =sum(real.(E_A_odd)  .< E_cl[i])
        n = n_SE + n_AE + n_SO + n_AO
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    #Average Entropy
    div_len = 15
    mean_S = Float64[]
    
    intv_E =  range(minimum(E), 200.,length=div_len)
    for i in 1: (div_len-1)
        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
        push!(mean_S, mean(Entropies[index]))
    end
    
    plot(E_S_even[1:N_states], Entropies[1:N_states], "o", color="black")
    plot(E_A_even[1:N_states], Entropies[N_states+1:2*N_states], "o", color="black")
    plot(E_S_odd[1:N_states], Entropies[2*N_states+1:3*N_states], "o", color="black")
    plot(E_A_odd[1:N_states], Entropies[3*N_states+1:4*N_states], "o", color="black")
    xlim(minimum(real.(E_S_even)) -3., 200)
    ylim(minimum(Entropies)-0.05, 2.5)
    plot(intv_E[1:end-1], mean_S, color = "red", label ="Mean")
    legend(fontsize=10, shadow=true, loc = "lower right")
    xlabel("E", fontsize=15)
    ylabel(L"S", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Entanglement_Entropy/Entanglement_Entropy_N_$(N)_p_$(p).png")

###


# -------------------------------- Wigner function  ----------------------------------------------
    Ψs = [S_even*ψ_S_even, S_odd*ψ_S_odd, A_even*ψ_A_even, A_odd*ψ_A_odd]
    Es = [E_S_even, E_S_odd,E_A_even, E_A_odd]
    Ps = [true, false, true, false]
    states=["Even Symmetric states", "Odd Symmetric states", "Even Asymmetric states","Odd Asymmetric states"]
    S_n = 3 #select symmetry

    #α = q+ip
    function D_mn(m,n,α)
        sum_ = 0
        for k in 0:min(m,n)
            sum_ += ( (α)^(m - k) )*( (-α')^(n - k) )/( factorial_log(k)*factorial_log(m-k)*factorial_log(n-k) ) 
        end
        return sum_*exp(abs2(α)/2)*sqrt(factorial_log(m)*factorial_log(n))
    end
    function W_function(Ψ::Vector{Float64}, α1::ComplexF64, α2::ComplexF64, even=true)
        ψ_α = 0.    
        j = 1
        if even
            for base_ in base_even
                n1,n2 = base_
                ψ_α += Ψ[j]*((α1')^(n1))*((α2')^(n2))/(sqrt(factorial_log(n1)*factorial_log(n2)))
                j+=1
            end
        else
            for base_ in base_odd
                n1,n2 = base_
                ψ_α += Ψ[j]*((α1')^(n1))*((α2')^(n2))/(sqrt(factorial_log(n1)*factorial_log(n2)))
                j+=1
            end
        end
        return (abs(ψ_α*exp(-abs(α1)^2 - abs(α2)^2))^2)/π^2
    end
###

    function W(α)
        α = 1+1im
        W_α = zeros(Complex,N,N)
        W_α[1,1] = exp(-2*abs2(α))
        for i in 1:N-1
            W_α[1,i+1] = W_α[1,i]*2*α/sqrt(i) #Calculating first column
            W_α[i+1,1] = (W_α[1,i+1])' #Calculating first row
        end
        for i in 1:N-1
            W_α[i+1,i+1] = 2*(α')*W_α[i+1,i]/sqrt(i+1) - W_α[i,i]
            #println("$(i+1),$(i+1) = $(i+1),$(i) +  $(i),$(i)")
            for j in i+2:N 
                W_α[i+1,j] = 2*(α')*W_α[i+1,j-1]/sqrt(j) - sqrt((i+1)/j)*W_α[i,j-1]
                W_α[j,i+1] = (W_α[i+1,j])'
                #println("$(i+1),$(j) = $(i+1), $(j-1) + $(i),$(j-1)")
            end
        end
        return W_α
    end
    
    W_sum = 0
    ψ = ψ_GS1
    W_α1 = W(2*α1)
    W_α2 = W(2*α2)
    for i in 1:2#N^2
        n1,n2 = n_base[i]
        for j in 1:3#N^2
            m1,m2 = n_base[j]
            W_sum += ψ[i]*(ψ[j]')*W_α1[m1+1,n1+1]*W_α2[m2+1,n2+1]
            println(i,j)
            println("$(n1+1), $(m1+1),$(n2+1), $(m2+1)")
        end
    end
    W_sum*(2/π)^2
    




for i in 1:N-1
        W_α[i+1,i+1] = 2*(α')*W_α[i+1,i]/sqrt(i+1) - W_α[i,i]
        println("$(i+1),$(i+1) = $(i+1),$(i) +  $(i),$(i)")
        for j in 2:(i+1)
            if (i+2) > N
                continue
            end
            println("$(j),$(i+2)")
            W_α[j,i+2] = 2*α*W_α[j-1,i+2]/sqrt(j) - sqrt((i+2)/j)*W_α[j-1,i+1]
            W_α[i+2,j] = (W_α[j,i+2])' 
            #W_α[i+1,j] = 2*(α')*W_α[i+1,j-1]/sqrt(j) - sqrt((i+1)/j)*W_α[i,j-1]
            #W_α[j,i+1] = (W_α[i+1,j])'
            println("$(i+1),$(j) = $(i+1), $(j-1) + $(i),$(j-1)")
            #println("$(j), $(i+2) =  $(j-1), $(i+2) + $(j-1),$(i+1)")
            #println("$(i+2), $(j) defined")
        end
    end


###


# -------------------------------- Entropy in the Uncoupeld states ---------------------------------------------------
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0.0, 1., 0., 5., 0.0, 1., 0., 5., .5;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies)
    E = all_energies[sorted_indices]
    ψ = all_states[:, sorted_indices]
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E0 = E[1:N_states];
    ψ0 = ψ[:,1:N_states];
    
    #Uncoupled KPOs
    E_unc, ψ_unc = H_un(p, N);
    length(E_unc)
    c_sq = abs2.(ψ0' * ψ_unc)


    
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
    
    
    Entropies = zeros(N_states);
    for i in 1:N_states
        Entropies[i] = Entanglement_entropy_fock(ψ0[:,i],N)
    end

    Avg_entropy = (Entropies' * c_sq)' #./ N_states 
    
    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_unc, Avg_entropy, "o", alpha = 0.1, color="black")
    
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel(L"E_{unc}", fontsize=15)
    ylabel(L"⟨S⟩", fontsize=15)
    xlim(E_unc[1]-100, 4000)
    ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Avg_entropy)
    max_S = maximum(Avg_entropy)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    n = 0
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i in indices_E
        n = sum(real.(E0)  .< E_cl[i])
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2),color=colors_[i], lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    E_lim = 100
    plot(E_unc, Avg_entropy, "o", color="black")
    xlim(minimum(real.(E_unc)) -5., E_lim+100)
    ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    legend(fontsize=10, shadow=true, loc = "upper left")
    xlabel(L"E_{unc}", fontsize=15)
    ylabel(L"⟨S⟩", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Uncoupled_Entropy/Entropy_unc_N_$(N)_p_$(p).png")

###


# -------------------------------- Entropy in the Uncoupeld states ---------------------------------------------------
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0.0, 1., 0., 5., 0.0, 1., 0., 5., .5;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies)
    E = all_energies[sorted_indices]
    ψ = all_states[:, sorted_indices]
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E0 = E[1:N_states];
    ψ0 = ψ[:,1:N_states];
    
    #Uncoupled KPOs
    E_unc, ψ_unc = H_un(p, N);
    length(E_unc)
    c_sq = abs2.(ψ0' * ψ_unc)


    
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
    
    
    Entropies = zeros(N_states);
    for i in 1:N_states
        Entropies[i] = Entanglement_entropy_fock(ψ0[:,i],N)
    end

    Avg_entropy = (Entropies' * c_sq)' #./ N_states 
    
    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_unc, Avg_entropy, "o", alpha = 0.1, color="black")
    
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel(L"E_{unc}", fontsize=15)
    ylabel(L"⟨S⟩", fontsize=15)
    xlim(E_unc[1]-100, 4000)
    ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Avg_entropy)
    max_S = maximum(Avg_entropy)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    n = 0
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i in indices_E
        n = sum(real.(E0)  .< E_cl[i])
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2),color=colors_[i], lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    E_lim = 100
    plot(E_unc, Avg_entropy, "o", color="black")
    xlim(minimum(real.(E_unc)) -5., E_lim+100)
    ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    legend(fontsize=10, shadow=true, loc = "upper left")
    xlabel(L"E_{unc}", fontsize=15)
    ylabel(L"⟨S⟩", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Uncoupled_Entropy/Entropy_unc_N_$(N)_p_$(p).png")

###


# -------------------------------- Qudits and coupling ---------------------------------------------------
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0.0, 1., 0., 5., 0.0, 1., 0., 5., .1;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd))
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd)

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies)
    E = all_energies[sorted_indices]
    ψ = all_states[:, sorted_indices]
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E0 = E[1:N_states];
    ψ0 = ψ[:,1:N_states];
    
    #Uncoupled KPOs
    n_qudits = 2 
    E_unc, ψ_unc = H_un(p, N, n_qudits);
    length(E_unc)
    c_sq = abs2.(ψ0' * ψ_unc)


    
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
    
    
    Entropies = zeros(N_states);
    for i in 1:N_states
        Entropies[i] = Entanglement_entropy_fock(ψ0[:,i],N)
    end

    Avg_entropy = (Entropies' * c_sq)' #./ N_states 
    
    fig = figure(figsize=(10,5), layout= "constrained")
    gs = fig.add_gridspec(1,2)
    element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
    ax = fig.add_subplot(element(0,0))
    plot(E_unc, Avg_entropy, "o", alpha = 0.1, color="black")
    
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel(L"E_{unc}", fontsize=15)
    ylabel(L"⟨S⟩", fontsize=15)
    xlim(E_unc[1]-100, 4000)
    ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")

    
    ax = fig.add_subplot(element(0,1))
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3)
    roots, E_cl, λs, s_λs = crit_energies(p)
    min_S = minimum(Avg_entropy)
    max_S = maximum(Avg_entropy)
    function unique_indices(x; tol::Float64=1e-3)
        unique_inds = Int[]
        seen = Float64[]

        for (i, val) in enumerate(x)
            if all(abs(val - s) > tol for s in seen)
                push!(seen, val)
                push!(unique_inds, i)
            end
        end
        return unique_inds
    end
    indices_E = unique_indices(E_cl)
    
    n = 0
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i in indices_E
        n = sum(real.(E0)  .< E_cl[i])
        plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2),color=colors_[i], lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3)), nb = $(n)")
    end

    E_lim = 100
    plot(E_unc, Avg_entropy, "o", color="black")
    xlim(minimum(real.(E_unc)) -5., E_lim+100)
    ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    legend(fontsize=10, shadow=true, loc = "upper left")
    xlabel(L"E_{unc}", fontsize=15)
    ylabel(L"⟨S⟩", fontsize=15)
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Uncoupled_Entropy/Entropy_unc_N_$(N)_p_$(p).png")

###


# -------------------------------- Qudits and Scrambling(Fotoc) ---------------------------------------------------
    #Fotoc for random state =  9794.542962686817

    N = 100
    #Definitions
    q1, p1 =  tensor(QuantumToolbox.position(N), qeye(N)), tensor(QuantumToolbox.momentum(N), qeye(N)); 
    q2, p2 =  tensor(qeye(N), QuantumToolbox.position(N)), tensor(qeye(N), QuantumToolbox.momentum(N)); 
    q1_sq, p1_sq = q1*q1, p1*p1;
    q2_sq, p2_sq = q2*q2, p2*p2;
    
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0.0, 1., 0., 5., 0.0, 1., 0., 5., 10.;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies);
    E = all_energies[sorted_indices];
    ψ = all_states[:, sorted_indices];
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E = E[1:N_states];
    ψ = ψ[:,1:N_states];
    
    #Uncoupled KPOs
    n_qudits = 4
    E_unc, ψ_unc, E_unc1, ψ_unc1, E_unc2, ψ_unc2 = H_un(p, N, n_states = n_qudits);
    ψ_0 = copy(ψ_unc); #Initial states

    #Coherent IC
        roots, E_cl, λs, s_λs = crit_energies(p);
        q1c,p1c,q2c,p2c = roots[1]
        q1c,q2c = 0., 0.
        ψ_0[:,1] = kron(coherent(N,(q1c + 0im)/sqrt(2)) , coherent(N,(q2c + 0im)/sqrt(2))).data;

        q1c,q2c = sqrt(ξ22), sqrt(ξ22)
        cat1, cat2 = coherent(N,(q1c + 0im)) + coherent(N, (-q1c + 0im)), coherent(N,(q2c + 0im)) + coherent(N,(-q2c + 0im))
        cat1, cat2 = cat1 / norm(cat1), cat2 / norm(cat2)
        ψ_0[:,1] = kron(cat1, cat2).data
    #

    t_final = 100.
    t_interval = range(0.0, t_final, length=100)
    #ψ_t = [zeros(Complex, N^2) for _ in t_interval]
    q1_avg, q1_sq_avg = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
    p1_avg, p1_sq_avg = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
    q2_avg, q2_sq_avg = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
    p2_avg, p2_sq_avg = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
    Fotoc = zeros(length(t_interval), n_qudits^2);
    n1, n2 = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
    t = time();
    for i in 1:n_qudits^2
        println("State: $(i) / $(n_qudits^2)")
        ψ_t = ψ_unc[:,i]
        #Measurements for IC 
            k=1
            q1_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q1 * QuantumObject(vec(ψ_t), dims= (N,N)))
            q1_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q1_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            p1_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p1 * QuantumObject(vec(ψ_t), dims= (N,N)))
            p1_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p1_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            
            n1_ = repeat(0:N-1, inner=N)
            n1[k,i] = sum(abs2.(ψ_t) .* n1_)
            n2_ = repeat(0:N-1, outer=N)
            n2[k,i] = sum(abs2.(ψ_t) .* n2_)

            q2_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q2 * QuantumObject(vec(ψ_t), dims= (N,N)))
            q2_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q2_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            p2_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p2 * QuantumObject(vec(ψ_t), dims= (N,N)))
            p2_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p2_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            
            Fotoc[k,i] = abs2(q1_sq_avg[k,i] - q1_avg[k,i]^2) + abs2(p1_sq_avg[k,i] - p1_avg[k,i]^2) + abs2(q2_sq_avg[k,i] - q2_avg[k,i]^2) + abs2(p2_sq_avg[k,i] - p2_avg[k,i]^2)
        
        for k in 2:length(t_interval)
            #evolution trhough eigenbasis
            c_j = (ψ' * ψ_0[:,i]) .* exp.(-1im*E*t_interval[k]) 
            ψ_cj = ψ .* conj(c_j)'
            ψ_t = sum(ψ_cj, dims=2)
            
            #Measurements
            q1_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q1 * QuantumObject(vec(ψ_t), dims= (N,N)))
            q1_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q1_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            p1_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p1 * QuantumObject(vec(ψ_t), dims= (N,N)))
            p1_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p1_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            
            n1_ = repeat(0:N-1, inner=N)
            n1[k,i] = sum(abs2.(ψ_t) .* n1_)
            n2_ = repeat(0:N-1, outer=N)
            n2[k,i] = sum(abs2.(ψ_t) .* n2_)

            q2_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q2 * QuantumObject(vec(ψ_t), dims= (N,N)))
            q2_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q2_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            p2_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p2 * QuantumObject(vec(ψ_t), dims= (N,N)))
            p2_sq_avg[k,i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p2_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
            
            Fotoc[k,i] = abs2(q1_sq_avg[k,i] - q1_avg[k,i]^2) + abs2(p1_sq_avg[k,i] - p1_avg[k,i]^2) + abs2(q2_sq_avg[k,i] - q2_avg[k,i]^2) + abs2(p2_sq_avg[k,i] - p2_avg[k,i]^2)
        end
        
    end
    println("Time elapsed: $(time() - t) seconds")
    fig = figure(figsize=(7,5), layout= "constrained");
    gs = fig.add_gridspec(1,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    ax = fig.add_subplot(element(0,0));
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", 
               "magenta", "teal", "gold", "navy", "lime", "coral"];
    labels_ = labels_states_KPOs(n_qudits);
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h", "H", "X", "d", "1", "2", "3"];
    for i in 1:n_qudits^2
        plot(t_interval, Fotoc[:,i], marker = markers_[i], color=colors_[i], label = labels_[i])
        #plot(t_interval, Fotoc[:,i], marker = markers_[i], color=colors_[i], label = "(q1,q2) = ($(q1c),$(q2c))")
    end
    legend(fontsize=10, shadow=true, loc = "upper right");
    xlabel(L"time", fontsize=15);
    ylabel(L"Fotoc", fontsize=15);
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Fotoc_unc_N_$(N)_p_$(p)_$(n_qudits).png")
    #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Fotoc_unc_N_$(N)_p_$(p)_coherent_$(q1c)_$(q2c).png")

    ##Plot for n_qudit =4
        fig = figure(figsize=(40,12), layout= "constrained");
        gs = fig.add_gridspec(2,4);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);

        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", 
                "magenta", "teal", "gold", "navy", "lime", "coral"];
        labels_ = labels_states_KPOs(n_qudits);
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h", "H", "X", "d", "1", "2", "3"];
        listing = [[1,1], [1,2], [1,3], [1,4], [2,1], [2,2], [2,3], [2,4]]
        c = 1
        for i in 1:8
            ax1 = fig.add_subplot(gs[listing[i][1], listing[i][2]])
            for j in 1:2
                plot(t_interval, Fotoc[:,c], marker = markers_[c], color=colors_[c], label = labels_[c])
                #plot(t_interval, Fotoc[:,i], marker = markers_[i], color=colors_[i], label = "(q1,q2) = ($(q1c),$(q2c))")     
                legend(fontsize=10, shadow=true, loc = "upper right");
                xlabel(L"time", fontsize=15);
                ylabel(L"Fotoc", fontsize=15);
                c += 1
            end
        end
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Fotoc_unc_N_$(N)_p_$(p)_$(n_qudits).png")
    ####    
###


# -------------------------------- Qudits and Scrambling(Fotoc) ---------------------------------------------------
    #Fotoc for random state =  9794.542962686817

    N = 100
    #Definitions
    q1, p1 =  tensor(QuantumToolbox.position(N), qeye(N)), tensor(QuantumToolbox.momentum(N), qeye(N)); 
    q2, p2 =  tensor(qeye(N), QuantumToolbox.position(N)), tensor(qeye(N), QuantumToolbox.momentum(N)); 
    q1_sq, p1_sq = q1*q1, p1*p1;
    q2_sq, p2_sq = q2*q2, p2*p2;
    
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0.0, 1., 0., 5., 0.0, 1., 0., 5., 9.;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies);
    E = all_energies[sorted_indices];
    ψ = all_states[:, sorted_indices];
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E = E[1:N_states];
    ψ = ψ[:,1:N_states];
    
    

    #Coherent IC
        roots, E_cl, λs, s_λs = crit_energies(p);
        q1c,p1c,q2c,p2c = roots[1]
        q1c,q2c = 0., 0.
        ψ_0[:,1] = kron(coherent(N,(q1c + 0im)/sqrt(2)) , coherent(N,(q2c + 0im)/sqrt(2))).data;

        q1c,q2c = sqrt(ξ22), sqrt(ξ22)
        cat1, cat2 = coherent(N,(q1c + 0im)) + coherent(N, (-q1c + 0im)), coherent(N,(q2c + 0im)) + coherent(N,(-q2c + 0im))
        cat1, cat2 = cat1 / norm(cat1), cat2 / norm(cat2)
        ψ_0[:,1] = kron(cat1, cat2).data
    #

    q1_avg, q1_sq_avg = zeros(length(E)), zeros(length(E));
    p1_avg, p1_sq_avg = zeros(length(E)), zeros(length(E));
    q2_avg, q2_sq_avg = zeros(length(E)), zeros(length(E));
    p2_avg, p2_sq_avg = zeros(length(E)), zeros(length(E));
    Fotoc = zeros(length(E));
    n1, n2 = zeros(length(E)), zeros(length(E));
    t = time();
    for i in 1:length(E)
        #println("State: $(i) / $(length(E))")
        ψ_t = ψ[:,i]
        q1_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q1 * QuantumObject(vec(ψ_t), dims= (N,N)))
        q1_sq_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q1_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
        p1_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p1 * QuantumObject(vec(ψ_t), dims= (N,N)))
        p1_sq_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p1_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
        
        n1_ = repeat(0:N-1, inner=N)
        n1[i] = sum(abs2.(ψ_t) .* n1_)
        n2_ = repeat(0:N-1, outer=N)
        n2[i] = sum(abs2.(ψ_t) .* n2_)

        q2_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q2 * QuantumObject(vec(ψ_t), dims= (N,N)))
        q2_sq_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * q2_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
        p2_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p2 * QuantumObject(vec(ψ_t), dims= (N,N)))
        p2_sq_avg[i] = real(QuantumObject(vec(ψ_t), dims= (N,N))' * p2_sq * QuantumObject(vec(ψ_t), dims= (N,N)))
        
        Fotoc[i] = abs2(q1_sq_avg[i] - q1_avg[i]^2) + abs2(p1_sq_avg[i] - p1_avg[i]^2) + abs2(q2_sq_avg[i] - q2_avg[i]^2) + abs2(p2_sq_avg[i] - p2_avg[i]^2)
    
    end
    println("Time elapsed: $(time() - t) seconds")
    
    #q1_sq_avg_low, p1_sq_avg_low, n1_low, Fotoc_low = q1_sq_avg, p1_sq_avg, n1, Fotoc
    plot(1:length(E), q1_sq_avg, "o")
    plot(1:length(E), q1_sq_avg_low, "o")

    plot(1:length(E), Fotoc, "o")
    plot(1:length(E), Fotoc_low, "o")
    
    fig = figure(figsize=(7,5), layout= "constrained");
    gs = fig.add_gridspec(1,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    ax = fig.add_subplot(element(0,0));
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", 
               "magenta", "teal", "gold", "navy", "lime", "coral"];
    labels_ = labels_states_KPOs(n_qudits);
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h", "H", "X", "d", "1", "2", "3"];
    for i in 1:n_qudits^2
        plot(t_interval, Fotoc[:,i], marker = markers_[i], color=colors_[i], label = labels_[i])
        #plot(t_interval, Fotoc[:,i], marker = markers_[i], color=colors_[i], label = "(q1,q2) = ($(q1c),$(q2c))")
    end
    legend(fontsize=10, shadow=true, loc = "upper right");
    xlabel(L"time", fontsize=15);
    ylabel(L"Fotoc", fontsize=15);
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Fotoc_unc_N_$(N)_p_$(p)_$(n_qudits).png")
    #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Fotoc_unc_N_$(N)_p_$(p)_coherent_$(q1c)_$(q2c).png")

    ##Plot for n_qudit =4
        fig = figure(figsize=(40,12), layout= "constrained");
        gs = fig.add_gridspec(2,4);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);

        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", 
                "magenta", "teal", "gold", "navy", "lime", "coral"];
        labels_ = labels_states_KPOs(n_qudits);
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h", "H", "X", "d", "1", "2", "3"];
        listing = [[1,1], [1,2], [1,3], [1,4], [2,1], [2,2], [2,3], [2,4]]
        c = 1
        for i in 1:8
            ax1 = fig.add_subplot(gs[listing[i][1], listing[i][2]])
            for j in 1:2
                plot(t_interval, Fotoc[:,c], marker = markers_[c], color=colors_[c], label = labels_[c])
                #plot(t_interval, Fotoc[:,i], marker = markers_[i], color=colors_[i], label = "(q1,q2) = ($(q1c),$(q2c))")     
                legend(fontsize=10, shadow=true, loc = "upper right");
                xlabel(L"time", fontsize=15);
                ylabel(L"Fotoc", fontsize=15);
                c += 1
            end
        end
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Fotoc_unc_N_$(N)_p_$(p)_$(n_qudits).png")
    ####    
###


# -------------------------------- Qudits and Scrambling(SP) ---------------------------------------------------
    N = 100
    #Definitions
    q1, p1 =  tensor(QuantumToolbox.position(N), qeye(N)), tensor(QuantumToolbox.momentum(N), qeye(N)); 
    q2, p2 =  tensor(qeye(N), QuantumToolbox.position(N)), tensor(qeye(N), QuantumToolbox.momentum(N)); 

    
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  10.0, 1., 0., 5., 10.0, 1., 0., 5., 5.0;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);

    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies);
    E = all_energies[sorted_indices];
    ψ = all_states[:, sorted_indices];
    
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E = E[1:N_states];
    ψ = ψ[:,1:N_states];
    
    #Uncoupled KPOs
    n_qudits = 4
    E_unc, ψ_unc = H_un(p, N, n_states = n_qudits);
    ψ_0 = ψ_unc #Initial states

    Qobj(ψ_0[:,1], dims=(N,N))
    println(typeof(N))
    ψ_0[:,1]
    t_final = 100.
    t_interval = range(0.0, t_final, length=100)
    #ψ_t = [zeros(Complex, N^2) for _ in t_interval]
    SP = zeros(length(t_interval), n_qudits^2);
    
    t = time();
    for i in 1:n_qudits^2
        SP[1,i] = 1.0
        println("State: $(i) / $(n_qudits^2)")
        for k in 2:length(t_interval)
            #evolution trhough eigenbasis
            c_j = (ψ' * ψ_0[:,i]) .* exp.(-1im*E*t_interval[k]) 
            ψ_cj = ψ .* conj(c_j)'
            ψ_t = sum(ψ_cj, dims=2)
            
            #Survival Probability
            SP[k,i] = abs2(ψ_0[:,i]' * vec(ψ_t))
        end
    end
    println("Time elapsed: $(time() - t) seconds")
    
    fig = figure(figsize=(7,5), layout= "constrained");
    gs = fig.add_gridspec(1,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    ax = fig.add_subplot(element(0,0));
    #colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    labels_ = labels_states_KPOs(n_qudits);
    #markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
    for i in 1:n_qudits^2
        plot(t_interval, SP[:,i], marker = markers_[i], color=colors_[i], label = labels_[i])
    end
    legend(fontsize=10, shadow=true, loc = "upper right");
    #plot(E_S_even[1:N_states], range(S_rand,S_rand, length=N_states), lw = 3,label = "Random state")
    #legend(fontsize=12, shadow=true, loc = "upper left")
    xlabel(L"time", fontsize=15);
    ylabel(L"SP", fontsize=15);
    #xlim(E_unc[1]-100, 4000)
    #ylim(Avg_entropy[1]-0.05, maximum(Avg_entropy)+0.05)
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/SP_unc_N_$(N)_p_$(p)_n_qudits_$(n_qudits).png")

###


# -------------------------------- Qudits and Scrambling(PR_unc/Entanglement Entropy) ---------------------------------------------------
        
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


    N = 100
    #Definitions
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., 20.;
    
    function Scrambling(p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        
        #Uncoupled KPOs
        E_unc, ψ_unc = H_un(p, N);
        
        n_qudits = 2
        E_0, ψ_0 = H_un(p, N, n_states = n_qudits);

        t_final = 100.
        t_interval = range(0.0, t_final, length=100)
        #ψ_t = [zeros(Complex, N^2) for _ in t_interval]
        PR_unc = zeros(length(t_interval), n_qudits^2);
        Entropy = zeros(length(t_interval), n_qudits^2);
        Leakage = zeros(length(t_interval), n_qudits^2);
        n1, n2 = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
        t = time();

        for i in 1:n_qudits^2
            ψ_t = ψ_0[:,i]
            PR_unc[1,i] = 1.
            n1_ = repeat(0:N-1, inner=N)
            n1[1,i] = sum(abs2.(ψ_t) .* n1_)
            n2_ = repeat(0:N-1, outer=N)
            n2[1,i] = sum(abs2.(ψ_t) .* n2_)
            Entropy[1,i] = 0.
            Leakage[1,i] = 0.
            println("State: $(i) / $(n_qudits^2)")
            for k in 2:length(t_interval)
                #Evolution trhough eigenbasis
                c_j = (ψ' * ψ_0[:,i]) .* exp.(-1im*E*t_interval[k]) 
                ψ_cj = ψ .* conj(c_j)'
                ψ_t = sum(ψ_cj, dims=2)
                
                ψ_t = vec(ψ_t)
                PR_unc[k,i] = ( 1 / sum(abs.(ψ_unc' * ψ_t).^4, dims=1))[1] 
                n1_ = repeat(0:N-1, inner=N)
                n1[k,i] = sum(abs2.(ψ_t) .* n1_)
                n2_ = repeat(0:N-1, outer=N)
                n2[k,i] = sum(abs2.(ψ_t) .* n2_)
                Entropy[k,i] = Entanglement_entropy_fock(ψ_t, N)
                Leakage[k,i] = 1 - sum(abs2.(ψ_0' * ψ_t))
            end
        end
        println("Time elapsed: $(time() - t) seconds")
        
        fig = figure(figsize=(17,15), layout= "constrained");
        gs = fig.add_gridspec(2,2);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        labels_ = labels_states_KPOs(n_qudits);
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        elements = [[0,0], [0,1], [1, 0], [1,1]];
        labels_plots = [L"PR_{unc}", L"⟨n_{1}⟩", L"S_{1}", "Leakage"]
        plots_list = [PR_unc, n1, Entropy, Leakage] 
        for i in 1:4
            ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
            for j in 1:n_qudits^2
                plot(t_interval, plots_list[i][:,j], marker = markers_[j], color=colors_[j], label = labels_[j])
            end
            if i == 1
                legend(fontsize=10, shadow=true, loc = "upper right");
            end
            ylabel(labels_plots[i], fontsize=15);
            if i == 3 || i ==4
                xlabel(L"time", fontsize=15);
            else
                xticks([])
            end
        end
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)", fontsize=15);
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/PR_unc_N_$(N)_p_$(p)_n_qudits_$(n_qudits).png")

    end
    Scrambling(p, N)
        
    function Fock0_Scrambling(p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        
        #Uncoupled KPOs
        E_unc, ψ_unc = H_un(p, N);
        
        n_qudits = 2
        aux = zeros(N)
        aux[1] = 1
        ψ_0 = kron(aux, aux);
        

        t_final = 100.
        t_interval = range(0.0, t_final, length=100)
        #ψ_t = [zeros(Complex, N^2) for _ in t_interval]
        PR_unc = zeros(length(t_interval), n_qudits^2);
        Entropy = zeros(length(t_interval), n_qudits^2);
        Leakage = zeros(length(t_interval), n_qudits^2);
        n1, n2 = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
        t = time();

        for i in 1:1
            ψ_t = ψ_0[:,i]
            PR_unc[1,i] = 1.
            n1_ = repeat(0:N-1, inner=N)
            n1[1,i] = sum(abs2.(ψ_t) .* n1_)
            n2_ = repeat(0:N-1, outer=N)
            n2[1,i] = sum(abs2.(ψ_t) .* n2_)
            Entropy[1,i] = 0.
            Leakage[1,i] = 0.
            println("State: $(i) / $(n_qudits^2)")
            for k in 2:length(t_interval)
                #Evolution trhough eigenbasis
                c_j = (ψ' * ψ_0[:,i]) .* exp.(-1im*E*t_interval[k]) 
                ψ_cj = ψ .* conj(c_j)'
                ψ_t = sum(ψ_cj, dims=2)
                
                ψ_t = vec(ψ_t)
                
                PR_unc[k,i] = ( 1 / sum(abs.(ψ_unc' * ψ_t).^4, dims=1))[1] 
                n1_ = repeat(0:N-1, inner=N)
                n1[k,i] = sum(abs2.(ψ_t) .* n1_)
                n2_ = repeat(0:N-1, outer=N)
                n2[k,i] = sum(abs2.(ψ_t) .* n2_)
                Entropy[k,i] = Entanglement_entropy_fock(ψ_t, N)
                Leakage[k,i] = 1 - sum(abs2.(ψ_0' * ψ_t))
            end
        end
        println("Time elapsed: $(time() - t) seconds")
        
        fig = figure(figsize=(17,15), layout= "constrained");
        gs = fig.add_gridspec(2,2);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        labels_ = labels_states_KPOs(n_qudits);
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        elements = [[0,0], [0,1], [1, 0], [1,1]];
        labels_plots = [L"PR_{unc}", L"⟨n_{1}⟩", L"S_{1}", "Leakage"]
        plots_list = [PR_unc, n1, Entropy, Leakage] 
        for i in 1:4
            ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
            for j in 1:1
                plot(t_interval, plots_list[i][:,j], marker = markers_[j], color=colors_[j], label = "|0⟩")
            end
            if i == 1
                legend(fontsize=10, shadow=true, loc = "upper right");
            end
            ylabel(labels_plots[i], fontsize=15);
            if i == 3 || i ==4
                xlabel(L"time", fontsize=15);
            else
                xticks([])
            end
        end
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)", fontsize=15);
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/PR_unc_N_$(N)_p_$(p)_n_Fock_0.png")

    end
    Fock0_Scrambling(p, N)

    function HighE_Scrambling(state, p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);

        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);

        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        
        #Uncoupled KPOs
        E_unc, ψ_unc = H_un(p, N);
        
        n_qudits = 1
        ψ_0 = ψ_unc[:,state];
        

        t_final = 100.
        t_interval = range(0.0, t_final, length=100)
        #ψ_t = [zeros(Complex, N^2) for _ in t_interval]
        PR_unc = zeros(length(t_interval), n_qudits^2);
        Entropy = zeros(length(t_interval), n_qudits^2);
        Leakage = zeros(length(t_interval), n_qudits^2);
        n1, n2 = zeros(length(t_interval), n_qudits^2), zeros(length(t_interval), n_qudits^2);
        t = time();

        for i in 1:1
            ψ_t = ψ_0
            PR_unc[1,i] = 1.
            n1_ = repeat(0:N-1, inner=N)
            n1[1,i] = sum(abs2.(ψ_t) .* n1_)
            n2_ = repeat(0:N-1, outer=N)
            n2[1,i] = sum(abs2.(ψ_t) .* n2_)
            Entropy[1,i] = 0.
            Leakage[1,i] = 0.
            println("State: $(i) / $(n_qudits^2)")
            for k in 2:length(t_interval)
                #Evolution trhough eigenbasis
                c_j = (ψ' * ψ_0[:,i]) .* exp.(-1im*E*t_interval[k]) 
                ψ_cj = ψ .* conj(c_j)'
                ψ_t = sum(ψ_cj, dims=2)
                
                ψ_t = vec(ψ_t)
                
                PR_unc[k,i] = ( 1 / sum(abs.(ψ_unc' * ψ_t).^4, dims=1))[1] 
                n1_ = repeat(0:N-1, inner=N)
                n1[k,i] = sum(abs2.(ψ_t) .* n1_)
                n2_ = repeat(0:N-1, outer=N)
                n2[k,i] = sum(abs2.(ψ_t) .* n2_)
                Entropy[k,i] = Entanglement_entropy_fock(ψ_t, N)
                Leakage[k,i] = 1 - sum(abs2.(ψ_0' * ψ_t))
            end
        end
        println("Time elapsed: $(time() - t) seconds")
        
        fig = figure(figsize=(17,15), layout= "constrained");
        gs = fig.add_gridspec(2,2);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
        elements = [[0,0], [0,1], [1, 0], [1,1]];
        labels_plots = [L"PR_{unc}", L"⟨n_{1}⟩", L"S_{1}", "Leakage"]
        plots_list = [PR_unc, n1, Entropy, Leakage] 
        for i in 1:4
            ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
            for j in 1:1
                plot(t_interval, plots_list[i], marker = markers_[j], color=colors_[j], label = "|$(state)⟩")
            end
            if i == 1
                legend(fontsize=10, shadow=true, loc = "upper right");
            end
            ylabel(labels_plots[i], fontsize=15);
            if i == 3 || i ==4
                xlabel(L"time", fontsize=15);
            else
                xticks()
            end
        end
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)", fontsize=15);
        title("$(round(E_unc[state],digits=3))")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/PR_unc_N_$(N)_p_$(p)_n_HighE_$(state).png")

    end
    HighE_Scrambling(30,p, N)
    
    #Seperate plots
        for i in 1:n_qudits^2
            ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
            plot(t_interval, PR_unc[:,i], marker = markers_[i], color=colors_[i], label = labels_[i])
            legend(fontsize=10, shadow=true, loc = "upper right");
            if isodd(i)
                ylabel(L"PR_{unc}", fontsize=15);
            else
                yticks([])
            end
            if i == 3 || i ==4
                xlabel(L"time", fontsize=15);
            else
                xticks([])
            end
        end
    #
    
###


#------------------------------------- Husimis with Classical E surfaces-------------------------------------
    #Q1Q2
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 0.0);
        N = 100;

        #Uncoupled KPOs
        n_qudits = 4; 
        E_unc, ψ_unc = H_un(p, N, n_states = n_qudits);
        ψ_0 = ψ_unc; #Initial states
        
        roots,E_cl, λs, s_λs = crit_energies(p);
        
        x_lim = 6;
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70; #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q);

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.

        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(n_qudits,n_qudits)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(n_qudits,n_qudits)

        #Multiple Husimi
        c =1
        t = time()
        labels_ = labels_states_KPOs(n_qudits);
        for i in 1:n_qudits
            for j in 1:n_qudits
                    println("State: $(c) / $(n_qudits^2)")
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_0[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.03)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == n_qudits
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("$(labels_[c]), E = $(round(real(E_unc[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", zorder = 3);
                    Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_0[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"q_{2}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == n_qudits
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("$(labels_[c]), E = $(round(real(E_unc[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis/Husimis_q1_q2_p_$(p)_n_qudits_$(n_qudits).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis/Husimis_q1_q2_p_$(p)_n_qudits_$(n_qudits)_2.png")
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_Delmar.png")    
    ###

    #Q1P1
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =(10.0, 1.0, 0.0, 5.0, 1.0, 1.0, 0.0, 5.0, 1.0);
        
        N = 100;
        E_even, ψ_even, E_odd, ψ_odd  = Coupled_kerr(p, N);

        roots,E_cl, λs, s_λs = crit_energies(p)
        
        x_lim = 6
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)

        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));

        #q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);

        Emin, Emax = E_cl[1], 2.


        fig1 = figure(figsize=(12,12), layout= "constrained")
        gs1 = fig1.add_gridspec(2,2)
        fig2 = figure(figsize=(12,12), layout= "constrained")
        gs2 = fig2.add_gridspec(2,2)

        #Multiple Husimi
        c =1
        t = time()
        for i in 1:2
            for j in 1:2
                    println("State: $(c) / 4")
                    ax1 = fig1.add_subplot(gs1[i, j])
                    CS = ax1.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", zorder = 3);
                    Qgrid = Q_function_grid_q1p1_full(QuantumObject(ψ_0[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax1.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]), vmax = 0.001)
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig1.colorbar(im, ax=ax1)
                    if j == 1
                        ax1.set_ylabel(L"p_{1}", fontsize=15)
                    else
                        ax1.set_yticks([])
                    end
                    if i == 5
                        ax1.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax1.set_xticks([])
                    end
                    ax1.set_title("E = $(round(real(E_unc[c]), digits=4))")

                    ax2 = fig2.add_subplot(gs2[i, j])
                    CS = ax2.contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 11), colors="black", zorder = 3);
                    Qgrid = Q_function_grid_q1p1_full(QuantumObject(ψ_0[:,c], dims=(N,N)), q1vals, q2vals, N)
                    #Qgrid = Q_function_grid_q1p1_full(ψcount], q1vals, q2vals, N)
                    im = ax2.imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
                    #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))

                    #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
                    fig2.colorbar(im, ax=ax2)
                    if j == 1
                        ax2.set_ylabel(L"p_{1}", fontsize=15)
                    else
                        ax2.set_yticks([])
                    end
                    if i == 5
                        ax2.set_xlabel(L"q_{1}", fontsize=15)
                    else
                        ax2.set_xticks([])
                    end
                    ax2.set_title("E = $(round(real(E_unc[c]), digits=4))")
                c+=1 
            end
        end
        time() - t
        fig1.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig2.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)")
        fig1.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis/Husimis_q1_p1_p_$(p)_n_qudits_$(n_qudits).png")
        fig2.savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Husimis/Husimis_q1_p1_p_$(p)_n_qudits_$(n_qudits)_2.png")
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum/Husimis_q1_q2_Delmar.png")    
    ##

###


# -------------------------------- Husimi Qudits evolution ---------------------------------------------------
    
    N = 100
    #Definitions
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  0., 1., 0., 5., 0., 1., 0., 5., 6.;
    roots,E_cl, λs, s_λs = crit_energies(p)
    t_final = 10.

    function Classical_surface_q1q2(x_lim)
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2
        E_Contours = H_class([q1, p1, q2, p2],p);
        Emin, Emax = E_cl[1], 2.
        CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 20), colors="black", linewidht=1.0, zorder = 3); 
        for i in 1:length(roots)
            plot(roots[i][1], roots[i][3], marker="o", markersize=7)
        end
 
    end
    function Classical_surface_q1p1(x_lim)
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = coordinates_x,coordinates_y, 0.,0.;#q1,q2
        E_Contours = H_class([q1, p1, q2, p2],p);
        Emin, Emax = E_cl[1], 2.
        CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 20), colors="black", linewidht=1.0, zorder = 3); 
        for i in 1:length(roots)
            if abs(roots[i][3]) < 1e-3 && abs(roots[i][4]) < 1e-3 
                plot_r, = plot(roots[i][1], roots[i][2], marker="o", markersize=7)
            end
        end
    end
    function Classical_surface_p1p2(x_lim)
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = 0., coordinates_x, 0., coordinates_y;#q1,q2
        E_Contours = H_class([q1, p1, q2, p2],p);
        Emin, Emax = E_cl[1], 2.
        CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 20), colors="black", linewidht=1.0, zorder = 3); 
        for i in 1:length(roots)
            if abs(roots[i][1]) < 1e-3 && abs(roots[i][3]) < 1e-3 
                plot_r, = plot(roots[i][2], roots[i][4], marker="o", markersize=7)
            end
        end
    end
    plt.ioff()
    function Husimi_evolution_q1q2(p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        #Uncoupled KPOs
        n_qudits = 2
        E_0, ψ_0 = H_un(p, N, n_states = n_qudits);
        t_interval = range(0.0, t_final, length=100)
        x_lim = 7
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
        #q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);
        Emin, Emax = E_cl[1], 2.
        fig = figure(figsize=(6,5), layout= "constrained")
        CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 20), colors="black", linewidht=1.0, zorder = 3);
        Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_0[:,1], dims=(N,N)), q1vals, q2vals, N)
        imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
        colorbar()
        xlabel(L"q_{1}",fontsize=15)
        ylabel(L"q_{2}",fontsize=15)
        fig.suptitle(L"|C^{+}C^{+⟩}"*" in "*L"Δ_{1}, K_{1}, ξ_{11}, ξ_{21}, Δ_{2}, K_{2}, ξ_{12}, ξ_{22}, γ", fontsize=15);
        title(L" = %$(p)", fontsize=15)
        i = 1
        text(-4.5, 6., "t = $(round(t_interval[i], digits=2))", fontsize=12, verticalalignment="top")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Qubit_evol/q1q2_$(p)_$(i).png")
        close()
        t = time()
        for i in 2:length(t_interval)
            #Evolution trhough eigenbasis
            c_j = (ψ' * ψ_0[:,1]) .* exp.(-1im*E*t_interval[i]) 
            ψ_cj = ψ .* conj(c_j)'
            ψ_t = sum(ψ_cj, dims=2)
            ψ_t = vec(ψ_t)
            fig = figure(figsize=(6,5), layout= "constrained")
            CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 20), colors="black", linewidht=1.0, zorder = 3);
            Qgrid = Q_function_grid_q1q2_full(QuantumObject(ψ_t, dims=(N,N)), q1vals, q2vals, N)
            imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
            #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
            #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
            colorbar()
            xlabel(L"q_{1}",fontsize=15)
            ylabel(L"q_{2}",fontsize=15)
            fig.suptitle(L"|C^{+}C^{+⟩}"*" in "*L"Δ_{1}, K_{1}, ξ_{11}, ξ_{21}, Δ_{2}, K_{2}, ξ_{12}, ξ_{22}, γ", fontsize=15);
            title(L" = %$(p)", fontsize=15)
            text(4.5, 6., "t = $(round(t_interval[i], digits=2))", fontsize=12, verticalalignment="top")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Qubit_evol/q1q2_$(p)_$(i).png")
            close()
        end
        println("Time elapsed: $(time() - t) seconds")
    end
    function Husimi_evolution_q1p1(p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        #Uncoupled KPOs
        n_qudits = 2
        E_0, ψ_0 = H_un(p, N, n_states = n_qudits)
        t_interval = range(0.0, t_final, length=100)
        x_lim = 7
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0; #q1,p1
        E_Contours = H_class([q1, p1, q2, p2],p);
        Emin, Emax = E_cl[1], 2.
        fig = figure(figsize=(6,5), layout= "constrained")
        CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 20), colors="black", linewidht=1.0, zorder = 3);
        Qgrid = Q_function_grid_q1p1_full(QuantumObject(ψ_0[:,1], dims=(N,N)), q1vals, q2vals, N)
        imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
        colorbar()
        xlabel(L"q_{1}",fontsize=15)
        ylabel(L"p_{1}",fontsize=15)
        fig.suptitle(L"|C^{+}C^{+⟩}"*" in "*L"Δ_{1}, K_{1}, ξ_{11}, ξ_{21}, Δ_{2}, K_{2}, ξ_{12}, ξ_{22}, γ", fontsize=15);
        title(L" = %$(p)", fontsize=15)
        i = 1
        text(4.5, 6., "t = $(round(t_interval[i], digits=2))", fontsize=12, verticalalignment="top")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Qubit_evol/q1p1_$(p)_$(i).png")
        close()
        t = time()
        for i in 2:length(t_interval)
            #Evolution trhough eigenbasis
            c_j = (ψ' * ψ_0[:,1]) .* exp.(-1im*E*t_interval[i]) 
            ψ_cj = ψ .* conj(c_j)'
            ψ_t = sum(ψ_cj, dims=2)
            ψ_t = vec(ψ_t)
            fig = figure(figsize=(6,5), layout= "constrained")
            CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 20), colors="black", linewidht=1.0, zorder = 3);
            Qgrid = Q_function_grid_q1p1_full(QuantumObject(ψ_t, dims=(N,N)), q1vals, q2vals, N)
            imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
            #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
            #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
            colorbar()
            xlabel(L"q_{1}",fontsize=15)
            ylabel(L"p_{1}",fontsize=15)
            fig.suptitle(L"|C^{+}C^{+⟩}"*" in "*L"Δ_{1}, K_{1}, ξ_{11}, ξ_{21}, Δ_{2}, K_{2}, ξ_{12}, ξ_{22}, γ", fontsize=15);
            title(L" = %$(p)", fontsize=15)
            text(4.5, 6., "t = $(round(t_interval[i], digits=2))", fontsize=12, verticalalignment="top")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Qubit_evol/q1p1_$(p)_$(i).png")
            close()
        end
        println("Time elapsed: $(time() - t) seconds")
    end
    function Husimi_evolution_p1p2(p, N)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        #Uncoupled KPOs
        n_qudits = 2
        E_0, ψ_0 = H_un(p, N, n_states = n_qudits);
        t_interval = range(0.0, t_final, length=100)
        x_lim = 7
        x = range(-x_lim,x_lim, length=1000);
        y = range(-x_lim,x_lim, length=1000);
        N_Q = 70 #dimension of the Q function N_Q^2
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
        #Equivalent of meshgrid
        coordinates_x = repeat(x', length(x), 1);
        coordinates_y = repeat(y, 1, length(y));
        q1, p1, q2, p2 = 0.,coordinates_x,0.,coordinates_y;
        E_Contours = H_class([q1, p1, q2, p2],p);
        Emin, Emax = E_cl[1], 2.
        fig = figure(figsize=(6,5), layout= "constrained")
        CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 20), colors="black", linewidht=1.0, zorder = 3);
        Qgrid = Q_function_grid_p1p2_full(QuantumObject(ψ_0[:,1], dims=(N,N)), q1vals, q2vals, N)
        imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
        #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
        colorbar()
        xlabel(L"p_{1}",fontsize=15)
        ylabel(L"p_{2}",fontsize=15)
        fig.suptitle(L"|C^{+}C^{+⟩}"*" in "*L"Δ_{1}, K_{1}, ξ_{11}, ξ_{21}, Δ_{2}, K_{2}, ξ_{12}, ξ_{22}, γ", fontsize=15);
        title(L" = %$(p)", fontsize=15)
        i = 1
        text(4.5, 6., "t = $(round(t_interval[i], digits=2))", fontsize=12, verticalalignment="top")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Qubit_evol/p1p2_$(p)_$(i).png")
        close()
        t = time()
        for i in 2:length(t_interval)
            #Evolution trhough eigenbasis
            c_j = (ψ' * ψ_0[:,1]) .* exp.(-1im*E*t_interval[i]) 
            ψ_cj = ψ .* conj(c_j)'
            ψ_t = sum(ψ_cj, dims=2)
            ψ_t = vec(ψ_t)
            fig = figure(figsize=(6,5), layout= "constrained")
            CS = contour(coordinates_x, coordinates_y, E_Contours, range(Emin, 2., length = 20), colors="black", linewidht=1.0, zorder = 3);
            Qgrid = Q_function_grid_p1p2_full(QuantumObject(ψ_t, dims=(N,N)), q1vals, q2vals, N)
            imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
            #imshow(Qgrid,origin="lower",cmap="OrRd",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
            #xticks([0,N_Q/2,N_Q], labels = [q1vals[1],q1vals[Int(N_Q/2)],q1vals[N_Q] ])
            colorbar()
            xlabel(L"p_{1}",fontsize=15)
            ylabel(L"p_{2}",fontsize=15)
            fig.suptitle(L"|C^{+}C^{+⟩}"*" in "*L"Δ_{1}, K_{1}, ξ_{11}, ξ_{21}, Δ_{2}, K_{2}, ξ_{12}, ξ_{22}, γ", fontsize=15);
            title(L" = %$(p)", fontsize=15)
            text(4.5, 6., "t = $(round(t_interval[i], digits=2))", fontsize=12, verticalalignment="top")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Qubit_evol/p1p2_$(p)_$(i).png")
            close()
        end
        println("Time elapsed: $(time() - t) seconds")
    end
    Husimi_evolution_q1q2(p, N)
    Husimi_evolution_q1p1(p, N)
    Husimi_evolution_p1p2(p, N)
    
###


# -------------------------------- Qudits Chaos protection in time 1 ---------------------------------------------------
        
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

    

    N_p = 100
    ps = range(0.01, 11.0, length=N_p)
    F = zeros(N_p);
    Leakage = zeros(N_p);
    Entroy_sum = zeros(N_p);
    PR_sum = zeros(N_p); 
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.;

    #XX gate construction
    Proj_cat = [1,2,N+1,N+2] #cat states positions
    XX_π2 = zeros(Complex, N^2, N^2)
    for i in 1:4
        XX_π2[Proj_cat[i], Proj_cat[i]] = 1/sqrt(2) +0im
        XX_π2[Proj_cat[i], Proj_cat[end - (i-1)]] = 0. + (1/sqrt(2))im
    end
    
    t = time()
    #for k in 1:N_p
    k=10
        γ = ps[k];
        println("$(k) / $(N_p)");
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ;
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        Entropies = zeros(N_states);
        for i in 1:N_states
            Entropies[i] = Entanglement_entropy_fock(ψ[:,i],N)
        end
        #Uncoupled KPOs
        E_unc, ψ_unc = H_un(p, N);
        ψ_0 = ψ_unc[:,1];
        PR = 1 ./ sum((abs.(ψ_unc' * ψ)).^4 , dims=1);
        Entroy_sum[k] = sum(abs2.(ψ' * ψ_0) .* Entropies)
        PR_sum[k] = sum(abs2.(ψ' * ψ_0) .* vec(PR))
        #t_gate = -π/(8*γ*ξ21) 
        t_gate = π/(8*γ*ξ21)
        c_j = (ψ' * ψ_0) .* exp.(-1im*E*t_gate); 
        ψ_cj = ψ .* conj(c_j)';
        ψ_t = vec(sum(ψ_cj, dims=2));
        Leakage[k] = 1 - (abs2.(ψ_unc[:,1]' * ψ_t) + abs2.(ψ_unc[:,N+2]' * ψ_t))
        H_matrix = H_full(p,N)
        H_matrix_unB  = ψ_unc' * H_matrix.data * ψ_unc + 2*ξ21*I(N^2)
        t = time()
        U_H = exp(-1im * H_matrix_unB * t_gate)
        println("Time elapsed: $(time() - t) seconds")

        F[k] = abs2(tr( XX_π2'* U_H ))/16
    #end
    println("Time elapsed: $(time() - t) seconds")


    fig = figure(figsize=(7,14), layout= "constrained");
    gs = fig.add_gridspec(4,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

    ax = fig.add_subplot(element(0,0))
    plot(ps, Entroy_sum, marker = markers_[1], color=colors_[1])
    ylabel(L"S_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(1,0))
    plot(ps, PR_sum, marker = markers_[2], color=colors_[2])
    ylabel(L"PR_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(2,0))
    plot(ps, Leakage, marker = markers_[3], color=colors_[3])
    ylabel(L"Leakage_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(3,0))
    plot(ps, Leakage, marker = markers_[3], color=colors_[3])
    ylabel(L"F_{XX}", fontsize=15);
    xlabel(L"γ", fontsize=15);
    
    
    
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Scrambling_Qubits_chaos_N_$(N)_γ_$(ps[end])_9tg.png")

    
    #Seperate plots
        for i in 1:n_qudits^2
            ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
            plot(t_interval, PR_unc[:,i], marker = markers_[i], color=colors_[i], label = labels_[i])
            legend(fontsize=10, shadow=true, loc = "upper right");
            if isodd(i)
                ylabel(L"PR_{unc}", fontsize=15);
            else
                yticks([])
            end
            if i == 3 || i ==4
                xlabel(L"time", fontsize=15);
            else
                xticks([])
            end
        end
    #
    
###


# -------------------------------- Qudits Chaos protection in time 2 ---------------------------------------------------
        
    function Entanglement_entropy_fock(ψ, N,ismixed = false)
        if ismixed
            ρ_A = zeros(ComplexF64,N,N)
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

    function t_gate_Δs(p)
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr(N,p);

        #Uncoupled KPO
        E_even_un, ψ_even_un, E_odd_un, ψ_odd_un = H_un(Int(N/2),p[1],p[3]);
        

        #Going to full Fock basis
        ψ_S_un, ψ_A_un =zeros(N), zeros(N)
        count=1
        for i in 1:2:N
            ψ_S_un[i] = ψ_even_un[:,1][count]
            ψ_A_un[i+1] = ψ_odd_un[:,1][count]
            count+=1
        end
        Cat_SS, Cat_AA  = kron(ψ_S_un,ψ_S_un), kron(ψ_A_un,ψ_A_un);

        function f_t_gate(t)
            ψ_is = Cat_SS
            ψ =  ψ_even_S * ((ψ_even_S' * ψ_is) .* exp.(1im*E_S_even*t)) 
            ψ +=  ψ_even_A * ((ψ_even_A' * ψ_is) .* exp.(1im*E_A_even*t)) 
            ψ +=  ψ_odd_S * ((ψ_odd_S' * ψ_is) .* exp.(1im*E_S_odd*t)) 
            ψ +=  ψ_odd_A * ((ψ_odd_A' * ψ_is) .* exp.(1im*E_A_odd*t)) 
            return abs2((Cat_SS')*ψ) - abs2((Cat_AA')*ψ)
        end
        t_gate_nD = (π/4)/(2*p[5]*p[3])
        return find_zero(f_t_gate, t_gate_nD, atol = 1e-3)
    end

    

    N_p = 100
    ps = range(0.01, 11.0, length=N_p)
    F = zeros(N_p);
    Leakage = zeros(N_p);
    Entroy_sum = zeros(N_p);
    Entropy_ψ = zeros(N_p);
    PR_sum = zeros(N_p); 
    N = 100
    
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.;

    E_unc, ψ_unc = H_un((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22,0.), N);
    ψ_0 = ψ_unc[:,1];
    ψ_ideal = (ψ_unc[:,1] + 1im * ψ_unc[:,N+2])/sqrt(2);
    
    
    t = time()
    for k in 1:N_p
        γ = ps[k];
        println("$(k) / $(N_p)");
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ;
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        Entropies = zeros(N_states);
        for i in 1:N_states
            Entropies[i] = Entanglement_entropy_fock(ψ[:,i],N)
        end
        #Uncoupled KPOs
        #t_gate = -π/(8*γ*ξ21) 
        t_gate = π/(8*γ*ξ21)
        c_j = (ψ' * ψ_0) .* exp.(-1im*E*t_gate); 
        ψ_cj = ψ .* conj(c_j)';
        ψ_t = vec(sum(ψ_cj, dims=2));
        Entropy_ψ[k] = Entanglement_entropy_fock(ψ_t,N)
        PR_sum[k] = 1 / sum((abs.(ψ_unc' * ψ_t)).^4 , dims=1)[1];
        Leakage[k] = 1 - (abs2.(ψ_unc[:,1]' * ψ_t) + abs2.(ψ_unc[:,N+2]' * ψ_t))
        F[k] = abs2(ψ_t' * ψ_ideal)
        Entroy_sum[k] = sum(abs2.(ψ' * ψ_t) .* Entropies)
    end
    println("Time elapsed: $(time() - t) seconds")

    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/PR_sum.jld", "PR_sum", PR_sum)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Leakage.jld", "Leakage", Leakage)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/F.jld", "F", F)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Entroy_sum.jld", "Entroy_sum", Entroy_sum)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Entropy_ψ.jld", "Entropy_ψ", Entropy_ψ)

    fig = figure(figsize=(7,20), layout= "constrained");
    gs = fig.add_gridspec(5,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

    ax = fig.add_subplot(element(0,0))
    plot(ps, Entroy_sum, marker = markers_[1], color=colors_[1])
    ylabel(L"S_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(1,0))
    plot(ps, Leakage, marker = markers_[2], color=colors_[2])
    ylabel(L"Leakage_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(2,0))
    plot(ps, F, marker = markers_[3], color=colors_[3])
    ylabel(L"F_{XX}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(3,0))
    plot(ps, PR_sum, marker = markers_[4], color=colors_[4])
    ylabel(L"PR_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(4,0))
    plot(ps, Entropy_ψ, marker = markers_[5], color=colors_[5])
    ylabel(L"S", fontsize=15);
    xlabel(L"γ", fontsize=15);
    
    
    
    
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Scrambling_Qubits_chaos_N_$(N)_γ_$(ps[end])_NEW.png")

   
###


# -------------------------------- Qudits Chaos protection in DELTA ---------------------------------------------------
        
    function Entanglement_entropy_fock(ψ, N,ismixed = false)
        if ismixed
            ρ_A = zeros(ComplexF64,N,N)
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

    

    N_p = 100
    ps = range(0.01, 11.0, length=N_p)
    F = zeros(N_p);
    Leakage = zeros(N_p);
    #Entroy_sum = zeros(N_p);
    Entropy_ψ = zeros(N_p);
    PR_sum = zeros(N_p); 
    ts_gate =zeros(N_p);
    N = 100
    
    K1, ξ11, ξ21, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 0., 5., 1.0;
    #K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ =  1., 0., 5., 1., 1., 0., 5., 1.0;
    
    t = time()
    for k in 1:N_p
        Δ1 = Δ2 = ps[k];
        println("$(k) / $(N_p)");
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ;
        E_unc, ψ_unc = H_un((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22,0.), N);
        ψ_0 = ψ_unc[:,1];
        ψ_ideal = (ψ_unc[:,1] + 1im * ψ_unc[:,N+2])/sqrt(2);
        
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #E_even, ψ_even, E_odd, ψ_odd = Coupled_kerr(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        #all_energies = vcat(real.(E_even), real.(E_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        #all_states = hcat(ψ_even, ψ_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];

        function f_t_gate(t)
            c_j = (ψ' * ψ_0) .* exp.(-1im*E*t); 
            ψ_cj = ψ .* conj(c_j)'
            ψ_t = vec(sum(ψ_cj, dims=2))
            return abs2((ψ_0')*ψ_t) - abs2((ψ_unc[:,N+2]')*ψ_t)
        end
        t_gate_guess = π/(8*γ*ξ21)
        t_gate = find_zero(f_t_gate, t_gate_guess, atol = 1e-3)
        ts_gate[k] = t_gate
        c_j = (ψ' * ψ_0) .* exp.(-1im*E*t_gate); 
        ψ_cj = ψ .* conj(c_j)';
        ψ_t = vec(sum(ψ_cj, dims=2));

        #Entropy_ψ[k] = Entanglement_entropy_fock(ψ_t,N)
        PR_sum[k] = 1 / sum((abs.(ψ_unc' * ψ_t)).^4 , dims=1)[1];
        Leakage[k] = 1 - (abs2.(ψ_unc[:,1]' * ψ_t) + abs2.(ψ_unc[:,N+2]' * ψ_t))
        F[k] = abs2(ψ_t' * ψ_ideal)
        #Entroy_sum[k] = sum(abs2.(ψ' * ψ_t) .* Entropies)
    end
    println("Time elapsed: $(time() - t) seconds")

    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/PR_sum_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))_detuned.jld", "PR_sum", PR_sum)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Leakage_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))_detuned.jld", "Leakage", Leakage)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/F_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))_detuned.jld", "F", F)
    save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Tgate_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))_detuned.jld", "ts_gate", ts_gate)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Entroy_sum.jld", "Entroy_sum", Entroy_sum)
    #save("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Entropy_ψ.jld", "Entropy_ψ", Entropy_ψ)

    PR_sum = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/PR_sum_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ)).jld")["PR_sum"]
    Leakage = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Leakage_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ)).jld")["Leakage"]
    F = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/F_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ)).jld")["F"]

    fig = figure(figsize=(7,21), layout= "constrained");
    gs = fig.add_gridspec(3,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

    ax = fig.add_subplot(element(0,0))
    plot(ps, Leakage, marker = markers_[2], color=colors_[2])
    ylabel(L"Leakage_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(1,0))
    plot(ps, F, marker = markers_[3], color=colors_[3])
    ylabel(L"F_{XX}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(2,0))
    plot(ps, PR_sum, marker = markers_[4], color=colors_[4])
    ylabel(L"PR_{++}", fontsize=15);
    xlabel(L"Δ", fontsize=15);
    fig.suptitle("K1, ξ11, ξ21, K2, ξ12, ξ22, γ = $((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))", fontsize=15);
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Scrambling_Qubits_chaos_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))_$(N)_Δ_detuned.png")
    close()
###

# -------------------------------- Qudits Chaos protection in DELTA and gamma ---------------------------------------------------
    
    ##CODE
        N_p = 100
        ps = range(0.01, 11.0, length=N_p)
        F = zeros(N_p, N_p);
        Leakage = zeros(N_p, N_p);
        PR_sum = zeros(N_p, N_p); 
        N = 100
        
        K1, ξ11, ξ21, K2, ξ12, ξ22 =  1., 0., 5., 1., 0., 5.;
            
        t = time()
        for j in 1:N_p
            γ = ps[j];
            for k in 1:N_p
                Δ1 = Δ2 = ps[k];
                println("$(k) / $(N_p)");
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ;
                E_unc, ψ_unc = H_un((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22,0.), N);
                ψ_0 = ψ_unc[:,1];
                ψ_ideal = (ψ_unc[:,1] + 1im * ψ_unc[:,N+2])/sqrt(2);
                
                E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
                #Organizing 
                all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
                all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
                # Get sorting indices for energies (ascending order)
                sorted_indices = sortperm(all_energies);
                E = all_energies[sorted_indices];
                ψ = all_states[:, sorted_indices];
                N_states = 4000 #From Convergence test
                #Fixing states below E = 0 (saddle point at (0,0,0,0))
                E = E[1:N_states];
                ψ = ψ[:,1:N_states];
                #Entropies = zeros(N_states);
                #for i in 1:N_states
                #    Entropies[i] = Entanglement_entropy_fock(ψ[:,i],N)
                #end 

                function f_t_gate(t)
                    c_j = (ψ' * ψ_0) .* exp.(-1im*E*t); 
                    ψ_cj = ψ .* conj(c_j)'
                    ψ_t = vec(sum(ψ_cj, dims=2))
                    return abs2((ψ_0')*ψ_t) - abs2((ψ_unc[:,N+2]')*ψ_t)
                end
                t_gate_guess = π/(8*γ*ξ21)
                t_gate = find_zero(f_t_gate, t_gate_guess, atol = 1e-3)

                c_j = (ψ' * ψ_0) .* exp.(-1im*E*t_gate); 
                ψ_cj = ψ .* conj(c_j)';
                ψ_t = vec(sum(ψ_cj, dims=2));

                #Entropy_ψ[k] = Entanglement_entropy_fock(ψ_t,N)
                PR_sum[k] = 1 / sum((abs.(ψ_unc' * ψ_t)).^4 , dims=1)[1];
                Leakage[k] = 1 - (abs2.(ψ_unc[:,1]' * ψ_t) + abs2.(ψ_unc[:,N+2]' * ψ_t))
                F[k] = abs2(ψ_t' * ψ_ideal)
                #Entroy_sum[k] = sum(abs2.(ψ' * ψ_t) .* Entropies)
            end
        end
        println("Time elapsed: $(time() - t) seconds")

        N_p=100
        F = zeros(N_p, N_p);
        Leakage = zeros(N_p, N_p);
        PR_sum = zeros(N_p, N_p);
        ts_gate = zeros(N_p, N_p);
        N=100
            data_miss= []
            for job in 1:100
                try
                    Leakage += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/Leakage_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["Leakage"]
                    F += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/F_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["F"]
                    PR_sum += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/PR_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["PR_sum"]
                    ts_gate += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/Tgate_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["ts_gate"]
                catch
                    #println("Missing job $(job)")
                    push!(data_miss, job)
                end    
            end
            println("data_miss = $(data_miss)")
            println(length(data_miss))
            Leakage
                
        PR_sum = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/PR_sum_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ)).jld")["PR_sum"]
        Leakage = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/Leakage_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ)).jld")["Leakage"]
        F = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Qubit_scrambling/F_Δ_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ)).jld")["F"]

        fig = figure(figsize=(7,21), layout= "constrained");
        gs = fig.add_gridspec(3,1);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        
        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

        ax = fig.add_subplot(element(0,0))
        plot(ps, Leakage, marker = markers_[2], color=colors_[2])
        ylabel(L"Leakage_{++}", fontsize=15);
        
        xticks([])
        ax = fig.add_subplot(element(1,0))
        plot(ps, F, marker = markers_[3], color=colors_[3])
        ylabel(L"F_{XX}", fontsize=15);
        xticks([])
        ax = fig.add_subplot(element(2,0))
        plot(ps, PR_sum, marker = markers_[4], color=colors_[4])
        ylabel(L"PR_{++}", fontsize=15);
        xlabel(L"Δ", fontsize=15);
        fig.suptitle("K1, ξ11, ξ21, K2, ξ12, ξ22, γ = $((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))", fontsize=15);
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Scrambling_Qubits_chaos_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22, γ))_$(N)_Δ.png")
    ###


    ## PLots from Slurm
        N_p = 100
        ps = range(0.01, 11.0, length=N_p)
        F = zeros(N_p, N_p);
        Leakage = zeros(N_p, N_p);
        PR_sum = zeros(N_p, N_p); 
        ts_gate = zeros(N_p, N_p); 
        N = 100
        
        K1, ξ11, ξ21, K2, ξ12, ξ22 =  1., 0., 5., 1., 0., 5.;
            
        t = time()
        for job in 1:N_p
            try
                F += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/F_p_$(( K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["F"];
                Leakage += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/Leakage_p_$(( K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["Leakage"];
                PR_sum += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/PR_p_$(( K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["PR_sum"];
                ts_gate += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Leakage/Tgate_p_$(( K1, ξ11, ξ21, K2, ξ12, ξ22))_N_$(N)_job_$(job).jld")["ts_gate"];
            catch
                println("$(job)")
            end
        end
        println("Time elapsed: $(time() - t) seconds")
        
        fig = figure(figsize=(10,10), layout= "constrained");
        gs = fig.add_gridspec(2,2);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        cmap_ = "cividis"
        ax = fig.add_subplot(element(0,0))
        
        im = imshow(Leakage,origin="lower",cmap=cmap_, extent=(0.01,11., 0.01,11.), vmax = 0.1)
        #cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
        cbar = plt.colorbar(im)#, cax = cax_)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(L"Leakage_{++}", fontsize=20)
        #bar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
        #cbar.ax.tick_params(axis="x", labelsize=15)
        ylabel(L"Δ", fontsize=15);
        xticks([])

        ax = fig.add_subplot(element(1,0))
        im = imshow(F,origin="lower",cmap=cmap_, extent=(0.01,11., 0.01,11.))#,vmax = 24)
        #cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
        cbar = plt.colorbar(im)#, cax = cax_)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(L"F_{XX}", fontsize=20)
        #bar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
        #cbar.ax.tick_params(axis="x", labelsize=15)
        ylabel(L"Δ", fontsize=15);
        xlabel(L"γ", fontsize=15);
        
        ax = fig.add_subplot(element(0,1))
        im = imshow(ts_gate,origin="lower",cmap=cmap_, extent=(0.01,11., 0.01,11.))#,vmax = 24)
        #cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
        cbar = plt.colorbar(im)#, cax = cax_)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(L"t_{g}", fontsize=20)
        #bar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
        #cbar.ax.tick_params(axis="x", labelsize=15)
        xticks([])
        yticks([])
        
        ax = fig.add_subplot(element(1,1))
        im = imshow(PR_sum,origin="lower",cmap=cmap_, extent=(0.01,11., 0.01,11.))#,vmax = 24)
        #cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
        cbar = plt.colorbar(im)#, cax = cax_)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(L"PR", fontsize=20)
        #bar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
        #cbar.ax.tick_params(axis="x", labelsize=15)
        yticks([])
        xlabel(L"γ", fontsize=15);

        fig.suptitle("K1, ξ11, ξ21, K2, ξ12, ξ22 = $((K1, ξ11, ξ21, K2, ξ12, ξ22))", fontsize=15);
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Scrambling_Qubits_chaos_p_$((K1, ξ11, ξ21, K2, ξ12, ξ22))_$(N)_γΔ.png")                
    ##
###


# -------------------------------- |++> evolution ---------------------------------------------------
    #Fotoc for random state =  9794.542962686817
        
    function Entanglement_entropy_fock(ψ, N,ismixed = false)
        if ismixed
            ρ_A = zeros(ComplexF64,N,N)
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

    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p =  1.0, 1., 0., 5., 1.0, 1., 0., 5., 1.;
     
    E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
    #Organizing 
    all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
    all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
    # Get sorting indices for energies (ascending order)
    sorted_indices = sortperm(all_energies);
    E = all_energies[sorted_indices];
    ψ = all_states[:, sorted_indices];
    N_states = 4000 #From Convergence test
    #Fixing states below E = 0 (saddle point at (0,0,0,0))
    E = E[1:N_states];
    ψ = ψ[:,1:N_states];
    E_unc, ψ_unc = H_un((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22,0.), N);
    ψ_0 = ψ_unc[:,1];
    ψ_ideal = (ψ_unc[:,1] + 1im * ψ_unc[:,N+2])/sqrt(2);


    function f_t_gate(t)
        c_j = (ψ' * ψ_0) .* exp.(-1im*E*t); 
        ψ_cj = ψ .* conj(c_j)'
        ψ_t = vec(sum(ψ_cj, dims=2))
        return abs2((ψ_0')*ψ_t) - abs2((ψ_unc[:,N+2]')*ψ_t)
    end
    t_gate_guess = π/(8*γ*ξ21)
    t_gate = find_zero(f_t_gate, t_gate_guess, atol = 1e-3)
    t_final = 4 * t_gate
    t_interval = range(0.0, t_final, length=100)
    n1, n2, n = zeros(length(t_interval)), zeros(length(t_interval)), zeros(length(t_interval));
    Entropy, Leakage = zeros(length(t_interval)), zeros(length(t_interval));
    pp_state, mm_state = zeros(length(t_interval)), zeros(length(t_interval));
    t = time();
    ψ_t = ψ_0 
    k = 1
    n1_ = repeat(0:N-1, inner=N)
    n1[k] = sum(abs2.(ψ_t) .* n1_)
    n2_ = repeat(0:N-1, outer=N)
    n2[k] = sum(abs2.(ψ_t) .* n2_)
    n[k] = n1[k] + n2[k]
    pp_state[k], mm_state[k] = abs2((ψ_0')*ψ_t), abs2((ψ_unc[:,N+2]')*ψ_t)

    for k in 2:length(t_interval)
        #Evolution trhough eigenbasis
        c_j = (ψ' * ψ_0) .* exp.(-1im*E*t_interval[k]) 
        ψ_cj = ψ .* conj(c_j)'
        ψ_t = sum(ψ_cj, dims=2)
        
        ψ_t = vec(ψ_t)
        
        n1_ = repeat(0:N-1, inner=N)
        n1[k] = sum(abs2.(ψ_t) .* n1_)
        n2_ = repeat(0:N-1, outer=N)
        n2[k] = sum(abs2.(ψ_t) .* n2_)
        n[k] = n1[k] + n2[k]
        pp_state[k], mm_state[k] = abs2((ψ_0')*ψ_t), abs2((ψ_unc[:,N+2]')*ψ_t)
        Entropy[k] = Entanglement_entropy_fock(ψ_t, N)
        Leakage[k] = 1 - sum(abs2.(ψ_0' * ψ_t)) - abs2.(ψ_unc[:,N+2]' * ψ_t)
    end
    
    println("Time elapsed: $(time() - t) seconds")
    plot(t_interval, pp_state, label="++ state")
    plot(t_interval, mm_state, label="-- state")
    
    fig = figure(figsize=(10,20), layout= "constrained");
    gs = fig.add_gridspec(4,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    labels_ = labels_states_KPOs(2);
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
    elements = [[0,0], [1,0], [2, 0], [3,0]];
    labels_plots = [L"|ψ|^2", L"⟨n_{1}⟩", L"S_{1}", "Leakage"]
    plots_list = [[pp_state, mm_state], [n1, n2, n], Entropy, Leakage] 
    for i in 1:4
        ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
        if  i == 1 
            plot(t_interval, plots_list[i][1], color="red", label = labels_[1])
            plot(t_interval, plots_list[i][2], color="blue", label = labels_[4])
        elseif i == 2
            plot(t_interval, plots_list[i][1], color="red", label = L"⟨n_{1}⟩")
            plot(t_interval, plots_list[i][2], color="blue", label = L"⟨n_{2}⟩")
            plot(t_interval, plots_list[i][3], color="green", label = L"⟨n⟩")
        else
            plot(t_interval, plots_list[i], color=colors_[i])
        end
        if i == 1
            legend(fontsize=10, shadow=true, loc = "upper right");
        end
        ylabel(labels_plots[i], fontsize=15);
        if  i ==4
            xlabel(L"time", fontsize=15);
        else
            xticks([])
        end
    end
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)", fontsize=15);
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/GateXX_N_$(N)_p_$(p).png")

###


# --------------------------------  Chaos comparision in time ---------------------------------------------------
        
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

    

    N_p = 100
    ps = range(0.01, 11.0, length=N_p)
    F = zeros(N_p);
    Leakage = zeros(N_p);
    Entroy_sum = zeros(N_p);
    PR_sum = zeros(N_p); 
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.;

    #XX gate construction
    Proj_cat = [1,2,N+1,N+2] #cat states positions
    XX_π2 = zeros(ComplexF64, N^2, N^2)
    for i in 1:4
        XX_π2[Proj_cat[i], Proj_cat[i]] = 1/sqrt(2) +0im
        XX_π2[Proj_cat[i], Proj_cat[end - (i-1)]] = 0. + (1/sqrt(2))im
    end
    
    t = time()
    for k in 1:N_p

        γ = ps[k];
        println("$(k) / $(N_p)");
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ;
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        Entropies = zeros(N_states);
        for i in 1:N_states
            Entropies[i] = Entanglement_entropy_fock(ψ[:,i],N)
        end
        #Uncoupled KPOs
        E_unc, ψ_unc = H_un(p, N);
        ψ_0 = ψ_unc[:,1];
        PR = 1 ./ sum((abs.(ψ_unc' * ψ)).^4 , dims=1);
        Entroy_sum[k] = sum(abs2.(ψ' * ψ_0) .* Entropies)
        PR_sum[k] = sum(abs2.(ψ' * ψ_0) .* vec(PR))
        #t_gate = -π/(8*γ*ξ21) 
        t_gate = π/(8*γ*ξ21)
        c_j = (ψ' * ψ_0) .* exp.(-1im*E*t_gate); 
        ψ_cj = ψ .* conj(c_j)';
        ψ_t = vec(sum(ψ_cj, dims=2));
        Leakage[k] = 1 - (abs2.(ψ_unc[:,1]' * ψ_t) + abs2.(ψ_unc[:,N+2]' * ψ_t))
        H_matrix = H_full(p,N)
        H_matrix_unB  = ψ_unc' * H_matrix.data * ψ_unc + 2*ξ21*I(N^2)
        t = time()
        U_H = exp(-1im * H_matrix_unB * t_gate)
        println("Time for exp: $(time() - t) seconds")
        F[k] = abs2(tr( XX_π2'* U_H ))/16
        println("Time for F: $(time() - t) seconds")
        using SparseArrays
        A_sparse = sparse(XX_π2)                # XX_π2 is already a matrix in your file
        UH_mat = typeof(U_H) <: AbstractMatrix ? U_H : Matrix(U_H)   # convert custom operator -> dense matrix
        UH_sparse = sparse(U_H)
        abs2(tr( A_sparse'* UH_sparse  ))/16
        
    end
    println("Time elapsed: $(time() - t) seconds")


    fig = figure(figsize=(7,14), layout= "constrained");
    gs = fig.add_gridspec(4,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

    ax = fig.add_subplot(element(0,0))
    plot(ps, Entroy_sum, marker = markers_[1], color=colors_[1])
    ylabel(L"S_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(1,0))
    plot(ps, PR_sum, marker = markers_[2], color=colors_[2])
    ylabel(L"PR_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(2,0))
    plot(ps, Leakage, marker = markers_[3], color=colors_[3])
    ylabel(L"Leakage_{++}", fontsize=15);
    xticks([])
    ax = fig.add_subplot(element(3,0))
    plot(ps, Leakage, marker = markers_[3], color=colors_[3])
    ylabel(L"F_{XX}", fontsize=15);
    xlabel(L"γ", fontsize=15);
    
    
    
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling/Scrambling_Qubits_chaos_N_$(N)_γ_$(ps[end])_9tg.png")

    
    #Seperate plots
        for i in 1:n_qudits^2
            ax = fig.add_subplot(element(elements[i][1],elements[i][2]));
            plot(t_interval, PR_unc[:,i], marker = markers_[i], color=colors_[i], label = labels_[i])
            legend(fontsize=10, shadow=true, loc = "upper right");
            if isodd(i)
                ylabel(L"PR_{unc}", fontsize=15);
            else
                yticks([])
            end
            if i == 3 || i ==4
                xlabel(L"time", fontsize=15);
            else
                xticks([])
            end
        end
    #
    
###


# --------------------------------  Maximum Chaos (UNC vs Coupl basis) ---------------------------------------------------

    N_p = 100
    ps = range(11.0, 100.0, length=N_p)
    PR_int = zeros(N_p);
    PR_unc = zeros(N_p); 
    N = 100
    Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 =  0., 1., 0., 5., 0., 1., 0., 5.;

    t = time()
    for k in 1:N_p
        γ = ps[k]
        println("$(k) / $(N_p)")
        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ;
        #interaction Hamiltonian
        p_int = 0., 0., 0., 0., 0., 0., 0., 0., γ
        H_int = H_full(p_int, N)
        
        E_int, ψ_int = eigen(real(Matrix(H_int.data)))
        #Uncoupled KPOs
        E_unc, ψ_unc = H_un(p, N);
        
        
        E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd = Coupled_kerr_equiv(p, N);
        #Organizing 
        all_energies = vcat(real.(E_S_even), real.(E_A_even), real.(E_S_odd), real.(E_A_odd));
        all_states = hcat(ψ_S_even, ψ_A_even, ψ_S_odd, ψ_A_odd);
        # Get sorting indices for energies (ascending order)
        sorted_indices = sortperm(all_energies);
        E = all_energies[sorted_indices];
        ψ = all_states[:, sorted_indices];
        N_states = 4000 #From Convergence test
        #Fixing states below E = 0 (saddle point at (0,0,0,0))
        E = E[1:N_states];
        ψ = ψ[:,1:N_states];
        PR_unc[k] = mean(1 ./ sum((abs.(ψ_unc' * ψ)).^4 , dims=1))
        PR_int[k] = mean(1 ./ sum((abs.(ψ_int' * ψ)).^4 , dims=1))
    end
    println("Time elapsed: $(time() - t) seconds")


    fig = figure(figsize=(7,7), layout= "constrained");
    gs = fig.add_gridspec(1,1);
    element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
    slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
    
    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];

    ax = fig.add_subplot(element(0,0))
    plot(ps, PR_int, marker = markers_[1], color=colors_[1], label= "Interaction basis")
    plot(ps, PR_unc, marker = markers_[2], color=colors_[2], label= "Uncoupled basis")
    plot(ps, PR_unc, marker = markers_[2], color=colors_[2])
    ylabel(L"⟨PR⟩", fontsize=15);
    xlabel(L"γ", fontsize=15);
    legend(fontsize=10, shadow=true, loc = "upper right");
    
    
    
    fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))", fontsize=15);
    
    savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Quantum_figures/Scrambling_Qubits_chaos_N_$(N)_γ_46.0s6_191.887_.png")

###



