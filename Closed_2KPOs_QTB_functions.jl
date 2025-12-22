module Coupled_KPOs_QTB_functions
    using LinearAlgebra
    using PyPlot
    using PyCall
    using BenchmarkTools
    using LaTeXStrings
    using QuantumToolbox, NLsolve
    export H_full ,Parity_matrices, Coupled_kerr, H_un,  Coupled_kerr_equiv, Q_function_grid_q1q2_full, Q_function_grid_q1p1_full,
            Q_function_grid_q2p2_full, Q_function_grid_p1p2_full, parities, total_fock, Convergency_test, H_un, labels_states_KPOs,
            Q_function_grid_q1q2_full_ρ, Q_function_grid_q1p1_full_ρ, Q_function_grid_q2p2_full_ρ, Q_function_grid_p1p2_full_ρ,
            labels_states_KPOs, crit_energies, H_class, Jacobian_qp, EqM_2, classify_fixed_point, Q_function_full,
            Wehrl_entropy_q1q2, Wehrl_entropy

    function Parity_matrices(basis)
            size_b = size(basis)[1]
            P = zeros(size_b, size_b)
            #Construting the Parity operator
            for i in 1:size_b
                for j in 1:size_b
                    if basis[i][1] == basis[j][2] && basis[i][2] == basis[j][1]
                        P[i, j] = 1
                    end
                end
            end
            #λ,v =  eigen(P, sortby = real)
            λ,v =  eigen(P)

            A_matrix = v[:,λ .<0]
            S_matrix = v[:,λ .>0]
            
            return S_matrix, A_matrix
    end
    
    function H_full(p,N)
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p
        a1, ad1, a2, ad2 = tensor(destroy(N), qeye(N)), tensor(create(N), qeye(N)), tensor(qeye(N), destroy(N)), tensor(qeye(N), create(N));
        n1, n2 = tensor(num(N), qeye(N)), tensor(qeye(N), num(N));
        H = K1*ad1^2 * a1^2 - ξ11 * (a1 + ad1) - ξ21 * (a1^2 + ad1^2) - Δ1 * n1 +
            K2*ad2^2 * a2^2 - ξ12 * (a2 + ad2) - ξ22 * (a2^2 + ad2^2) - Δ2 * n2 - 
            γ*(ad1*a2 + ad2*a1);
        return H
    end


    function Q_function_grid_q1q2_full(Ψ, q1vals, q2vals, N)
        Q = zeros(Float64, length(q1vals), length(q2vals))
        for (j, q1) in enumerate(q1vals)
            for (i, q2) in enumerate(q2vals)
                α1 = (1/sqrt(2))*(q1+ (0.)*im)
                α2 = (1/sqrt(2))*(q2+ (0.)*im)
                Q[i, j] = Q_function_full(Ψ,α1, α2, N)
            end
        end
        return Q
    end
    function Q_function_grid_q1p1_full(Ψ, q1vals, p1vals, N)
        Q = zeros(Float64, length(q1vals), length(p1vals))
        for (j, q1) in enumerate(q1vals)
            for (i, p1) in enumerate(p1vals)
                α1 = (1/sqrt(2))*(q1 + p1*im)
                α2 = 0. + (0.)*im
                Q[i, j] = Q_function_full(Ψ,α1, α2, N)
            end
        end
        return Q
    end
    function Q_function_grid_q2p2_full(Ψ, q2vals, p2vals, N)
        Q = zeros(Float64, length(q2vals), length(p2vals))
        for (j, q2) in enumerate(q2vals)
            for (i, p2) in enumerate(p2vals)
                α1 = 0. + (0.)*im
                α2 = (1/sqrt(2))*(q2 + p2*im)
                Q[i, j] = Q_function_full(Ψ,α1, α2, N)
            end
        end
        return Q
    end
    function Q_function_grid_p1p2_full(Ψ, p1vals, p2vals, N)
        Q = zeros(Float64, length(p1vals), length(p2vals))
        for (j, p1) in enumerate(p1vals)
            for (i, p2) in enumerate(p2vals)
                α1 = (1/sqrt(2))*(0. + p1*im)
                α2 = (1/sqrt(2))*(0. + p2*im)
                Q[i, j] = Q_function_full(Ψ,α1, α2, N)
            end
        end
        return Q
    end


    function Q_function_grid_q1q2_full_ρ(ρ, q1vals, q2vals, N)
        Q = zeros(Float64, length(q1vals), length(q2vals))
        for (j, q1) in enumerate(q1vals)
            for (i, q2) in enumerate(q2vals)
                α1 = (1/sqrt(2))*(q1+ (0.)*im)
                α2 = (1/sqrt(2))*(q2+ (0.)*im)
                Q[i, j] = Q_function_full_ρ(ρ,α1, α2, N)
            end
        end
        return Q
    end
    function Q_function_grid_q1p1_full_ρ(ρ, q1vals, p1vals, N)
        Q = zeros(Float64, length(q1vals), length(p1vals))
        for (j, q1) in enumerate(q1vals)
            for (i, p1) in enumerate(p1vals)
                α1 = (1/sqrt(2))*(q1 + p1*im)
                α2 = 0. + (0.)*im
                Q[i, j] = Q_function_full_ρ(ρ,α1, α2, N)
            end
        end
        return Q
    end
    function Q_function_grid_q2p2_full_ρ(ρ, q2vals, p2vals, N)
        Q = zeros(Float64, length(q2vals), length(p2vals))
        for (j, q2) in enumerate(q2vals)
            for (i, p2) in enumerate(p2vals)
                α1 = 0. + (0.)*im
                α2 = (1/sqrt(2))*(q2 + p2*im)
                Q[i, j] = Q_function_ρ(ρ,α1, α2, N)
            end
        end
        return Q
    end
    function Q_function_grid_p1p2_full_ρ(ρ, p1vals, p2vals, N)
        Q = zeros(Float64, length(p1vals), length(p2vals))
        for (j, p1) in enumerate(p1vals)
            for (i, p2) in enumerate(p2vals)
                α1 = (1/sqrt(2))*(0. + p1*im)
                α2 = (1/sqrt(2))*(0. + p2*im)
                Q[i, j] = Q_function_ρ(ρ,α1, α2,even, N)
            end
        end
        return Q
    end


    function H_un(p,N; n_states = N)
        """
            Uncoupled H of 2 KPOs
            p :: indivuals parameters Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ 
            N :: size
            n_state :: how many states from both Kpos to considered
            (default) is all. 

            return
            E_unc :: Energies of the combine KPOs
            ψ_unc :: Complete Eigenstates of the combine KPOs
        """
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p
        a, ad, n = destroy(N), create(N), num(N);
        H_unc1 = K1*ad^2 * a^2 - ξ21 * (a^2 + ad^2) - ξ11 * (a + ad) - Δ1 * n;
        H_unc2 = K2*ad^2 * a^2 - ξ22 * (a^2 + ad^2) - ξ12 * (a + ad) - Δ2 * n;

        #If linear Drive is present the H is not symmetric anymore (N = even, odd)
        if ξ11 == 0. && ξ12 == 0.
            base_unc = 0:N-1;
            bin_even_unc, bin_odd_unc = base_unc .% 2 .!= 1, base_unc .% 2 .== 1;
            #Using QTB
            #H_even_unc1, H_odd_unc1 = Qobj(H_unc1[bin_even_unc,:][:,bin_even_unc]), Qobj(H_unc1[bin_odd_unc,:][:,bin_odd_unc]);
            #H_even_unc2, H_odd_unc2 = Qobj(H_unc2[bin_even_unc,:][:,bin_even_unc]), Qobj(H_unc2[bin_odd_unc,:][:,bin_odd_unc]);
            #Using LA
            H_even_unc1, H_odd_unc1 = Matrix(H_unc1[bin_even_unc,:][:,bin_even_unc]), Matrix(H_unc1[bin_odd_unc,:][:,bin_odd_unc]);
            H_even_unc2, H_odd_unc2 = Matrix(H_unc2[bin_even_unc,:][:,bin_even_unc]), Matrix(H_unc2[bin_odd_unc,:][:,bin_odd_unc]);
            #Diagonalization
            E_even_unc1, ψ_even_unc1_ = eigen(H_even_unc1);
            E_odd_unc1, ψ_odd_unc1_ = eigen(H_odd_unc1);
            E_even_unc2, ψ_even_unc2_ = eigen(H_even_unc2);
            E_odd_unc2, ψ_odd_unc2_ = eigen(H_odd_unc2);
            #From Symmetry to Complete Fock
            ψ_even_unc1 = zeros(N,Int(N/2));
            ψ_even_unc1[bin_even_unc .> 0, :] = ψ_even_unc1_;
            ψ_odd_unc1 = zeros(N,Int(N/2));
            ψ_odd_unc1[bin_odd_unc .> 0, :] = ψ_odd_unc1_;
            ψ_even_unc2 = zeros(N,Int(N/2));
            ψ_even_unc2[bin_even_unc .> 0, :] = ψ_even_unc2_;
            ψ_odd_unc2 = zeros(N,Int(N/2));
            ψ_odd_unc2[bin_odd_unc .> 0, :] = ψ_odd_unc2_;

            # Compbining simmetries into a single state matrix
            ψ_unc1, ψ_unc2 = zeros(N,N), zeros(N,N)
            ψ_unc1[:,1:2:N], ψ_unc1[:,2:2:N] = ψ_even_unc1, ψ_odd_unc1
            ψ_unc2[:,1:2:N], ψ_unc2[:,2:2:N] = ψ_even_unc2, ψ_odd_unc2
            ψ_unc = kron(ψ_unc1[:,1:n_states],ψ_unc2[:,1:n_states])

            #Energies
            E_unc1, E_unc2 = zeros(N), zeros(N)
            E_unc1[1:2:N], E_unc1[2:2:N] = E_even_unc1, E_odd_unc1
            E_unc2[1:2:N], E_unc2[2:2:N] = E_even_unc2, E_odd_unc2
            E_unc = vec(E_unc1[1:n_states] .+ E_unc2[1:n_states]')

            return E_unc, ψ_unc, E_unc1, ψ_unc1, E_unc2, ψ_unc2
        else
            E1, ψ1_ = eigen(H_unc1);
            E2, ψ2_ = eigen(H_unc2);

            # Compbining simmetries into a single state matrix
            ψ_unc = kron(ψ1_[:,1:n_states],ψ2_[:,1:n_states])

            #Energies
            E_unc1, E_unc2 = zeros(N), zeros(N)
            E_unc = vec(E1[1:n_states] .+ E2[1:n_states]')

            return E_unc, ψ_unc
        end
    end

    function Coupled_kerr_equiv(p,N)
        """
            H of the equivalent coupled Kerr
            Δ1, ξ21, K1, = Δ2, ξ22, K2 = Δ, ξ2, K   
            
        """
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p
        if Δ1 != Δ2 || ξ21 != ξ22 || K1 != K2 || ξ11 != 0. || ξ12 != 0.
            return println("Not equivalent KPOs, use another H")
        end
        H = H_full(p,N)
        n_base = [[n1, n2] for n1 in 0:(N-1) for n2 in 0:(N-1)] 
        sum_n_base = zeros(N*N)
        for i in 1:N*N
            sum_n_base[i] = sum(n_base[i])   
        end
        bin_even = sum_n_base .% 2 .!= 1
        bin_odd = sum_n_base .% 2 .== 1
        base_odd = n_base[findall(isodd,sum_n_base)]
        base_even = n_base[findall(iseven,sum_n_base)]
        H_even = H[bin_even,:][:,bin_even]
        H_odd = H[bin_odd,:][:,bin_odd]

        S_even, A_even = Parity_matrices(base_even)
        H_S_even = S_even'*H_even*S_even
        H_A_even = A_even'*H_even*A_even
        S_odd, A_odd = Parity_matrices(base_odd)
        H_S_odd = S_odd'*H_odd*S_odd
        H_A_odd = A_odd'*H_odd*A_odd

        E_S_even, ψ_S_even_ = eigen(H_S_even, sortby=real);
        E_A_even, ψ_A_even_ = eigen(H_A_even, sortby=real);
        E_S_odd, ψ_S_odd_ = eigen(H_S_odd, sortby=real);
        E_A_odd, ψ_A_odd_ = eigen(H_A_odd, sortby=real);

        #From Symmetry to total Fock N,N
        ψ_S_even = zeros(N^2,size(ψ_S_even_)[1])
        ψ_S_even[bin_even .> 0, :] = S_even * ψ_S_even_
        ψ_A_even = zeros(N^2,size(ψ_A_even_)[1])
        ψ_A_even[bin_even .> 0, :] = A_even * ψ_A_even_
        
        ψ_S_odd = zeros(N^2,size(ψ_S_odd_)[1])
        ψ_S_odd[bin_odd .> 0, :] = S_odd * ψ_S_odd_
        ψ_A_odd = zeros(N^2,size(ψ_A_odd_)[1])
        ψ_A_odd[bin_odd .> 0, :] = A_odd * ψ_A_odd_

        return E_S_even, ψ_S_even, E_A_even, ψ_A_even, E_S_odd, ψ_S_odd, E_A_odd, ψ_A_odd 
    end
    
    function Coupled_kerr(p,N)
        """
            H of the equivalent coupled Kerr
            Δ1, ξ21, K1, = Δ2, ξ22, K2 = Δ, ξ2, K   
            
        """
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p
        if Δ1 == Δ2 && ξ21 == ξ22 && K1 == K2 && ξ11 == 0. && ξ12 == 0.
            return println("Equivalent KPOs, use another H because of additional symmetries")
        end

        #If linear Drive is present the H is not symmetric anymore (N = even, odd)
        if ξ11 == 0. && ξ12 == 0.
            H = H_full(p,N)
            n_base = [[n1, n2] for n1 in 0:(N-1) for n2 in 0:(N-1)] 
            sum_n_base = zeros(N*N)
            for i in 1:N*N
                sum_n_base[i] = sum(n_base[i])   
            end
            bin_even = sum_n_base .% 2 .!= 1
            bin_odd = sum_n_base .% 2 .== 1
            H_even = H[bin_even,:][:,bin_even]
            H_odd = H[bin_odd,:][:,bin_odd]

            E_even, ψ_even_ = eigen(Matrix(H_even), sortby=real);
            E_odd, ψ_odd_ = eigen(Matrix(H_odd), sortby=real);
            
            #From Symmetry to total Fock N,N
            ψ_even = zeros(N^2,size(ψ_even_)[1])
            ψ_even[bin_even .> 0, :] = ψ_even_
            
            ψ_odd = zeros(N^2,size(ψ_odd_)[1])
            ψ_odd[bin_odd .> 0, :] = ψ_odd_
            
            return E_even, ψ_even, E_odd, ψ_odd
        else
            H = H_full(p,N)
            return eigen(H, sortby=real)
        end  
    end

    function Q_function_full_ρ(ρ, α1::ComplexF64, α2::ComplexF64,N)
        ψ_coh = kron(coherent(N, α1), coherent(N, α2))        
        return real((dot(ψ_coh, ρ * ψ_coh)) / π^2)

    end

    function Q_function_full(Ψ, α1::ComplexF64, α2::ComplexF64,N)
        ψ_coh = kron(coherent(N, α1), coherent(N, α2))
        return abs2(dot(ψ_coh, Ψ)) / π^2
    end


    function Convergency_test(p, N1, N2, plot_fig = false ,save_fig = false)
        if N1 < N2
            println("N1 must be larger than N2")
            return NaN
        end
        E2, ψ2 = Coupled_kerr_equiv(p, N2)
        E, ψ = Coupled_kerr_equiv(p, N1)
        
        se = abs.(E[1:length(E2)] - E2)
        n_se = (findall(x -> x > 1e-3, se))[1] - 1
        E_conv = n_se
        
        PR2 = vec( 1 ./ sum(abs.(ψ2).^4, dims=1) )
        PR = vec( 1 ./ sum(abs.(ψ).^4, dims=1) )
        se = abs.(PR[1:length(PR2)] - PR2)
        n_se = (findall(x -> x > 1e-3, se))[1] - 1 
        
        state_conv = n_se

        if plot_fig
            fig = figure(figsize=(8,4), layout= "constrained")
            gs = fig.add_gridspec(1,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            plot(1:length(E_S_even2), (E_S_even[1:length(E_S_even2)] - E_S_even2), ".",label="S Even")
            plot(1:length(E_S_odd2), (E_S_odd[1:length(E_S_odd2)] - E_S_odd2),".", label="S Odd")
            plot(1:length(E_A_even2), (E_A_even[1:length(E_A_even2)] - E_A_even2),".", label="S Even")
            plot(1:length(E_A_odd2), (E_A_odd[1:length(E_A_odd2)] - E_A_odd2), ".",label="AS Odd")
            plot(1:length(E_S_even2), range(-1e-3,-1e-3,length=length(E_S_even2)), label=L"10^{-3}")
            xlabel("state")
            ylabel(L"E_{%$(N1)} - E_{%$(N2)}")
            legend(fontsize=12)
            title("#levels = $(sum(E_conv))",fontsize=12)
        
            ax = fig.add_subplot(element(0,1))
            plot(1:length(PR_S_even2), (PR_S_even[1:length(PR_S_even2)] - PR_S_even2), ".",label="S Even")
            plot(1:length(PR_S_odd2), (PR_S_odd[1:length(PR_S_odd2)] - PR_S_odd2),".", label="S Odd")
            plot(1:length(PR_A_even2), (PR_A_even[1:length(PR_A_even2)] - PR_A_even2),".", label="S Even")
            plot(1:length(PR_A_odd2), (PR_A_odd[1:length(PR_A_odd2)] - PR_A_odd2), ".",label="AS Odd")
            plot(1:length(PR_S_even2), range(-1e-3,-1e-3,length=length(PR_S_even2)))
            xlabel("state")
            ylabel(L"PR_{100} - PR_{30}")
            #ylim(-1e-2,1e-2)

            title("#levels = $(sum(state_conv))",fontsize=12)
            fig.suptitle(L"Δ"*"= $(p[1]), "*L"ξ_{2}"*"= $(p[3]), γ = $(p[5]),",fontsize=14)
        end

        if save_fig
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes/Quantum/Convergency/Convergency_p_$(p)_N_$(N2).png")
        end
        return E_conv, state_conv
    end

    function labels_states_KPOs(n_states)
        labels_s = [L"C^{+}", L"C^{-}"]
        for i in 1:Int(n_states/2- 1)
            push!(labels_s,L"ψ_{e,%$i}^{+}")
            push!(labels_s,L"ψ_{e,%$i}^{-}")
        end
        labels_s
        labels_states= []
        for i in 1:Int(n_states)
            for j in 1:Int(n_states)
                push!(labels_states, "|"*labels_s[i]*labels_s[j]*"⟩")
            end
        end
        return labels_states
    end

    function H_class(u,parameters)
        q1,p1,q2,p2 = u
        δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ = parameters
        @. return -δ1*(q1^2 + p1^2)/2 + K1*((q1^2 + p1^2)^2)/4 - sqrt(2)*ξ11*q1 - ξ21*(q1^2 - p1^2) -
                    δ2*(q2^2 + p2^2)/2 + K2*((q2^2 + p2^2)^2)/4 - sqrt(2)*ξ12*q2 - ξ22*(q2^2 - p2^2) - 
                    γ*(q1*q2 + p1*p2)
    end
 
    function EqM_2(u, parameters)
        @inbounds begin
            du = zeros(4)
            q1, p1, q2, p2 = u
            δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ = parameters

            # Precompute reusable terms
            q1_sq, p1_sq, q2_sq, p2_sq = q1^2, p1^2, q2^2, p2^2
            sum1 = K1 * (p1_sq + q1_sq)
            sum2 = K2 * (p2_sq + q2_sq)

            # Equations of motion
            du[1] = (-δ1 + 2 * ξ21 + sum1) * p1 - γ * p2 
            du[2] = (δ1 + 2 * ξ21 - sum1) * q1 + γ * q2 + sqrt(2)*ξ11 
            du[3] = (-δ2 + 2 * ξ22 + sum2) * p2 - γ * p1 
            du[4] = (δ2 + 2 * ξ22 - sum2) * q2 + γ * q1 + sqrt(2)*ξ12 
        end
        return du
    end
    
    function classify_fixed_point(λ, tol = 1e-10)
           real_parts = real.(λ)
            imag_parts = imag.(λ)
            #println("Real parts: ", real_parts)
            #println("Imaginary parts: ", imag_parts)

            #number of positive real λ 
            n_pos = count(x -> x > tol, real_parts)
            #number of nevative real λ
            n_neg = count(x -> x < -tol, real_parts)
            # if it has imaginary λ
            has_imag = any(abs.(imag_parts) .> tol)
        
            if n_pos > 0 && n_neg > 0 #Saddle real of + and - 
                if has_imag
                    return "Saddle-focus"
                else
                    return "Saddle"
                end
            elseif all(x -> x < -tol, real_parts) #Stable node all directions toward the fixed point
                return has_imag ? "Stable focus (spiral sink)" : "Stable node"
            elseif all(x -> x > tol, real_parts) #Unstable node all directions away the fixed point
                return has_imag ? "Unstable focus (spiral source)" : "Unstable node"
            elseif all(abs.(real_parts) .< tol) && has_imag
                return "Center"
            else
                return "Unclassified"
            end
 
    end

    function crit_energies(parameters)
        # Root-finding function
        function find_roots_EqM(parameters, initial_guesses)
            unique_roots = []
            for guess in initial_guesses
                f!(F, u) = (F .= EqM_2(u, parameters))
                result = nlsolve(f!, guess)
                root = result.zero
                
                # Check uniqueness
                if !any(x -> norm(x - root) < 1e-2, unique_roots)
                    push!(unique_roots, root)
                end
            end
            return unique_roots
        end
        
        # Generate multiple initial guesses with Float64 values
        initial_guesses = [[Float64(x), Float64(y), Float64(z), Float64(w)] for x in -4:4, y in -4:4, z in -4:4, w in -4:4];
        initial_guesses = reshape(initial_guesses, :)


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

    function EqM!(du, u, parameters, t)
        @inbounds begin
            q1, p1, q2, p2 = u
            δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ = parameters

            # Precompute reusable terms
            q1_sq, p1_sq, q2_sq, p2_sq = q1^2, p1^2, q2^2, p2^2
            sum1 = K1 * (p1_sq + q1_sq)
            sum2 = K2 * (p2_sq + q2_sq)

            # Equations of motion
            du[1] = (-δ1 + 2 * ξ21 + sum1) * p1 - γ * p2 
            du[2] = (δ1 + 2 * ξ21 - sum1) * q1 + γ * q2 + sqrt(2)*ξ11 
            du[3] = (-δ2 + 2 * ξ22 + sum2) * p2 - γ * p1 
            du[4] = (δ2 + 2 * ξ22 - sum2) * q2 + γ * q1 + sqrt(2)*ξ12 
        end
        nothing
    end

    function Jacobian_qp(u::Vector{Float64},p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64})
        @inbounds begin
        q1, p1, q2, p2 = u
        δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ = p
        
        # Precompute reusable terms
        p1_sq, q1_sq = p1^2, q1^2
        p2_sq, q2_sq = p2^2, q2^2

        matrix = zeros(4,4)
        matrix[1,1] = 2*K1*q1*p1 ; matrix[1,2] = -δ1 + 2*ξ21 + 3*K1*p1_sq + K1*q1_sq; matrix[1,3]=0.; matrix[1,4] = -γ
        matrix[2,1] = δ1 + 2*ξ21 - K1*p1_sq - 3*K1*q1_sq; matrix[2,2] = -2*K1*p1*q1 ; matrix[2,3]=γ; matrix[2,4] = 0.
        matrix[3,1] = 0.; matrix[3,2] = -γ; matrix[3,3]=2*K2*p2*q2 ; matrix[3,4] = -δ2 + 2*ξ22 + K2*q2_sq + 3*K2*p2_sq
        matrix[4,1] = γ; matrix[4,2] = 0.; matrix[4,3]= δ2 + 2*ξ22 - 3*K2*q2_sq - K2*p2_sq; matrix[4,4] = -2*K2*p2*q2

        end
        return matrix
    end

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


    
    function Wehrl_entropy(ψ, q1vals,  p1vals, q2vals, p2vals)
        N_Q = length(q1vals)
        Q = zeros(N_Q^4)
        N = Int(sqrt(length(ψ)))
        Δq1, Δp1, Δq2, Δp2 = step(q1vals), step(p1vals), step(q2vals), step(p2vals)
        count=1
        for q1 in q1vals, p1 in q2vals, q2 in q2vals, p2 in q2vals
            α1 = (1/sqrt(2))*(q1 + p1*1im)
            α2 = (1/sqrt(2))*(q2 + p2*1im)
            Q[count] = Q_function_full(Qobj(ψ, dims=(N,N)), α1, α2, N)
            count +=1
        end
        Q /= sum(Q) #normalization
        Q = Q[Q .> 1e-14];
        return -sum((Q .* log.(Q)))*Δq1*Δp1*Δq2*Δp2
    end
    

    
end