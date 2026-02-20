module Classical_open_KPOs 
    using LinearAlgebra
    using JLD, NLsolve
    using DifferentialEquations
    using Random, Distributions, Dates,Roots, Polynomials

    export EqM!, Jacobian_qp, GS, G_evolution!, Lyapunov_max, p1_sq, generate_initial_conditions, 
            generate_initial_conditions_p2_0, crit_energies, H_class, closed_crit_energies,
            Poincare_Lyapunov, Poincare, classify_fixed_point, EqM_2

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
            δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ, κ1, κ2 = parameters

            # Precompute reusable terms
            q1_sq, p1_sq, q2_sq, p2_sq = q1^2, p1^2, q2^2, p2^2
            sum1 = K1 * (p1_sq + q1_sq)
            sum2 = K2 * (p2_sq + q2_sq)

            # Equations of motion
            du[1] = (-δ1 + 2 * ξ21 + sum1) * p1 - γ * p2 - κ1*q1/2
            du[2] = (δ1 + 2 * ξ21 - sum1) * q1 + γ * q2 + sqrt(2)*ξ11 - κ1*p1/2
            du[3] = (-δ2 + 2 * ξ22 + sum2) * p2 - γ * p1 - κ2*q2/2
            du[4] = (δ2 + 2 * ξ22 - sum2) * q2 + γ * q1 + sqrt(2)*ξ12 - κ2*p2/2
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

            real_parts[abs.(imag_parts) .> tol]
        
            if n_pos > 0 && n_neg > 0 #Saddle real of + and - 
                if has_imag
                    if any(abs.(real_parts[abs.(imag_parts) .> tol]) .> tol)
                        return "Saddle-spiral"
                    else
                        return "Saddle-center"
                    end
                else
                    return "Saddle"
                end
            elseif all(x -> x < -tol, real_parts) #Stable node all directions toward the fixed point
                return has_imag ? "Spiral sink" : "Stable node"
            elseif all(x -> x > tol, real_parts) #Unstable node all directions away the fixed point
                return has_imag ? "Spiral source" : "Unstable node"
            elseif all(abs.(real_parts) .< tol) && has_imag
                return "Center"
            else
                return "Unclassified"
            end
 
    end
    
    function crit_energies(parameters,xx_range)
        # Root-finding function
        function find_roots_EqM(parameters, initial_guesses)
            unique_roots = []
            for guess in initial_guesses
                f!(F, u) = (F .= EqM_2(u, parameters))
                result = nlsolve(f!, guess, ftol=1e-6, xtol=1e-6)
                root = result.zero
                
                # Check uniqueness
                if !any(x -> norm(x - root) < 1e-2, unique_roots) && norm(EqM_2(root, parameters)) < 1e-6
                    push!(unique_roots, root)
                end
            end
            return unique_roots
        end
        
        # Generate multiple initial guesses with Float64 values
        initial_guesses = [[Float64(x), Float64(y), Float64(z), Float64(w)] for x in -xx_range:xx_range, y in -xx_range:xx_range, z in -xx_range:xx_range, w in -xx_range:xx_range];
        initial_guesses = reshape(initial_guesses, :)


        roots_ = find_roots_EqM(parameters, initial_guesses)
        #println("Unique Roots of EqM = 0: ", roots)
        R = length(roots_)
        λs =  zeros(Complex,R, 4)
        s_λs = []
        for i in 1:R
            λ, v = eigen(Jacobian_qp(roots_[i], parameters))
            λs[i,:] = λ
            push!(s_λs, classify_fixed_point(λ)) 
        end

        return Vector{Vector{Float64}}(roots_), λs, Vector{String}(s_λs)

    end

    function closed_crit_energies(parameters, xx_range)
        # Root-finding function
        function find_roots_EqM(parameters, initial_guesses)
            unique_roots = []
            for guess in initial_guesses
                f!(F, u) = (F .= EqM_2(u, parameters))
                result = nlsolve(f!, guess, ftol=1e-6, xtol=1e-6)
                root = result.zero
                
                # Check uniqueness
                if !any(x -> norm(x - root) < 1e-2, unique_roots)
                    push!(unique_roots, root)
                end
            end
            return unique_roots
        end
        
        # Generate multiple initial guesses with Float64 values
        initial_guesses = [[Float64(x), Float64(y), Float64(z), Float64(w)] for x in -xx_range:xx_range, y in -xx_range:xx_range, z in -xx_range:xx_range, w in -xx_range:xx_range];
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

        return Vector{Vector{Float64}}(roots_[sorted_indices]), E_cl[sorted_indices], λs[sorted_indices,:], Vector{String}(s_λs[sorted_indices])
    end


    function EqM!(du, u, parameters, t)
        @inbounds begin
            q1, p1, q2, p2 = u
            δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ, κ1, κ2 = parameters

            # Precompute reusable terms
            q1_sq, p1_sq, q2_sq, p2_sq = q1^2, p1^2, q2^2, p2^2
            sum1 = K1 * (p1_sq + q1_sq)
            sum2 = K2 * (p2_sq + q2_sq)

            # Equations of motion
            du[1] = (-δ1 + 2 * ξ21 + sum1) * p1 - γ * p2 - κ1*q1/2
            du[2] = (δ1 + 2 * ξ21 - sum1) * q1 + γ * q2 + sqrt(2)*ξ11 - κ1*p1/2
            du[3] = (-δ2 + 2 * ξ22 + sum2) * p2 - γ * p1 - κ2*q2/2
            du[4] = (δ2 + 2 * ξ22 - sum2) * q2 + γ * q1 + sqrt(2)*ξ12 - κ2*p2/2
        end
        nothing
    end

    function Jacobian_qp(u::Vector{Float64},p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64})
        @inbounds begin
        q1, p1, q2, p2 = u
        δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ, κ1, κ2 = p
        
        # Precompute reusable terms
        p1_sq, q1_sq = p1^2, q1^2
        p2_sq, q2_sq = p2^2, q2^2

        matrix = zeros(4,4)
        matrix[1,1] = 2*K1*q1*p1 - κ1/2; matrix[1,2] = -δ1 + 2*ξ21 + 3*K1*p1_sq + K1*q1_sq; matrix[1,3]=0.; matrix[1,4] = -γ
        matrix[2,1] = δ1 + 2*ξ21 - K1*p1_sq - 3*K1*q1_sq; matrix[2,2] = -2*K1*p1*q1 - κ1/2; matrix[2,3]=γ; matrix[2,4] = 0.
        matrix[3,1] = 0.; matrix[3,2] = -γ; matrix[3,3]=2*K2*p2*q2 - κ2/2; matrix[3,4] = -δ2 + 2*ξ22 + K2*q2_sq + 3*K2*p2_sq
        matrix[4,1] = γ; matrix[4,2] = 0.; matrix[4,3]= δ2 + 2*ξ22 - 3*K2*q2_sq - K2*p2_sq; matrix[4,4] = -2*K2*p2*q2 - κ2/2

        end
        return matrix
    end

    function GS(A)
        M, N = size(A)
        Q = zeros(M, N)
        R = zeros(N, N)

        @inbounds begin
            for j in 1:N
                v = A[:, j]
                for k in 1:j-1
                    R[k, j] = dot(Q[:, k], v)
                    @simd for i in 1:M
                        v[i] -= R[k, j] * Q[i, k]
                    end
                end
                norm_v = norm(v)
                if norm_v > 1e-12
                    R[j, j] = norm_v
                    @simd for i in 1:M
                        Q[i, j] = v[i] / norm_v
                    end
                else
                    R[j, j] = 0.0
                    @simd for i in 1:M
                        Q[i, j] = 0.0
                    end
                end
            end
        end
        return Q, R
    end

    #function vector to do a evolution like the 
    function G_evolution!(dG,G, J,t)
        @inbounds begin
        mul!(dG, J, G)  # In-place multiplication
        end
        nothing
    end

    function Lyapunov_max(u_i::Vector{Float64}, p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}, N::Int64, Δt::Float64,err::Float64)
        @inbounds begin
            prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, p)
            # Solve the problem
            sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=N*1000)
            traj = sol.u
            
            #Defining the matrix for ICs vectors
            G = Matrix{Float64}(I, 4, 4)
            λ = zeros(4)
            λ1 = zeros(4)
            λ_t = zeros(N)

            
            for i in 1:Int(N)
                x = traj[i]
                
                J = Jacobian_qp(x,p)
                #Evolution of G from J 

                prob = ODEProblem(G_evolution!, G, (0.0, Δt), J)
                
                sol = solve(prob,Tsit5(),save_everystep = false,abstol=err, reltol=err)
                G = sol.u[end]
                
                G,R = GS(G)
                if i*Δt > 100
                    λ += log.((diag(R)))
                end
                λ1 += log.((diag(R)))
                λ_t[i] = maximum(λ1)/(i*Δt)
                
            end
        end
        #println(time()-t)
        T = (N - 100/Δt)Δt
        λ = λ/(T)

        λ_max = maximum(λ)
        return λ_max, λ_t
    end

        
    function poincare_sos(Δt, N, parameters, u_i, poin_sos=4)
        #Interpolation for the zeros 
        function linear_interp(x,x1,y1,x2,y2)
            return ((x-x1)/(x2-x1))*y2 + ((x2-x)/(x2-x1))*y1
        end  
        function t_y0(t1,t2,y1,y2)
            return (y2*t1 - y1*t2)/(y2-y1)
        end

        prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, parameters)
        # Solve the problem
        sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
        traj = sol.u
        λ_max  = Lyapunov_max_Poincare(traj, parameters,N, Δt, 1e-3)
        #plot((1:N)*Δt,λs, ".", label="Δt = $(Δt), λ = $(round(λ_max,digits=3))")
        #Component for the poincaré plot
        comp_sol = sol[poin_sos,1,:]
        poi_all =  comp_sol[2:end-1] .* comp_sol[3:end] .< 0
        
        poi_pos = comp_sol[2:end-1] .> 0 # just positive p2  
        poi_pos = poi_pos .& poi_all 
        ##poi_pos = poi_all 

        #In between times of zeros
        t_ini = (0:Δt:N*Δt)[2:end-1][poi_pos] #times + for zeros
        t_end = (0:Δt:N*Δt)[3:end][poi_pos] # times - for zeros
        comp_ini = comp_sol[2:end-1][poi_pos] #p2 positive zero
        comp_end = comp_sol[3:end][poi_pos] #p2 negatie zero
        t_values=  @. t_y0(t_ini, t_end, comp_ini, comp_end) #Interpolation

        dim_sols = size(t_values)[1]
        solutions = zeros(dim_sols,3)

        index = 1
        for component in 1:4
            if component == poin_sos
                continue
            end
            solutions[:,index] = @. linear_interp(t_values, t_ini,sol[component,1,:][2:end-1][poi_pos], t_end, sol[component,1,:][3:end][poi_pos])
            index+=1  
        end
        λ_maxs = λ_max*ones(dim_sols)
        return solutions, λ_maxs
    end

    function Lyapunov_max_Poincare(traj, p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}, N::Int64, Δt::Float64,err::Float64)
        @inbounds begin
            #Defining the matrix for ICs vectors
            G = Matrix{Float64}(I, 4, 4)
            λ = zeros(4)
            
            for i in 1:Int(N)
                x = traj[i]
                
                J = Jacobian_qp(x,p)
                #Evolution of G from J 

                prob = ODEProblem(G_evolution!, G, (0.0, Δt), J)
                
                sol = solve(prob,Tsit5(),save_everystep = false,abstol=err, reltol=err)
                G = sol.u[end]
                
                G,R = GS(G)
                if i*Δt > 100
                    λ += log.((diag(R)))
                end
                
            end
        end
        #println(time()-t)
        T = (N - 100/Δt)Δt
        λ = λ/(T)

        λ_max = maximum(λ)
        return λ_max
    end

    function Poincare_Lyapunov(E,parameters,N, Δt,q1_l,q1_r, min_ics = 100)
        #Well separated initial positions
        ICs, ωs = Weighted_initial_conditions(E, parameters, q1_l,q1_r, min_ics = min_ics)
        n_points = size(ICs)[1]

        final_points = 0. #just to have a variable after the loop 
        λ_maxs = 0

        g=0 #Flag for final_points

        for i in 1:n_points
            g += 1
            points, λ_max = poincare_sos(Δt, Int(N), parameters, ICs[i])
            if g == 1
                final_points = copy(points)
                λ_maxs = copy(λ_max)
            else
                final_points = vcat(final_points, points)
                λ_maxs = vcat(λ_maxs, λ_max)
            end
        end
        return final_points,λ_maxs
    end

        
    function poincare_wo_λ(Δt, N, parameters, u_i, poin_sos=4)
        #Interpolation for the zeros 
        function linear_interp(x,x1,y1,x2,y2)
            return ((x-x1)/(x2-x1))*y2 + ((x2-x)/(x2-x1))*y1
        end  
        function t_y0(t1,t2,y1,y2)
            return (y2*t1 - y1*t2)/(y2-y1)
        end

        prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, parameters)
        # Solve the problem
        sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
        
        #Component for the poincaré plot
        comp_sol = sol[poin_sos,1,:]
        poi_all =  comp_sol[2:end-1] .* comp_sol[3:end] .< 0
        
        poi_pos = comp_sol[2:end-1] .> 0 # just positive p2  
        poi_pos = poi_pos .& poi_all 
        ##poi_pos = poi_all 

        #In between times of zeros
        t_ini = (0:Δt:N*Δt)[2:end-1][poi_pos] #times + for zeros
        t_end = (0:Δt:N*Δt)[3:end][poi_pos] # times - for zeros
        comp_ini = comp_sol[2:end-1][poi_pos] #p2 positive zero
        comp_end = comp_sol[3:end][poi_pos] #p2 negatie zero
        t_values=  @. t_y0(t_ini, t_end, comp_ini, comp_end) #Interpolation

        dim_sols = size(t_values)[1]
        solutions = zeros(dim_sols,3)

        index = 1
        for component in 1:4
            if component == poin_sos
                continue
            end
            solutions[:,index] = @. linear_interp(t_values, t_ini,sol[component,1,:][2:end-1][poi_pos], t_end, sol[component,1,:][3:end][poi_pos])
            index+=1  
        end
        return solutions
    end


    function Poincare(E,parameters,N, Δt,q1_l,q1_r, min_ics = 100)
        #Well separated initial positions
        ICs, ωs = Weighted_initial_conditions(E, parameters, q1_l,q1_r, min_ics = min_ics)
        n_points = size(ICs)[1]

        final_points = 0. #just to have a variable after the loop 

        g=0 #Flag for final_points

        for i in 1:n_points
            g += 1
            points = poincare_wo_λ(Δt, Int(N), parameters, ICs[i])
            if g == 1
                final_points = copy(points)
            else
                final_points = vcat(final_points, points)
            end
        end
        return final_points
    end

    
    function p2_polynomial_coeffs(q1, p1, q2, E, params)
        """
            returns coefficients (a0, a1, a2, a3, a4) for polynomial a0 + a1 p2 + a2 p2^2 + a3 p2^3 + a4 p2^4 = 0
        """
    
        δ1, K1, ξ11, ξ21, δ2, K2, ξ12, ξ22, γ, κ1, κ2 = params

        q1sq = q1^2
        p1sq = p1^2
        q2sq = q2^2

        a4 = K2 / 4.0                        
        a3 = 0.0                             
        a2 = -δ2/2.0 + (K2/2.0)*q2sq + ξ22  
        a1 = -γ * p1                         
        # constant term: H with p2=0 minus E
        a0 = -δ1*(q1sq + p1sq)/2 + K1*((q1sq + p1sq)^2)/4 - sqrt(2)*ξ11*q1 - ξ21*(q1sq - p1sq) -
            δ2*(q2sq)/2 + K2*(q2sq^2)/4 - sqrt(2)*ξ12*q2 - ξ22*(q2sq) - γ*(q1*q2) - E

        return (a0, a1, a2, a3, a4)
    end
    
    function real_p2_roots(q1, p1, q2, E, params; imag_tol=1e-6)
        """
            returns real roots of the polynomial a0 + a1 p2 + a2 p2^2 + a3 p2^3 + a4 p2^4 = 0
        """
        a0,a1,a2,a3,a4 = p2_polynomial_coeffs(q1,p1,q2,E,params)
        coeffs = [a0, a1, a2, a3, a4]
        poly = Polynomial(coeffs)   # constant + ... + a4 x^4
        rts = Polynomials.roots(poly)
        realr = Float64[]
        for z in rts
            if abs(imag(z)) <= imag_tol
                push!(realr, real(z))
            end
        end
        return realr
    end
end