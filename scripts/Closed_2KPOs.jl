    using LinearAlgebra
    using JLD
    using PyPlot
    using PyCall
    using DifferentialEquations
    #using Random, Distributions, Dates,Roots
    using Random, Distributions, Dates, Polynomials, StatsBase 
    #using BenchmarkTools, NLsolve
    using DynamicalSystems,DelimitedFiles
    pygui(true)
    include("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Closed_2KPOs_functions.jl")
    using .Lyapunov_Energies

    # Delmar Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
    # Delmar Energies cE = [-48.2526,  -23.8816, -20.8769,  -10.9252,  -3.82856,  1.54256]
    # ---------------------- LYapunov x Energy --------------------------------------------------------------------------------
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
        
        roots, cE, λs, s_λ = crit_energies(p,7);

        n_E = 10
        Es = range(cE[1],cE[end], length=n_E)
        #ps = [zeros(1,1) for i in 1:n_E]
        λs = [Float64[] for i in 1:n_E]

        #Random initial positions
        q_l = -5.
        q_r = 5.
        Random.seed!(122)
        d = Uniform(q_l, q_r)



        #Variables for Lyapunov 
        N = Int(2e4)
        Δt = 1e-2 #smallest time inteval for the Lapunov calculation
        t = N*Δt

        #Multiple Energies
        #=Memory test
            j=2
            mem1 = @allocated generate_initial_conditions(Es[j], parameters) #1.11Gb
            mem2 = @allocated generate_initial_conditions_p2_0(Es[j], parameters) #7.15 Mb
            mem3 = @allocated  Lyapunov_max(Ics[1], parameters, N, Δt, 1e-3) #1Gb
            println("$((mem1 + mem2 + mem3)/ 1e9) Gb")
        =#
        t= time()
        for j in n_E:n_E
            ICs, w = Weighted_initial_conditions(Es[j], parameters,-5.,5.)
            lim = length(ICs)
            ICs = vcat(ICs,generate_initial_conditions(Es[j], parameters, min_ics=100 - lim))
            
            n_ICs= size(ICs)[1]
            for i in 1:n_ICs
                λ_max  = Lyapunov_max(ICs[i], parameters, N, Δt, 1e-3)
                push!(λs[j], λ_max)
            end 
        end
        time()-t

        λs = Float64[]
        t= time()
        E = 0
            ICs, w = Weighted_initial_conditions(E, parameters,-5.,5.)
            lim = length(ICs)
            ICs = vcat(ICs,generate_initial_conditions(Es[j], parameters, min_ics=100 - lim))
            
            n_ICs= size(ICs)[1]
            for i in 1:n_ICs
                λ_max  = Lyapunov_max(ICs[i], parameters, N, Δt, 1e-3)
                push!(λs, λ_max)
            end 
        time()-t

        
        x = Float64[]
        y = Float64[]
        μ_E, σ_E = zeros(n_E), zeros(n_E)
        fig = figure(figsize=(5, 4), layout="constrained")
        for j in 1:n_E
            x = vcat(x,range(Es[j],Es[j], length=length(λs[j])))
            y = vcat(y,λs[j])
            μ_E[j], σ_E[j]= mean(λs[j]), var(λs[j])
        
            title("δ = $(parameters[1]), " *L"ξ"*" = $(parameters[3]), γ = $(parameters[6])")
            plot(range(Es[j],Es[j], length=length(λs[j])), λs[j], ".", color="red")
            #ylim(-6,6)
            #xlim(-6,6)
            xlabel(L"E",fontsize=12)
            ylabel(L"λ",fontsize=12)
        end
        

        #plot(Es, range(1,1, length=100))
        #Form cluster results 

            using DelimitedFiles
            ns_job = [1, 25, 50, 75, 100]
            #p = (0.0, 1.0, 10.0, 5.0, 0.0, 1.0, 10.0, 5.0, 1.0)
            p = (10.0, 1.0, 0.0, 5.0, 10.0, 1.0, 0.0, 5.0, 1.0)
            p
            job=1;
            data = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat");
            data2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat")[ns_job[job]:ns_job[job+1], :];
            
            for job in 2:4
                data = vcat(data, readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat"))
                data2 = vcat(data2, readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat")[ns_job[job]:ns_job[job+1], :])
            end
        ###
        
        

        fig = figure(figsize=(10,5), layout="constrained");
        gs = fig.add_gridspec(1,2);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        ax = fig.add_subplot(element(0,0));
        plot(data[:,1], data[:,2], ".", color="red", markersize=5);
        
        roots, E_cl, λs, s_λs = crit_energies(p);
        E_cl'
        min_S = minimum(data[:,2]);
        max_S = maximum(data[:,2]);
        for i in [1, 3, 5, 10,11]
            plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3))")
        end
        plot(data2[:,1], data2[:,2], "-", color="blue", markersize=3, label = "Mean ($(round(mean(data2[:,2][2:end]), digits=3)))");
        #ylim(-6,6)
        #xlim(-6,6)
        xlabel(L"E",fontsize=12);
        ylabel(L"λ",fontsize=12);
        
        ax = fig.add_subplot(element(0,1));
        plot(data[:,1], data[:,2], ".", color="red", markersize=5);
        
        for i in [1, 3, 5, 10,11]
            plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3))")
        end
        plot(data2[:,1], data2[:,2], "-", color="blue", markersize=3, label = "Mean (Total = $(round(mean(data2[:,2][2:end]), digits=3)))");
        legend(loc="upper left",fontsize=10, shadow=true);
        #ylim(-6,6)
        xlim(E_cl[1], E_cl[5]);
        xlabel(L"E",fontsize=12);
        ylabel(L"λ",fontsize=12);
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");

        
        
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energies_$(p).png")
        
        
        λs_10 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_10.jld")[" λs"];
        λmean_10 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_10.jld")["λ_mean2"];
        λWmean_10 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Weighted_Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_10.jld")["λ_mean"];
        λs_100 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_100.jld")[" λs"];
        λmean_100 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_100.jld")["λ_mean2"];
        λWmean_100 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Weighted_Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_100.jld")["λ_mean"];
        λs_300 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_300.jld")[" λs"];
        λmean_300 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_300.jld")["λ_mean2"];
        λWmean_300 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Weighted_Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_300.jld")["λ_mean"];
        λs_500 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_500.jld")[" λs"];
        λmean_500 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_500.jld")["λ_mean2"];
        λWmean_500 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Weighted_Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_500.jld")["λ_mean"];
        λs_1000 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_1000.jld")[" λs"];
        λmean_1000 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_1000.jld")["λ_mean2"];
        λWmean_1000 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Weighted_Mean_Lyapunov_Energies_(0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0)_E_-60.5_-0.0_1_25_ICs_1000.jld")["λ_mean"];
        
        
        fig = figure(figsize=(7,8), layout="constrained");
        plot(1:1000, λs_1000[2], "s", color="purple",label= L"ICs = 1000, λ_{mean} = %$(round(λmean_1000[2], digits=3)), λ_{Wmean} = %$(round(λWmean_1000[2], digits=3))")
        plot(1:500, λs_500[2], "o", color="yellow",label= L"ICs = 500, λ_{mean} = %$(round(λmean_500[2], digits=3)), λ_{Wmean} = %$(round(λWmean_500[2], digits=3))")
        plot(1:300, λs_300[2], "*", color="blue",label= L"ICs = 300, λ_{mean} = %$(round(λmean_300[2], digits=3)), λ_{Wmean} = %$(round(λWmean_300[2], digits=3))")
        plot(1:100, λs_100[2], ">", color="red",label= L"ICs = 100, λ_{mean} = %$(round(λmean_100[2], digits=3)), λ_{Wmean} = %$(round(λWmean_100[2], digits=3))")
        plot(1:10, λs_10[2], "v", color="green",label= L"ICs = 10, λ_{mean} = %$(round(λmean_10[2], digits=3)), λ_{Wmean} = %$(round(λWmean_10[2], digits=3))")
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
        legend(loc="upper right", fontsize=10, shadow=true)
        ylim(0.0, 1.5)
        xlabel("IC index",fontsize=12)
        ylabel(L"λ",fontsize=12)
        title("E = $(0)", fontsize=12)
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICS_test.png")

        

        fig = figure(figsize=(7,8), layout="constrained");
        gs = fig.add_gridspec(2,1)
        element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
        ax = fig.add_subplot(element(0,0))
        #plot(1:200, data3[:,2][1:200], "o", color="blue",label= L"p_{2} = 0")
        #plot(201:400, data3[:,2][201:400], "*", color="cyan",label= "Roots method")
        plot(1:973, data3[:,2][1:973], "o", color="blue",label= L"p_{2} = 0")
        plot(201:400, data3[:,2][201:400], "*", color="cyan",label= "Roots method")
        legend(loc="upper right", fontsize=10, shadow=true)
        xlabel("IC index",fontsize=12)
        ylabel(L"λ",fontsize=12)
        title("E = $(-45)", fontsize=12)
        
        ax = fig.add_subplot(element(1,0))
        plot(1:200, data3[:,2][401:600], "o", color="blue",label= L"p_{2} = 0")
        plot(201:400, data3[:,2][601:800], "*", color="cyan",label= "Roots method")
        xlabel("IC index",fontsize=12)
        ylabel(L"λ",fontsize=12)
        title("E = $(-25)", fontsize=12)
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");
        
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energies_$(p)_2000.png")
    ###

    #----------------------------------Poincare Lyapunov - Energy based -----------------------------------------------
        #Single Energie
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
            E = -32.;
            N = Int(2e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            N*Δt
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.
            final_points, λ_maxs = Poincare_Lyapunov(E,p,N, Δt,q1_l,q1_r, q2_l,q2_r)
            
            
            
            fig = figure(figsize=(5, 10), layout="constrained")
            gs = fig.add_gridspec(2,1)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            scat_plot = ax.scatter(final_points[:,1], final_points[:,3], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"q_2",fontsize=12)

            ax = fig.add_subplot(element(1,0))
            scat_plot = ax.scatter(final_points[:,1], final_points[:,2], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"p_1",fontsize=12)




        ###

        #Multiple Energies
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213    
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 1., 1., 0., 5., 1., 1., 0., 5., 1.;
            n_E = 5
            roots_, cE, λs, s_λ = crit_energies(p);
            Es = range(cE[1]+1.,cE[4], length=n_E)
            #Es = range(cE[5]+1.,cE[9], length=n_E)
            collect(Es)
            ps = [zeros(1,1) for i in 1:n_E]
            λs = [zeros(1) for i in 1:n_E]

            N, Δt = Int(2e4), 1e-2; #smallest time inteval for the Lapunov calculation
            N*Δt
            #limits for Poincare plot
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.

            t= time()
            for i in 1:n_E
                final_points, λ_maxs = Poincare_Lyapunov(Es[i], p, N, Δt,q1_l,q1_r, q2_l,q2_r)
                ps[i] = final_points
                λs[i] = λ_maxs
            end
            println("Time taken: ", time()-t, " seconds")

            fig = figure(figsize=(60, 10), layout="constrained")
            gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.25])
            #gs = fig.add_gridspec(2,5)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            fig.subplots_adjust(right=0.85) 
            for i in 1:5
                ax = fig.add_subplot(element(0,i-1))
                title("E = $(Es[i])")
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,3], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-q1_l,q1_l)
                if i == 1
                    ax.set_ylabel(L"q_2",fontsize=12)
                else
                    yticks([])
                end
                xticks([])
                ax.set_xlim(-q1_l,q1_l)

                ax = fig.add_subplot(element(1,i-1))
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,2], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-q1_l,q1_l)
                ax.set_xlim(-q1_l,q1_l)
                
                if i == 1
                    ax.set_ylabel(L"p_1",fontsize=12)
                else
                    yticks([])
                end
                ax.set_xlabel(L"q_1",fontsize=12)
                
                if i == 5
                    #[x, y, width, height]
                    cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
                    cbar = plt.colorbar(scat_plot, cax = cax_)
                end
            end
            text(0.3, -0.02, "λ", fontsize=18, verticalalignment="top")

            fig.suptitle(L" Δ_{1}, K_{1}, ξ_{1,1}, ξ_{2,1}, Δ_{2}, K_{2}, ξ_{1,2}, ξ_{2,2}, γ "*"= $(p)",fontsize=14)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_$(p)_E_$(collect(Es)).png")
            close()
        #####

        #Multiple parameters 
            n_p = 3
            γs = [0, 0.3 , 1.]
            ps = [zeros(1,1) for i in 1:n_p]
            λs = [zeros(1) for i in 1:n_p]

            for i in 1:n_p
                parameters = 0.,0.,5.,5.,1.,γs[i]
                final_points, λ_maxs = Poincare_Lyapunov(-32.,parameters)
                ps[i] = final_points
                λs[i] = λ_maxs
            end

            fig = figure(figsize=(30, 10), layout="constrained")
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.])
            #gs = fig.add_gridspec(2,5)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            fig.subplots_adjust(right=0.85) 
            for i in 1:n_p
                ax = fig.add_subplot(element(0,i-1))
                #title("E = $(Es[i])")
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,3], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-6,6)
                if i == 1
                    ax.set_ylabel(L"q_2",fontsize=12)
                else
                    yticks([])
                end
                xticks([])
                ax.set_xlim(-6,6)

                ax = fig.add_subplot(element(1,i-1))
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,2], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-6,6)
                ax.set_xlim(-6,6)
                
                if i == 1
                    ax.set_ylabel(L"p_1",fontsize=12)
                else
                    yticks([])
                end
                ax.set_xlabel(L"q_1",fontsize=12)
                
                if i == 5
                    #[x, y, width, height]
                    cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
                    cbar = plt.colorbar(scat_plot, cax = cax_)
                end
            end
            
            text(1.1, -0.02, "λ", transform=ax.transAxes, fontsize=14, verticalalignment="top")

            fig.suptitle("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6])",fontsize=14)
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes/Classical_Kerr/Figures/Poincare_$(parameters)_E_$(collect(Es)).png")
            savefig("C:/Users/edson/Desktop/a.png")

        #####

        ### More...
            N = Int(2e4)
            Δt = 1e-2 #smallest time inteval for the Lapunov calculation
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
            
            roots, cE = crit_energies(parameters);
            cE = [-48.2526,  -23.8816, -20.8769,  -10.9252,  -3.82856,  1.54256]
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.
            final_points, λ_maxs = Poincare_Lyapunov2(parameters, N, Δt,q1_l,q1_r, q2_l,q2_r)
            fig = figure(figsize=(5, 10), layout="constrained")
            gs = fig.add_gridspec(2,1)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            scat_plot = ax.scatter(final_points[:,1], final_points[:,3], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"q_2",fontsize=12)

            ax = fig.add_subplot(element(1,0))
            scat_plot = ax.scatter(final_points[:,1], final_points[:,2], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"p_1",fontsize=12)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICs_Delmar.png")
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICs_$(parameters).png")
                

            N = Int(2e4)
            Δt = 1e-2 #smallest time inteval for the Lapunov calculation
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
            
            roots, cE, λs, s_λs = crit_energies(parameters);
            s_λs
            #Classification fixed point
            #1 :: center, minimum 

        ###
    ####

    #-----------------------Counting Fixed points analysis -------------------------------------------------
        N_p = 100
        ps = collect(range( 0.01,  110., length = N_p))
        θ1s = fill(NaN, N_p, 30)
        s_λs = fill("", N_p, 30)
        roots_number = zeros(N_p)
        p1p2_points = [[] for i in 1:N_p]
        #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. # For γ
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1. # For ξ2
        K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1. # For Δ

        for j in 1:N_p
            #parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            #parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ  
            parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 
            roots, cE, λs, s_λ = crit_energies(parameters);
            roots_number[j] = R = length(roots)
            θ1= zeros(R)
            #println(j)
            
            for i in 1:R
                θ1[i] = atan(roots[i][3], roots[i][1])
                if roots[i][2] > 1e-3 || roots[i][4] > 1e-3
                    push!(p1p2_points[j], roots[i])
                end
            end
            #sort(q1, by=real)'
            #sort(q11, by=real)' 
            sorted_indices = sortperm(θ1)
            roots, s_λ = roots[sorted_indices], s_λ[sorted_indices]
            
            for i in 1:R
                #println(j,i)
                θ1s[j, i] = atan(roots[i][3], roots[i][1])
                s_λs[j, i] = s_λ[i]         
            end 
        end
        p1p2_points
        p1p2_number = length.(p1p2_points)
        plot(ps, roots_number, "-o")
        xlabel(L"Δ",fontsize=12)
        ylabel("Number of FP",fontsize=12)
        title("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_numebr_Δ_$(ps[1])_$(ps[end]).png")
        
        
        plot(ps, p1p2_number, "-o")
        xlabel(L"γ",fontsize=12)
        ylabel("Number of FP out of "*L"q_{1}q{2}"*"-plane",fontsize=12)
        title("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))")
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_p1p2_γ_$(ps[1])_$(ps[end]).png")
        
    ###

    #-----------------------Fixed points analysis -------------------------------------------------
        N_p = 100
        ps = collect(range( 0.01,  13., length = N_p))
        θ1s = fill(NaN, N_p, 30)
        s_λs = fill("", N_p, 30)
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 0., 0., 0., 0., 0., 0., 0. # For γ
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1. # For ξ2
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1. # For Δ
        
        for j in 1:N_p
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, ps[j]    
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.,  1., 0., 2.8^2, 0., 10.3/10.4, 0., 2.5^2, ps[j]
            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            #parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
            #parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 

            roots, cE, λs, s_λ = crit_energies(parameters,7);
            R = length(roots)
            θ1= zeros(R)
            #println(j)
            
            for i in 1:R
                θ1[i] = atan(roots[i][3], roots[i][1])
            end
            #sort(q1, by=real)'
            #sort(q11, by=real)' 
            sorted_indices = sortperm(θ1)
            roots, s_λ = roots[sorted_indices], s_λ[sorted_indices]
            
            for i in 1:R
                #println(j,i)
                θ1s[j, i] = atan(roots[i][3], roots[i][1])
                s_λs[j, i] = s_λ[i]         
            end 
        end


        function plot_γ()
            fig = figure(figsize=(20, 10), layout="constrained")
            gs = fig.add_gridspec(2,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            i=1
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue", label= "Saddle")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red", label= "Saddle-focus")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green", label= "Minimum")
            for i in 2:9
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green")
            end
            ax.set_xlabel(L"γ",fontsize=12)
            ax.set_ylabel(L"θ",fontsize=12)
            legend(fontsize=10, shadow=true, loc = "upper right")
            
            j, k, l = 2, 42, 86
            plot(range(ps[j], ps[j], length = 10), range(-π, π, length = 10), "-", lw =2) 
            plot(range(ps[k], ps[k], length = 10), range(-π, π, length = 10), "-", lw =2 ) 
            plot(range(ps[l], ps[l], length = 10), range(-π, π, length = 10), "-", lw =2 ) 


            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            roots, E_cl, λs, s_λ = crit_energies(parameters,7);
            ax = fig.add_subplot(element(0,1))
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters,7)
            title(L"γ"*" = $(round(ps[j],digits=3))")
            
            
            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[k] 
            ax = fig.add_subplot(element(1,0))
            roots, E_cl, λs, s_λ = crit_energies(parameters,7);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7)
            title(L"γ"*" = $(round(ps[k],digits=3))")
            
            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[l] 
            ax = fig.add_subplot(element(1,1))
            roots, E_cl, λs, s_λ = crit_energies(parameters,7);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:3], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[4:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7)
            title(L"γ"*" = $(round(ps[l],digits=3))")
            
            suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(round.(parameters[1:8],digits=3))")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_equivalent_γ.png")
        end
        plot_γ()

        
        
        function plot_ξ()
            fig = figure(figsize=(20, 10), layout="constrained")
            gs = fig.add_gridspec(2,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            i=1
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue", label= "Saddle")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red", label= "Saddle-focus")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green", label= "Minimum")
            for i in 2:9
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green")
            end
            ax.set_xlabel(L"ξ_{2}",fontsize=12)
            ax.set_ylabel(L"θ",fontsize=12)
            legend(fontsize=10, shadow=true, loc = "upper right")
            
            j, k, l = 2, 10, 50
            plot(range(ps[j], ps[j], length = 10), range(-π, π, length = 10), "-", lw =2) 
            plot(range(ps[k], ps[k], length = 10), range(-π, π, length = 10), "-", lw =2 ) 
            plot(range(ps[l], ps[l], length = 10), range(-π, π, length = 10), "-", lw =2 ) 


            parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ 
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            ax = fig.add_subplot(element(0,1))
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:3], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[4:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 3)
            title(L"ξ_{2}"*" = $(round(ps[j],digits=3))")
            
            
            parameters = Δ1, K1, ξ11, ps[k], Δ2, K2, ξ12, ps[k], γ  
            ax = fig.add_subplot(element(1,0))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:4], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[5:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 4)
            title(L"ξ_{2}"*" = $(round(ps[k],digits=3))")
            
            parameters = Δ1, K1, ξ11, ps[l], Δ2, K2, ξ12, ps[l], γ 
            ax = fig.add_subplot(element(1,1))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:4], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[5:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7)
            title(L"ξ_{2}"*" = $(round(ps[l],digits=3))")
            
            suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(round.(parameters[1:8],digits=3))")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_equivalent_ξ2.png")
        end
        plot_ξ()

        function plot_Δ()
            fig = figure(figsize=(20, 10), layout="constrained")
            gs = fig.add_gridspec(2,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            i=1
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue", label= "Saddle")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red", label= "Saddle-focus")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green", label= "Minimum")
            for i in 2:9
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green")
            end
            ax.set_xlabel(L"Δ",fontsize=12)
            ax.set_ylabel(L"θ",fontsize=12)
            legend(fontsize=10, shadow=true, loc = "upper right")
            
            j, k, l = 10, 75, 80
            plot(range(ps[j], ps[j], length = 10), range(-π, π, length = 10), "-", lw =2) 
            plot(range(ps[k], ps[k], length = 10), range(-π, π, length = 10), "-", lw =2 ) 
            plot(range(ps[l], ps[l], length = 10), range(-π, π, length = 10), "-", lw =2 ) 


            parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            ax = fig.add_subplot(element(0,1))
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7)
            title(L"Δ"*" = $(round(ps[j],digits=3))")
            
            
            parameters = ps[k], K1, ξ11, ξ21, ps[k], K2, ξ12, ξ22, γ 
            ax = fig.add_subplot(element(1,0))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 10)
            title(L"Δ"*" = $(round(ps[k],digits=3))")
            
            parameters = ps[l], K1, ξ11, ξ21, ps[l], K2, ξ12, ξ22, γ 
            ax = fig.add_subplot(element(1,1))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:9], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[10:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 10)
            title(L"Δ"*" = $(round(ps[l],digits=3))")
            
            suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(round.(parameters[1:8],digits=3))")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_equivalent_Δ.png")
        end
        plot_Δ()
    ###

    #-----------------------Fixed points analysis -q1p1------------------------------------------------
    
        function plot_Δ_q1p1()
            fig = figure(figsize=(15, 10), layout="constrained")
            gs = fig.add_gridspec(2,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 15.,  1., 0., 5., 15., 1., 0., 5., 1.
            roots, cE, λs, s_λ = crit_energies(parameters);
            #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            plots_rs = []
            roots
            for i in 1:length(roots)
                if abs(roots[i][3]) < 1e-3
                    println(roots[i])
                    plot_r, = plot(roots[i][1], roots[i][2], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                    push!(plots_rs, plot_r)
                end
            end
            legend1 = legend(handles=plots_rs[1:1], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            countor_energy(parameters, 6, false)
            title(L"Δ"*" = $(round(Δ1 ,digits=3))")
            
            ax = fig.add_subplot(element(0,1))
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 20.,  1., 0., 5., 20., 1., 0., 5., 1.
            roots, cE, λs, s_λ = crit_energies(parameters);
            #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            plots_rs = []
            roots
            for i in 1:length(roots)
                if abs(roots[i][3]) < 1e-3
                    println(roots[i])
                    plot_r, = plot(roots[i][1], roots[i][2], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                    push!(plots_rs, plot_r)
                end
            end
            legend1 = legend(handles=plots_rs[1:1], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            #legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7, false)
            title(L"Δ"*" = $(round(Δ1 ,digits=3))")

            ax = fig.add_subplot(element(1,0))
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 30.,  1., 0., 5., 30., 1., 0., 5., 1.
            roots, cE, λs, s_λ = crit_energies(parameters);
            #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            plots_rs = []
            roots
            for i in 1:length(roots)
                if abs(roots[i][3]) < 1e-3
                    println(roots[i])
                    plot_r, = plot(roots[i][1], roots[i][2], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                    push!(plots_rs, plot_r)
                end
            end
            legend1 = legend(handles=plots_rs[1:3], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            #legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 9, false)
            title(L"Δ"*" = $(round(Δ1 ,digits=3))")

            ax = fig.add_subplot(element(1,1))
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 40.,  1., 0., 5., 40., 1., 0., 5., 1.
            roots, cE, λs, s_λ = crit_energies(parameters);
            #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            plots_rs = []
            roots
            for i in 1:length(roots)
                if abs(roots[i][3]) < 1e-3
                    plot_r, = plot(roots[i][1], roots[i][2], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                    push!(plots_rs, plot_r)
                end
            end
            #legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            #ax.add_artist(legend1)
            #legend2 = legend(handles=plots_rs[7:end], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)

            countor_energy(parameters, 10, false)
            title(L"Δ"*" = $(round(Δ1 ,digits=3))")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_Δ_q1p1_2.png")
        end
        plot_Δ_q1p1()
        #suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(round.(parameters[1:8],digits=3))")
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points_stability_Δ1_Delmar_2.png")
    
        #### p1p2 plot surface
            
            function countor_energy_p1p2(parameters, xx)
                    roots, E_cl, λs, s_λ = crit_energies(parameters)
                    R = length(roots)
                    θ= zeros(R)        
                    for i in 1:R
                        θ[i] = atan(roots[i][3], roots[i][1])
                    end
                    x = range(-xx, xx, length=1000);
                    y = range(-xx, xx, length=1000);

                    #Equivalent of meshgrid
                    coordinates_x = repeat(x', length(x), 1);
                    coordinates_y = repeat(y, 1, length(y));

                    q1, p1, q2, p2 = 0., coordinates_x,0., coordinates_y,0,0

                    E_Contours = H_class([q1, p1, q2, p2],parameters);

                    Emin, Emax = E_cl[1],30.
                    CS = contourf(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 51));
                    contour(coordinates_x, coordinates_y, E_Contours, [0]); # Only draw contour line for E = 0
                    
                    xlabel(L"p_1",fontsize=12)
                    ylabel(L"p_2",fontsize=12) #q1,p1

                    cbar = colorbar(CS, label="E")
            end
            
            function plot_Δ_p1p2()
                fig = figure(figsize=(15, 10), layout="constrained")
                gs = fig.add_gridspec(2,2)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

                ax = fig.add_subplot(element(0,0))
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 15.,  1., 0., 5., 15., 1., 0., 5., 1.
                roots, cE, λs, s_λ = crit_energies(parameters);
                #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                plots_rs = []
                roots
                for i in 1:length(roots)
                    if abs(roots[i][3]) < 1e-3 && abs(roots[i][1]) < 1e-3 
                        println(roots[i])
                        plot_r, = plot(roots[i][2], roots[i][4], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                        push!(plots_rs, plot_r)
                    end
                end
                legend1 = legend(handles=plots_rs[1:1], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
                ax.add_artist(legend1)
                countor_energy_p1p2(parameters, 4)
                title(L"Δ"*" = $(round(Δ1 ,digits=3))")
                
                ax = fig.add_subplot(element(0,1))
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 20.,  1., 0., 5., 20., 1., 0., 5., 1.
                roots, cE, λs, s_λ = crit_energies(parameters);
                #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                plots_rs = []
                roots
                for i in 1:length(roots)
                    if abs(roots[i][3]) < 1e-3 && abs(roots[i][1]) < 1e-3 
                        println(roots[i])
                        plot_r, = plot(roots[i][2], roots[i][4], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                        push!(plots_rs, plot_r)
                    end
                end
                legend1 = legend(handles=plots_rs[1:1], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
                ax.add_artist(legend1)
                #legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
                countor_energy_p1p2(parameters, 6)
                title(L"Δ"*" = $(round(Δ1 ,digits=3))")

                ax = fig.add_subplot(element(1,0))
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 30.,  1., 0., 5., 30., 1., 0., 5., 1.
                roots, cE, λs, s_λ = crit_energies(parameters);
                #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                plots_rs = []
                roots
                for i in 1:length(roots)
                    if abs(roots[i][3]) < 1e-3 && abs(roots[i][1]) < 1e-3 
                        println(roots[i])
                        plot_r, = plot(roots[i][2], roots[i][4], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                        push!(plots_rs, plot_r)
                    end
                end
                legend1 = legend(handles=plots_rs[1:3], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
                ax.add_artist(legend1)
                #legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
                countor_energy_p1p2(parameters, 7)
                title(L"Δ"*" = $(round(Δ1 ,digits=3))")

                ax = fig.add_subplot(element(1,1))
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 40.,  1., 0., 5., 40., 1., 0., 5., 1.
                roots, cE, λs, s_λ = crit_energies(parameters);
                #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                plots_rs = []
                roots
                for i in 1:length(roots)
                    if abs(roots[i][3]) < 1e-3 && abs(roots[i][1]) < 1e-3 
                        println(roots[i])
                        plot_r, = plot(roots[i][2], roots[i][4], marker="o", markersize=7, label="E = $(round(cE[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                        push!(plots_rs, plot_r)
                    end
                end
                legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
                ax.add_artist(legend1)
                legend2 = legend(handles=plots_rs[7:end], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)

                countor_energy_p1p2(parameters, 9)
                title(L"Δ"*" = $(round(Δ1 ,digits=3))")
                savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_Δ_p1p2_2.png")
            end
            plot_Δ_p1p2()
        ###
    ###

    #Testing N and Δt for Lyapunov Calculation 
        function Lyapunov_max2(u_i::Vector{Float64}, p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,}, N::Int64, Δt::Float64,err::Float64)
            @inbounds begin
                prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, p)
                # Solve the problem
                sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
                traj = sol.u
                
                #Defining the matrix for ICs vectors
                G = Matrix{Float64}(I, 4, 4)
                λ = zeros(4) 
                λ1 = zeros(4)
                λs = zeros(N)
                
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
                    λs[i] = maximum(λ1)/(i*Δt)
                    
                end
            end
            #println(time()-t)
            T = (N - 100/Δt)Δt
            λ = λ/(T)

            λ_max = maximum(λ)
            return λ_max, λs
        end
            #Random initial positions
        q_l = -5.
        q_r = 5.
        Random.seed!(122)
        d = Uniform(q_l, q_r)



        #Variables for Lyapunov 
        N = Int(2e4)
        Δt = 1e-2 #smallest time inteval for the Lapunov calculation
        t = N*Δt
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0., 1., 0., 5., 0., 1., 0., 5., 2.;
        n_E = 3
        Es = range(cE[1],cE[end], length=n_E)



        j=3
        ICs = generate_initial_conditions(Es[j], parameters) #1.11Gb
        u_i = ICs[1] 

        λ_max, λs = Lyapunov_max2(u_i, parameters, N, Δt, 1e-3) #1Gb
        t = range(0, 200, length = N)
        plot(t, λs)
        println(λ_max)


        N2 = Int(2e6)
        Δt2 = 1e-4 #smallest time inteval for the Lapunov calculation
        t2 = N2*Δt2
        λ_max2, λs2 = Lyapunov_max2(u_i, parameters, N2, Δt2, 1e-3) #1Gb
        t = range(0, 200, length = N2)
        plot(t, λs2)
        println(λ_max2)

    ####

    # ---------------------- LYapunov time test --------------------------------------------------------------------------------
        function Lyapunov_max2(u_i::Vector{Float64}, p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,}, N::Int64, Δt::Float64,err::Float64) 
            @inbounds begin
                prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, p)
                # Solve the problem
                sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=N*1000)
                traj = sol.u
                
                #Defining the matrix for ICs vectors
                G = Matrix{Float64}(I, 4, 4)
                λ = zeros(4) 
                λ1 = zeros(4)
                λs = zeros(N)
                
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
                    λs[i] = maximum(λ1)/(i*Δt)
                    
                end
            end
            #println(time()-t)
            T = (N - 100/Δt)Δt
            λ = λ/(T)

            λ_max = maximum(λ)
            return λ_max, λs
        end
        function Lyapunov_adaptative(u_i::Vector{Float64}, σ_thr::Float64, t_max::Float64, p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,}, Δt::Float64,err::Float64; t_range= 10.0) 
            @inbounds begin
                prob = ODEProblem(EqM!, u_i, (0.0, t_max), saveat = Δt, p)
                # Solve the problem
                sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=Int(t_max/Δt)*1000)
                traj = sol.u
                ts = sol.t

                #Defining the matrix for ICs vectors
                G = Matrix{Float64}(I, 4, 4)
                λ = zeros(4) 
                λ1 = zeros(4)
                N = Int(t_max/Δt) #number of points
                Int_t = Int.(0:(t_range/Δt):(t_max/Δt)) .+ 1
                count = 1
                σ, λ_bar = 1.0, 1.0 # just to initialize
            
                for count in 1:length(Int_t)
                    if σ > σ_thr
                        λs = zeros(Int_t[count]-1)
                        println("$(Int_t[count]), $(Int_t[count]-1)")
                        for i in Int_t[count]:(Int_t[count+1]-1)
                            x = traj[i]
                            
                            J = Jacobian_qp(x,p)
                            #Evolution of G from J 

                            prob = ODEProblem(G_evolution!, G, (0.0, Δt), J)
                            
                            sol = solve(prob, Tsit5(), save_everystep = false, abstol=err, reltol=err)
                            G = sol.u[end]
                            
                            G,R = GS(G)
                            
                            λ1 += log.((diag(R)))
                            λs[i] = maximum(λ1)/(i*Δt)
                        end
                        λ_bar = mean(λs)
                        σ = std(λs)
                    else
                        break
                    end
                end
            end
            return λ_bar
        end
        
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
        
        roots, cE, λs, s_λ = crit_energies(parameters);
        
        n_E = 10
        Es = range(cE[1],cE[end], length=n_E)
        #ps = [zeros(1,1) for i in 1:n_E]
        λs = [Float64[] for i in 1:n_E]

        #Random initial positions
        q_l = -5.
        q_r = 5.
        Random.seed!(122)
        d = Uniform(q_l, q_r)

        #Variables for Lyapunov 
        N = Int(2e5)
        Δt = 1e-3 #smallest time inteval for the Lapunov calculation
        t = N*Δt
        u_i = [-1.1451149691219942, 0.3415408918391698, 4.423034394320682, -0.4021709240961071]
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
        


        prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, parameters)
        # Solve the problem
        sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, maxiters=N*1000)
        traj = sol.u
        sol.t
        size(traj)[1]
        for i in 19000:20000
            println(H_class(traj[i], parameters))
        end
        t_range = sol.t
        traj = hcat(t_range, reduce(hcat, traj)')
        reduce(hcat, traj)

        writedlm("C:/Users/edson/Desktop/trajectory.dat", traj)
        n_ICs = 1000
        p = parameters
        λ_0 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(0)_ICs_$(n_ICs).jld")["λs"];
        λ_p5 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(5.0)_ICs_$(n_ICs).jld")["λs"];
        λ_m5 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(-5.0)_ICs_$(n_ICs).jld")["λs"];
        
        hist(λ_0, bins=30, label = L"E = 0, \bar{λ} = %$(round(mean(λ_0), digits=3)), Int_r = %$(round(sum(λ_0 .< 0.1)/length(λ_0),digits=2))", color = "green", histtype="step")
        hist(λ_p5, bins=30, label = L"E = 5, \bar{λ} = %$(round(mean(λ_p5), digits=3)), Int_r = %$(round(sum(λ_p5 .< 0.1)/length(λ_p5),digits=2))", color = "red", histtype="step")
        hist(λ_m5, bins=30, label = L"E = -5, \bar{λ} = %$(round(mean(λ_m5), digits=3)), Int_r = %$(round(sum(λ_m5 .< 0.1)/length(λ_m5),digits=2))", color = "blue", histtype="step")
        plot(range(mean(λ_0), mean(λ_0), length=2), range(0,200,length=2), color="green", lw=2, ls="--")
        plot(range(mean(λ_p5), mean(λ_p5), length=2), range(0,200,length=2), color="red", lw=2, ls="--")
        plot(range(mean(λ_m5), mean(λ_m5), length=2), range(0,200,length=2), color="blue", lw=2, ls="--")
        
        ylabel("Counts", fontsize=18)
        xlabel(L"λ", fontsize=18)
        title("p = $(p)")
        legend(fontsize=10, loc="upper right")
        xlim(0,3.5)
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_Histogram_$(n_ICs)_3Energies.png")

        ICs = generate_initial_conditions(0., parameters)
        #for i in 1:length(ICs)
        i =1
            println("Initial Condition $(i)")
            u_i = ICs[i]
            λ_max, λs = Lyapunov_max2(u_i, parameters, N, Δt, 1e-3)
            λ_max1 = Lyapunov_adaptative(u_i, 0.1, 200.0, parameters, Δt, 1e-3)
            t = range(0, 200, length = N)
            plot(t, λs)
        #end
        println(λ_max)

        t = time()
        N2 = Int(2e7)
        Δt2 = 1e-4 #smallest time inteval for the Lapunov calculation
        t2 = N2*Δt2
        λ_max2, λs2 = Lyapunov_max2(u_i, parameters, N2, Δt2, 1e-3) #1Gb
        time() - t
        t = range(0, 200, length = N2)
        plot(t, λs2)
        println(λ_max2)

        t = time()
        N3 = Int(2e6)
        Δt3 = 1e-4 #smallest time inteval for the Lapunov calculation
        t3 = N3*Δt3
        λ_max2, λs2 = Lyapunov_simp(u_i, parameters, N3, Δt3, 1e-3) #1Gb
        time() - t
        t = range(0, 200, length = N3)
        plot(t, λs2)
        println(λ_max2)
        
        
        
        
        

        coupled_KPOs = CoupledODEs(EqM!, u_i, parameters)
        steps = 1000000
        lyapunovspectrum(coupled_KPOs,steps)
        12:18
    ####


    # ---------------------- Mean Lyapunov x ΔE --------------------------------------------------------------------------------
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
        roots,E_cl, rest =  crit_energies(p)
        
        using DelimitedFiles
        
        ns_job = [1, 50, 100]
        job=1;
        data = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_$(ns_job[job])_E_$(E_cl[1])$(ns_job[job+1]).dat");
        data2 = readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat")[ns_job[job]:ns_job[job+1], :];
        
        for job in 2:4
            data = vcat(data, readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat"))
            data2 = vcat(data2, readdlm("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_$(ns_job[job])_$(ns_job[job+1]).dat")[ns_job[job]:ns_job[job+1], :])
        end
        


        fig = figure(figsize=(10,5), layout="constrained");
        gs = fig.add_gridspec(1,2);
        element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
        slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
        ax = fig.add_subplot(element(0,0));
        plot(data[:,1], data[:,2], ".", color="red", markersize=5);
        
        roots, E_cl, λs, s_λs = crit_energies(p);
        E_cl'
        min_S = minimum(data[:,2]);
        max_S = maximum(data[:,2]);
        for i in [1, 3, 5,9]
            plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3))")
        end
        plot(data2[:,1], data2[:,2], "-", color="blue", markersize=3, label = "Mean ($(round(mean(data2[:,2][2:end]), digits=3)))");
        #ylim(-6,6)
        #xlim(-6,6)
        xlabel(L"E",fontsize=12);
        ylabel(L"λ",fontsize=12);
        
        ax = fig.add_subplot(element(0,1));
        plot(data[:,1], data[:,2], ".", color="red", markersize=5);
        
        for i in [1, 3, 5, 10,11]
            plot(range(E_cl[i],E_cl[i], length=2), range(min_S,max_S, length=2), lw = 3,label = s_λs[i] * "= $(round(E_cl[i], digits=3))")
        end
        plot(data2[:,1], data2[:,2], "-", color="blue", markersize=3, label = "Mean (Total = $(round(mean(data2[:,2][2:end]), digits=3)))");
        legend(loc="upper left",fontsize=10, shadow=true);
        #ylim(-6,6)
        xlim(E_cl[1], E_cl[5]);
        xlabel(L"E",fontsize=12);
        ylabel(L"λ",fontsize=12);
        fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(p)");

        
        
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energies_$(p).png")
        
        
    ###
    
    # ---------------------- Protection tail --------------------------------------------------------------------------------
        using LinearAlgebra
        using Pkg
        Pkg.add(["DifferentialEquations"])
        Pkg.add(["Dates"])
        Pkg.add(["Roots"])
        Pkg.add(["Distributions"])
        Pkg.add(["JLD"])
        using JLD
        using DifferentialEquations
        using Random, Distributions, Dates,Roots

        include("Closed_2KPOs_functions.jl")
        using .Lyapunov_Energies


        job = parse(Int, ARGS[1])

        parameters = Tuple(parse.(Float64, ARGS[2:end]))
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters
        println("Parameters Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(parameters)")
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
        
                
        roots, cE, λs, s_λs = crit_energies(parameters);

    
        #Variables for Lyapunov
        N_Ly = Int(2e3)   #t = dt*N
        Δt = 1e-1 #smallest time inteval for the Lapunov calculation
        t = N_Ly*Δt
    


        λ_tail = 1e-1
        Es = Vector{Float64}()
        E = cE[1] + 0.5 #Initial_energy
        dE = 1 #Energy step
        λs_ICs = Vector{Vector{Float64}}()
        λ_max = 0. #Initial value for the maximum Lyapunov exponent
        while λ_max < λ_tail
            ICs, ωs = Weighted_initial_conditions(Es[j], p, -5., 5.)
            n_ICs = size(ICs)[1]

            λs = Vector{Float64}()
            for i in 1:n_ICs
                λ_max  = Lyapunov_max(ICs[i], parameters, N_Ly, Δt, 1e-3)
                push!(λs, λ_max)
            end
            push!(λs_ICs, λs)
            push!(Es, E)
            E += dE
            λ_max = maximum(λs)
            if λ_max < 1e-2
                println("λ_max < 0.01 for E = $(E)")
            end
            if λ_max < 1e-3
                println("λ_max < 0.001 for E = $(E)")
            end
        end
        save("data/Lyapunov_Erange/λ_p_$(parameters)_.jld", "λs_ICs", λs_ICs)
        save("data/Lyapunov_Erange/E_p_$(parameters)_.jld", "Es", Es)



        
    ###

    
    # ---------------------- Protection tail 2 --------------------------------------------------------------------------------
        #CODE
            using LinearAlgebra
            using Pkg
            Pkg.add(["DifferentialEquations"])
            Pkg.add(["Dates"])
            Pkg.add(["Roots"])
            Pkg.add(["Distributions"])
            Pkg.add(["JLD"])
            using JLD
            using DifferentialEquations
            using Random, Distributions, Dates,Roots

            include("Closed_2KPOs_functions.jl")
            using .Lyapunov_Energies


            job = parse(Int, ARGS[1])

            parameters = Tuple(parse.(Float64, ARGS[2:end]))
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters
            println("Parameters Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $(parameters)")

        
            #Variables for Lyapunov
            N_Ly = Int(2e3)   #t = dt*N
            Δt = 1e-1 #smallest time inteval for the Lapunov calculation
            t = N_Ly*Δt
        

            function find_E_tail(E_start, p)
                λ_tail = 1e-1
                E = E_start + 0.5 #Initial_energy
                dE = 1 #Energy step
                λ_max = 0. #Initial value for the maximum Lyapunov exponent
                flag = true
                E_tail, E_tail2 = 0., 0. #Initial value for the maximum Lyapunov exponent
                while flag
                    ICs, ωs = Weighted_initial_conditions(E, p, -5., 5.)
                    n_ICs = size(ICs)[1]
                    
                    for i in 1:n_ICs
                        λ_max  = Lyapunov_max(ICs[i], parameters, N_Ly, Δt, 1e-3)
                        if λ_max > λ_tail
                            flag = false
                            E_tail = E
                        end
                    end
                    if λ_max < 1e-2
                        E_tail2 = E
                    end
                    E += dE
                end
                return E_tail, E_tail2, λ_max
            end

        
            function adaptive_Etail_search(λ_tail, E_start, parameters, steps=[10., 5.0, 1.0, 0.5, 0.1])
                """
                    Search for the smallest energy `E` such that λ >= λ_tail`.
                    It starts from `E_start` and refines using successive step sizes.
                """
                
                E_low = E_start
                E_high = nothing

                for ΔE in steps
                    E = E_low - ΔE
                    flag_ = true
                    while flag_
                        E += ΔE
                        println("E = $(E)")
                        ICs, ωs = Weighted_initial_conditions(E, parameters, -5., 5.)
                        n_ICs = size(ICs)[1]

                        for i in 1:n_ICs
                            λ_max  = Lyapunov_max(ICs[i], parameters, N_Ly, Δt, 1e-3)
                            if λ_max > λ_tail
                                flag_ = false
                                println("hit")
                                break #once λ is larger we dont need to compute more λs
                            end
                        end
                    end
                    # Found crossing in this step size
                    E_high = E
                    E_low = E - ΔE
                    println("E_high = $(E_high), E_low = $(E_low)")
                end

                return (E_high + E_low)/2
            end


            
            n_p = 100
            p_ = range(0.01, 5., length = n_p)
            E_tail = zeros(n_p)
            ns_job = [1, 50, 101]

            for i in ns_job[job]:(ns_job[job]-1)
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]
                roots, cE, λs, s_λs = crit_energies(p);
                E_start = cE[1] + 0.5
                E_tail[i] = adaptive_Etail_search(1e-1, E_start, p)
            end
            save("data/Lyapunov_Erange/E_tail_p_$(parameters)_γ_$(ns_job[job])_$(ns_job[job+1]).jld", "Es", Es)
        ###

        
        ###Plots
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
        
            n_p = 100
            p_ = range(.01, 13., length = n_p)
            ns_job = 1:2:101
            λ_tail = 1e-1
            step = 0.1
            Es = fill(NaN, 100)
            no_data = []
            for job in 1:length(ns_job)
                try 
                    E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_γ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                    GS = zeros(2)
                    count = 1
                    for i in ns_job[job]:(ns_job[job+1]-1)
                        roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
                        GS[count] = cE[1]
                        count+=1
                    end
                    Es[ns_job[job]:ns_job[job+1]-1] = GS + E_tail
                    
                catch
                    println("Missing job $(job)")
                    push!(no_data,job)
                end
            end
            println("no_data = $(no_data)")

            Es[9]
            plot(p_, Es, "o", label = L"λ_{t} = %$(λ_tail)")
            xlabel(L"γ", fontsize=15)
            ylabel(L"ΔE_{tail}", fontsize=15)
            title("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.")
            legend(fontsize=10, shadow=true, loc = "upper left");
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Energy_tail_$(p)_λ_$(λ_tail)_γ.png")

            ###Lyapunov_Energies_(0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0)_E_-12.5_0.0_1_50_ICs_100



        
    ###

    ## ------------------Microcanonical sampling For Lyapunov vs Energy----------------------------------
        #CODE
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
            
            roots_, cE, λs, s_λ = crit_energies(p,7);
            
            n_E = 2
            Es = range(cE[1],cE[end], length=n_E)
            #ps = [zeros(1,1) for i in 1:n_E]
            λs = [Float64[] for i in 1:n_E]
            λ_mean, λ_mean2 = zeros(n_E), zeros(n_E)
            #Variables for Lyapunov 
            N = Int(2e5)
            Δt = 1e-3 #smallest time inteval for the Lapunov calculation
            t = N*Δt


            #=Memory test
                j=2
                mem1 = @allocated generate_initial_conditions(Es[j], parameters) #1.11Gb
                mem2 = @allocated generate_initial_conditions_p2_0(Es[j], parameters) #7.15 Mb
                mem3 = @allocated  Lyapunov_max(Ics[1], parameters, N, Δt, 1e-3) #1Gb
                println("$((mem1 + mem2 + mem3)/ 1e9) Gb")
            =#
            
            t= time()
            #for j in 1:n_E
                #ICs, ωs = Weighted_initial_conditions(Es[j], p, -5., 5.)
                ICs, ωs = Weighted_initial_conditions(0, p, -5., 5.)
                lim = length(ICs)
                #ICs
                #In case no IC is found (happens for GS)
                if isempty(ICs)
                    continue
                end
                
                n_ICs= size(ICs)[1]
                #λs = Float64[]
                for i in 1:n_ICs
                    λ_max  = Lyapunov_max(ICs[i], p, N, Δt, 1e-3)
                    push!(λs, λ_max)
                end
                λ_max, λ_max2 = Lyapunov_max2(ICs[1], p, N, Δt, 1e-3)
                plot((1:N).*Δt,λ_max2)
                xlabel("t")
                ylabel("λ")
                title("$(ICs[1])")
                #Mean Lyapunov exponent with weights
                λ_mean[j] = sum( ωs .* λs[j]) / sum(ωs)
                #Mean Lyapunov exponent without weights
                λ_mean2[j] = sum(λs[j]) / length(λs[j])
            
            function Lyapunov_max2(u_i::Vector{Float64}, p::Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,}, N::Int64, Δt::Float64,err::Float64)
                @inbounds begin
                    prob = ODEProblem(EqM!, u_i, (0.0, N*Δt),saveat = Δt, p)
                    # Solve the problem
                    sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
                    traj = sol.u
                    
                    #Defining the matrix for ICs vectors
                    G = Matrix{Float64}(I, 4, 4)
                    λ = zeros(4)
                    λs = zeros(N)
                    λ1 = zeros(4)
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
                        λs[i] = maximum(λ1)/(i*Δt)
                        
                    end
                end
                #println(time()-t)
                T = (N - 100/Δt)Δt
                λ = λ/(T)

                λ_max = maximum(λ)
                return λ_max,λs
            end

            data = zeros(100,5)
            for i in 1:100
                data[i,5] = λs[i]
                #for j in 1:4
                    data[i,1:4] = ICs[i]
                #end
            end
            #end
            println("$((time()-t)/ 60) minutes")
            writedlm("C:/Users/edson/Desktop/data.dat", data)
            save("data/Lyapunov/Weighted_Mean_Lyapunov_Energies_$(p)_E_$(Es[1])_$(Es[end])_$(ns_job[job])_$(ns_job[job+1]).dat", "λ_mean", λ_mean)
            save("data/Lyapunov/Mean_Lyapunov_Energies_$(p)_E_$(Es[1])_$(Es[end])_$(ns_job[job])_$(ns_job[job+1]).dat", "λ_mean2", λ_mean2)
            save("data/Lyapunov/Lyapunov_Energies_$(p)_E_$(Es[1])_$(Es[end])_$(ns_job[job])_$(ns_job[job+1]).dat", " λs", λs)
        ###

        ### Testing Lyapunov , detuning
                E = -70
                ICs, ωs = Weighted_initial_conditions(E, p, -5., 5.);
                lim = length(ICs)

                #In case no IC is found (happens for GS)
                if isempty(ICs)
                    continue
                end
                
                n_ICs= 10#size(ICs)[1]
                λs = Float64[]
                for i in 1:n_ICs
                    λ_max  = Lyapunov_max(ICs[i], p, N, Δt, 1e-3)
                    push!(λs, λ_max)
                end 

                #Mean Lyapunov exponent with weights
                λ_mean[j] = sum( ωs .* λs[j]) / sum(ωs)
                #Mean Lyapunov exponent without weights
                λ_mean2[j] = sum(λs[j]) / length(λs[j])
                
        ###
        
        #Single Energie
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
            E = -32.;
            N = Int(2e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            N*Δt
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.
            final_points, λ_maxs = Poincare_Lyapunov(E,p,N, Δt,q1_l,q1_r, q2_l,q2_r)
            
            
            
            fig = figure(figsize=(5, 10), layout="constrained")
            gs = fig.add_gridspec(2,1)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            scat_plot = ax.scatter(final_points[:,1], final_points[:,3], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"q_2",fontsize=12)

            ax = fig.add_subplot(element(1,0))
            scat_plot = ax.scatter(final_points[:,1], final_points[:,2], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"p_1",fontsize=12)




        ###

        #Multiple Energies
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213    
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 1., 1., 0., 5., 1., 1., 0., 5., 1.;
            n_E = 5
            roots_, cE, λs, s_λ = crit_energies(p);
            Es = range(cE[1]+1.,cE[4], length=n_E)
            #Es = range(cE[5]+1.,cE[9], length=n_E)
            collect(Es)
            ps = [zeros(1,1) for i in 1:n_E]
            λs = [zeros(1) for i in 1:n_E]

            N, Δt = Int(2e4), 1e-2; #smallest time inteval for the Lapunov calculation
            N*Δt
            #limits for Poincare plot
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.

            t= time()
            for i in 1:n_E
                final_points, λ_maxs = Poincare_Lyapunov(Es[i], p, N, Δt,q1_l,q1_r, q2_l,q2_r)
                ps[i] = final_points
                λs[i] = λ_maxs
            end
            println("Time taken: ", time()-t, " seconds")

            fig = figure(figsize=(60, 10), layout="constrained")
            gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.25])
            #gs = fig.add_gridspec(2,5)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            fig.subplots_adjust(right=0.85) 
            for i in 1:5
                ax = fig.add_subplot(element(0,i-1))
                title("E = $(Es[i])")
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,3], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-q1_l,q1_l)
                if i == 1
                    ax.set_ylabel(L"q_2",fontsize=12)
                else
                    yticks([])
                end
                xticks([])
                ax.set_xlim(-q1_l,q1_l)

                ax = fig.add_subplot(element(1,i-1))
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,2], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-q1_l,q1_l)
                ax.set_xlim(-q1_l,q1_l)
                
                if i == 1
                    ax.set_ylabel(L"p_1",fontsize=12)
                else
                    yticks([])
                end
                ax.set_xlabel(L"q_1",fontsize=12)
                
                if i == 5
                    #[x, y, width, height]
                    cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
                    cbar = plt.colorbar(scat_plot, cax = cax_)
                end
            end
            text(0.3, -0.02, "λ", fontsize=18, verticalalignment="top")

            fig.suptitle(L" Δ_{1}, K_{1}, ξ_{1,1}, ξ_{2,1}, Δ_{2}, K_{2}, ξ_{1,2}, ξ_{2,2}, γ "*"= $(p)",fontsize=14)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_$(p)_E_$(collect(Es)).png")
            close()
        #####


        #Multiple parameters 
            n_p = 3
            γs = [0, 0.3 , 1.]
            ps = [zeros(1,1) for i in 1:n_p]
            λs = [zeros(1) for i in 1:n_p]

            for i in 1:n_p
                parameters = 0.,0.,5.,5.,1.,γs[i]
                final_points, λ_maxs = Poincare_Lyapunov(-32.,parameters)
                ps[i] = final_points
                λs[i] = λ_maxs
            end

            fig = figure(figsize=(30, 10), layout="constrained")
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.])
            #gs = fig.add_gridspec(2,5)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            fig.subplots_adjust(right=0.85) 
            for i in 1:n_p
                ax = fig.add_subplot(element(0,i-1))
                #title("E = $(Es[i])")
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,3], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-6,6)
                if i == 1
                    ax.set_ylabel(L"q_2",fontsize=12)
                else
                    yticks([])
                end
                xticks([])
                ax.set_xlim(-6,6)

                ax = fig.add_subplot(element(1,i-1))
                scat_plot = ax.scatter(ps[i][:,1], ps[i][:,2], c = λs[i], s = 0.1, vmin=0., vmax = 3.)
                ax.set_ylim(-6,6)
                ax.set_xlim(-6,6)
                
                if i == 1
                    ax.set_ylabel(L"p_1",fontsize=12)
                else
                    yticks([])
                end
                ax.set_xlabel(L"q_1",fontsize=12)
                
                if i == 5
                    #[x, y, width, height]
                    cax_ = fig.add_axes([0.96, 0.05, 0.02, 0.905]) 
                    cbar = plt.colorbar(scat_plot, cax = cax_)
                end
            end
            
            text(1.1, -0.02, "λ", transform=ax.transAxes, fontsize=14, verticalalignment="top")

            fig.suptitle("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6])",fontsize=14)
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes/Classical_Kerr/Figures/Poincare_$(parameters)_E_$(collect(Es)).png")
            savefig("C:/Users/edson/Desktop/a.png")

        #####

        ### More...
            N = Int(2e4)
            Δt = 1e-2 #smallest time inteval for the Lapunov calculation
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
            
            roots, cE = crit_energies(parameters);
            cE = [-48.2526,  -23.8816, -20.8769,  -10.9252,  -3.82856,  1.54256]
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.
            final_points, λ_maxs = Poincare_Lyapunov2(parameters, N, Δt,q1_l,q1_r, q2_l,q2_r)
            fig = figure(figsize=(5, 10), layout="constrained")
            gs = fig.add_gridspec(2,1)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            scat_plot = ax.scatter(final_points[:,1], final_points[:,3], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"q_2",fontsize=12)

            ax = fig.add_subplot(element(1,0))
            scat_plot = ax.scatter(final_points[:,1], final_points[:,2], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"p_1",fontsize=12)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICs_Delmar.png")
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICs_$(parameters).png")
                

            N = Int(2e4)
            Δt = 1e-2 #smallest time inteval for the Lapunov calculation
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
            
            roots, cE, λs, s_λs = crit_energies(parameters);
            s_λs
            #Classification fixed point
            #1 :: center, minimum 

        ###
    ####
    ## ---------------------------------------------------------------------------------------------------------------

    

    ###-------------------------------------------- Figure 2 - Classical only ------------------------------------------------------
        
        #function plotting_Clas()
            ###Plots γ
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0., 1., 0., 5., 0., 1., 0., 5., 1.
                ICs= 1000
                fig = figure(figsize=(10,30), layout="constrained");
                gs = fig.add_gridspec(3,3);      
                element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
                symb_ = ["γ", "Δ", "ξ_{2}"]
                for k in 1:3
                    λs_p = [[Float64[]] for i in 1:3]
                    λmean_p = [Float64[] for i in 1:3]
                    Energies = [Float64[] for i in 1:3]

                    for j in 1:3
                        ax = fig.add_subplot(element(j-1,k-1))
                        if k ==1
                            ps = [0.1, 1.,5.]
                            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                        elseif k ==2
                            ps = [0.1, 1.,5.]
                            p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                        else
                            ps = [2., 5., 10.]
                            p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ;
                        end

                        roots, cE, λs_p31, s_λ = crit_energies(p,7);
                        n_E = 500
                        Es = range(cE[1],300+cE[1], length=n_E)
                        Energies[j] = Es .- cE[1] 
                        λs = [Float64[] for i in 1:n_E]
                        λ_mean = zeros(n_E)
                        data_miss= []
                        for job in 1:100
                            try
                                λ_mean[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_job_$(job)__ICs_$(ICs).jld")["λ_mean"][(5*(job-1) + 1):5*job]
                                λs[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_job_$(job)_ICs_$(ICs).jld")["λs"][(5*(job-1) + 1):5*job]
                            catch
                                #println("Missing job $(job)")
                                push!(data_miss, job)
                            end    
                        end
                        λs_p[j] = λs 
                        λmean_p[j] = λ_mean
                        #println("γ = $(ps[j])")
                        #println("data_miss = $(data_miss)")
                        #println(length(data_miss))



                        #pltos
                        plot(Energies[j], λmean_p[j], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                        for i in 1:length(λs_p[j])
                            scatter(range(Energies[j][i], Energies[j][i], length=length(λs_p[j][i])), λs_p[j][i], color="black", alpha=0.5,s=1);
                        end
                        ax.text(.03, 0.85, L"%$(symb_[k]) = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                        if k ==1 && j ==1
                            legend(fontsize=20, shadow=true, loc = "upper right");
                        end
                        if j ==3
                            xlabel("E", fontsize = 20)
                            xticks([0,100,200,300], fontsize=15)
                        else
                            xticks([])
                        end
                        if k ==1
                            ylabel("λ", fontsize = 20)
                            yticks([0,2,4,6], fontsize=15)
                            #xlim(0, 130)
                            #ylim(-.05, 3.5)
                        else
                            yticks([])
                        end
                        #yticks([0, 1, 2, 3], fontsize=15)
                        #xlim(0, 130)
                        ylim(-.05, 6.0)
                    end
                end
                savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energy.png")
            ###

            ###Plots Δ
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
            
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1.,5.]
                Energies = [Float64[] for i in 1:3]
                
                for j in 1:2
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                for j in 3:3
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                #Tail data
                p = 0., K1, ξ11, ξ21, 0., K2, ξ12, ξ22, γ;
                n_p = 100
                p_ = range(.01, 13., length = n_p)
                ns_job = 1:2:101
                λ_tail = 1e-1
                step = 0.1
                Es_tail = fill(NaN, 100)
                no_data = []
                for job in 1:length(ns_job)
                    try 
                        E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_Δ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                        GS = zeros(2)
                        count = 1
                        #for i in ns_job[job]:(ns_job[job+1]-1)
                        #    roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
                        #    GS[count] = cE[1]
                        #    count+=1
                        #end
                        if E_tail[1] == 0.0
                            continue                        end 
                        Es_tail[ns_job[job]:ns_job[job+1]-1] = E_tail 
                    catch
                        println("Missing job $(job)")
                        push!(no_data,job)
                    end
                end
                println("no_data = $(no_data)")
                
                elements = [[0,1], [1,1], [2,1]]
                for k in 1:3
                    ax = fig.add_subplot(element(elements[k][1],elements[k][2]))
                    plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩_{E}");
                    for i in 1:length(λs_p[k])
                        scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                    end
                    ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    if k ==3
                        xlabel("E", fontsize = 20)
                        xticks([-120, -80, -40, 0], fontsize=15)
                    else
                        xticks([])
                    end
                    xlim(-130, 0)
                    ylim(-.05, 3.5)
                    yticks([])
                end

                ax = fig.add_subplot(element(3,1))
                plot(p_, Es_tail, "o", color = "red")
                xlabel(L"Δ", fontsize=20)
                ylim(20,100)
                yticks([])
            ###
        #end
        plotting_Clas()
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_and_TAIL.png")

        #Cluster visualization()
            ###Plots γ
            function Cluster_visualizationγ(j)
                k = j
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            function Cluster_visualizationγ2(j)
                k = j
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            
            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [20.0, 46.0,60.0]#[0.5, 1., 5.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                Cluster_visualizationγ2(k)
                #if k ==3
                #    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                #else
                #    xticks([])
                #end
                ax.text(.03, 0.85, L"γ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                #xlim(0, 3000)
                ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_γ_big2.png")

            # PLots Δ
            function Cluster_visualizationΔ(j)
                k = j
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1., 5.]
                Energies = [Float64[] for i in 1:3]
                p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                #Es = range(cE[1],0, length=n_E)
                Es = range(0,0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0 - cE[1],500.0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
            end
            function Cluster_visualizationΔ2(j)
                k = j
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1., 5.]
                Energies = [Float64[] for i in 1:3]
                p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                #Es = range(cE[1],0, length=n_E)
                Es = range(0,0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0 - cE[1],500.0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
            end
            
            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [0.5, 1., 5.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                if k==3
                    Cluster_visualizationΔ2(k)
                else
                    Cluster_visualizationΔ(k)
                end
                if k ==3
                    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                else
                    xticks([])
                end
                ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                xlim(0, 1200)
                ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_Δ.png")
            ###Plots Δ
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
            
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1.,5.]
                Energies = [Float64[] for i in 1:3]
                
                for j in 1:2
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                for j in 3:3
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                #Tail data
                p = 0., K1, ξ11, ξ21, 0., K2, ξ12, ξ22, γ;
                n_p = 100
                p_ = range(.01, 13., length = n_p)
                ns_job = 1:2:101
                λ_tail = 1e-1
                step = 0.1
                Es_tail = fill(NaN, 100)
                no_data = []
                for job in 1:length(ns_job)
                    try 
                        E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_Δ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                        GS = zeros(2)
                        count = 1
                        #for i in ns_job[job]:(ns_job[job+1]-1)
                        #    roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
                        #    GS[count] = cE[1]
                        #    count+=1
                        #end
                        if E_tail[1] == 0.0
                            continue                        end 
                        Es_tail[ns_job[job]:ns_job[job+1]-1] = E_tail 
                    catch
                        println("Missing job $(job)")
                        push!(no_data,job)
                    end
                end
                println("no_data = $(no_data)")
                
                elements = [[0,1], [1,1], [2,1]]
                for k in 1:3
                    ax = fig.add_subplot(element(elements[k][1],elements[k][2]))
                    plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩_{E}");
                    for i in 1:length(λs_p[k])
                        scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                    end
                    ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    if k ==3
                        xlabel("E", fontsize = 20)
                        xticks([-120, -80, -40, 0], fontsize=15)
                    else
                        xticks([])
                    end
                    xlim(-130, 0)
                    ylim(-.05, 3.5)
                    yticks([])
                end

                ax = fig.add_subplot(element(3,1))
                plot(p_, Es_tail, "o", color = "red")
                xlabel(L"Δ", fontsize=20)
                ylim(20,100)
                yticks([])
            ###





            ### Plots ξ2
                ###Plots γ
            function Cluster_visualizationξ2(j)
                println(ps[j])
                k = j
                Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.0
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            function Cluster_visualizationξ22(j)
                println(ps[j])
                k = j
                Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.0
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end

            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [2., 5., 10.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                if k==3
                    Cluster_visualizationξ2(k)
                else
                    Cluster_visualizationξ22(k)
                end
                if k ==3
                    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                else
                    xticks([])
                end
                ax.text(.03, 0.85, L"ξ_{2} = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                #xlim(0, 1200)
                #ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_ξ2.png")
            
            ###
        #end

    ###
    
    
    ###-------------------------------------------- Figure 2 - Classical Lyapunov / Entropy ------------------------------------------------------
        
        #function Option 1 - 0.1, 1, 5 and Neff=5
            ###Plots γ
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0., 1., 0., 5., 0., 1., 0., 5., 1.
                ICs= 1000
                N = 200
                Neff = 5
                N_states=4000
                Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["Entropies"]
                Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["Es"]
                intv_Es = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["intv_Es"]
                mean_Ss = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22))_Δ.jld")["mean_Ss"]


                fig = figure(figsize=(12,8), layout="constrained");
                gs = fig.add_gridspec(2,3); 
                element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
                xx_lim = 200
                
                    for j in 1:3
                        ax = fig.add_subplot(element(0,j-1))
                        ps = [0.1, 1.,5.]
                        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];  

                        roots, cE, λs_p31, s_λ = crit_energies(p,7);
                        n_E = 500
                        Es1 = range(cE[1],300+cE[1], length=n_E)
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
                                println("Missing job $(job)")
                                push!(data_miss, job)
                            end    
                        end
                        println("γ = $(ps[j])")
                        println("data_miss = $(data_miss)")
                        println(length(data_miss))



                        #pltos
                        plot(Energies, λ_mean, "-", color="blue", markersize=5, label = L"⟨λ⟩");
                        for i in 1:length(λs)
                            scatter(range(Energies[i], Energies[i], length=length(λs[i])), λs[i], color="black", alpha=0.2,s=1);
                        end
                        ax.text(.03, 0.85, L"γ = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                        if j ==1
                            legend(fontsize=20, shadow=true, loc = "upper right");
                        end
                        #xlim(-0.05, xx_lim)
                        if j == 1
                            xlabel("E", fontsize = 20)
                            #xticks([0,100,200], fontsize=15)
                        else
                            #xticks([])
                        end
                        if j ==1
                            ylabel("λ", fontsize = 20)
                            yticks([0,2,4,6], fontsize=15)
                            #xlim(0, 130)
                            #ylim(-.05, 3.5)
                        else
                            yticks([])
                        end
                        #yticks([0, 1, 2, 3], fontsize=15)
                        #xlim(0, 130)
                        ylim(-.05, 6.0)



                        ax = fig.add_subplot(element(1,j-1))
                        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
                        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
                        
                        #=
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
                        =#
                        
                        #=
                        plot(Es[:,j] .- Es[:,j][1], Entropies[:,j], "o", color=colors_[j], alpha = 0.2)
                        plot(intv_Es[:,j], mean_Ss[:,j], color="black", label="⟨S⟩")
                        xlim(-5, xx_lim)
                        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
                        ylim(0.3,4)
                        ylabel(L"S", fontsize=20)
                        xlabel("E", fontsize=20,labelpad=-15)
                        xticks([0,xx_lim],fontsize=15)
                        yticks([0,2,4],fontsize=15)
                        =#

                    end
                savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energy.png")
            ###


        #end

        #function Option 1 - 0.1, 1, 10
            ###Plots γ
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0., 1., 0., 5., 0., 1., 0., 5., 1.
                ICs= 1000
                
 
                fig = figure(figsize=(12,8), layout="constrained");
                gs = fig.add_gridspec(2,3); 
                element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
                xx_lim = 300
                
                    for j in 1:3
                        ax = fig.add_subplot(element(0,j-1))
                        ps = [0.1, 1.,10.]
                        p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];  

                        roots, cE, λs_p31, s_λ = crit_energies(p,7);
                        n_E = 500
                        Es1 = range(cE[1],300+cE[1], length=n_E)
                        println( cE[1] )
                        Energies = Es1 .- cE[1] 
                        λs = [Float64[] for i in 1:n_E]
                        λ_mean = zeros(n_E)
                        data_miss= []
                        for job in 1:100
                            try
                                λ_mean[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_job_$(job)__ICs_$(ICs).jld")["λ_mean"][(5*(job-1) + 1):5*job]
                                λs[(5*(job-1) + 1):5*job] = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_job_$(job)_ICs_$(ICs).jld")["λs"][(5*(job-1) + 1):5*job]
                            catch
                                #println("Missing job $(job)")
                                push!(data_miss, job)
                            end    
                        end
                        #println("γ = $(ps[j])")
                        #println("data_miss = $(data_miss)")
                        #println(length(data_miss))



                        #pltos
                        plot(Energies, λ_mean, "-", color="blue", markersize=5, label = L"⟨λ⟩");
                        for i in 1:length(λs)
                            scatter(range(Energies[i], Energies[i], length=length(λs[i])), λs[i], color="black", alpha=0.2,s=1);
                        end
                        ax.text(.03, 0.85, L"γ = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                        if j ==1
                            legend(fontsize=20, shadow=true, loc = "upper right");
                        end
                        #xlim(-0.05, xx_lim)
                        if j == 1
                            xlabel("E", fontsize = 20)
                            #xticks([0,100,200], fontsize=15)
                        else
                            #xticks([])
                        end
                        if j ==1
                            ylabel("λ", fontsize = 20)
                            yticks([0,2,4,6], fontsize=15)
                            #xlim(0, 130)
                            #ylim(-.05, 3.5)
                        else
                            yticks([])
                        end
                        #yticks([0, 1, 2, 3], fontsize=15)
                        #xlim(0, 130)
                        ylim(-.05, 6.0)



                        ax = fig.add_subplot(element(1,j-1))
                        colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
                        markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
                        
                        #=
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
                        =#
                        
                        
                        Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$(p).jld")["Entropies"]
                        E = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$(p).jld")["E"]
                        intv_E = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/IntvE_p_$(p).jld")["intv_E"]
                        mean_S = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Mean_Ss_p_$(p).jld")["mean_S"]


                        plot(E .- E[1], Entropies, "o", color=colors_[j], alpha = 0.2)
                        plot(intv_E[1:end-1], mean_S, color="black", label="⟨S⟩")
                        xlim(-5, xx_lim)
                        legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
                        ylim(0.3,4)
                        ylabel(L"S", fontsize=20)
                        xlabel("E", fontsize=20,labelpad=-15)
                        xticks([0,xx_lim],fontsize=15)
                        yticks([0,2,4],fontsize=15)
                        

                    end
                savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energy.png")
            ###


        #end
        plotting_Clas()
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_and_TAIL.png")

        #function Option 1 - 0.1, 1, 5 until 600 
            ###Plots γ
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0., 1., 0., 5., 0., 1., 0., 5., 1.
                ICs= 1000
                Neff= 3
            
 
                fig = figure(figsize=(12,8), layout="constrained");
                gs = fig.add_gridspec(2,3); 
                element(i,j) = get(gs, (i,j)); # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j);
                xx_lim = 500
                
                for j in 1:3
                    ax = fig.add_subplot(element(0,j-1))
                    ps = [0.1, 1.,5.]
                    p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];  

                    roots, cE, λs_p31, s_λ = crit_energies(p,7);
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
                    ax.text(.03, 0.85, L"γ = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
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

                    ax = fig.add_subplot(element(1,j-1))
                    colors_ = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"];
                    markers_ = ["o", "v", "s", "D", "^", "<", ">", "p", "*", "h"];
                                            
                    
                    Entropies = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Entropy_p_$(p)_Neff_$(Neff).jld")["Entropies"]
                    E = (load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/Entropy/Energies_p_$(p)_Neff_$(Neff).jld")["E"]) ./ Neff^2
                    mean_S = Float64[]
                    div_len = 40
                    intv_E =  range(minimum(E),maximum(E)+1,length=div_len)
                    for i in 1: (div_len-1)
                        index = findall(x -> x >= intv_E[i] && x < intv_E[i+1], E)
                        push!(mean_S, mean(Entropies[index]))
                    end

                    plot(E .- E[1], Entropies, "o", color=colors_[j], alpha = 0.2)
                    plot(intv_E[1:end-1] .- E[1] , mean_S, color="black", label="⟨S⟩")
                    xlim(-5, xx_lim)
                    legend(frameon=false,fontsize=15, shadow=true, loc = "upper left")
                    ylim(0.3,4)
                    ylabel(L"S", fontsize=20)
                    xlabel("E", fontsize=20,labelpad=-15)
                    xticks([0,xx_lim],fontsize=15)
                    yticks([0,2,4],fontsize=15)
                    
                end
                savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_Energy.png")
            
            
            
            
            ###


        #end
        plotting_Clas()
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_and_TAIL.png")


        #Cluster visualization()
            ###Plots γ
            function Cluster_visualizationγ(j)
                k = j
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            function Cluster_visualizationγ2(j)
                k = j
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            
            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [20.0, 46.0,60.0]#[0.5, 1., 5.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                Cluster_visualizationγ2(k)
                #if k ==3
                #    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                #else
                #    xticks([])
                #end
                ax.text(.03, 0.85, L"γ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                #xlim(0, 3000)
                ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_γ_big2.png")

            # PLots Δ
            function Cluster_visualizationΔ(j)
                k = j
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1., 5.]
                Energies = [Float64[] for i in 1:3]
                p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                #Es = range(cE[1],0, length=n_E)
                Es = range(0,0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0 - cE[1],500.0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
            end
            function Cluster_visualizationΔ2(j)
                k = j
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1., 5.]
                Energies = [Float64[] for i in 1:3]
                p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                #Es = range(cE[1],0, length=n_E)
                Es = range(0,0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0 - cE[1],500.0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
            end
            
            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [0.5, 1., 5.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                if k==3
                    Cluster_visualizationΔ2(k)
                else
                    Cluster_visualizationΔ(k)
                end
                if k ==3
                    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                else
                    xticks([])
                end
                ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                xlim(0, 1200)
                ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_Δ.png")
            ###Plots Δ
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
            
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1.,5.]
                Energies = [Float64[] for i in 1:3]
                
                for j in 1:2
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                for j in 3:3
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                #Tail data
                p = 0., K1, ξ11, ξ21, 0., K2, ξ12, ξ22, γ;
                n_p = 100
                p_ = range(.01, 13., length = n_p)
                ns_job = 1:2:101
                λ_tail = 1e-1
                step = 0.1
                Es_tail = fill(NaN, 100)
                no_data = []
                for job in 1:length(ns_job)
                    try 
                        E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_Δ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                        GS = zeros(2)
                        count = 1
                        #for i in ns_job[job]:(ns_job[job+1]-1)
                        #    roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
                        #    GS[count] = cE[1]
                        #    count+=1
                        #end
                        if E_tail[1] == 0.0
                            continue                        end 
                        Es_tail[ns_job[job]:ns_job[job+1]-1] = E_tail 
                    catch
                        println("Missing job $(job)")
                        push!(no_data,job)
                    end
                end
                println("no_data = $(no_data)")
                
                elements = [[0,1], [1,1], [2,1]]
                for k in 1:3
                    ax = fig.add_subplot(element(elements[k][1],elements[k][2]))
                    plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩_{E}");
                    for i in 1:length(λs_p[k])
                        scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                    end
                    ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    if k ==3
                        xlabel("E", fontsize = 20)
                        xticks([-120, -80, -40, 0], fontsize=15)
                    else
                        xticks([])
                    end
                    xlim(-130, 0)
                    ylim(-.05, 3.5)
                    yticks([])
                end

                ax = fig.add_subplot(element(3,1))
                plot(p_, Es_tail, "o", color = "red")
                xlabel(L"Δ", fontsize=20)
                ylim(20,100)
                yticks([])
            ###





            ### Plots ξ2
                ###Plots γ
            function Cluster_visualizationξ2(j)
                println(ps[j])
                k = j
                Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.0
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            function Cluster_visualizationξ22(j)
                println(ps[j])
                k = j
                Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.0
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end

            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [2., 5., 10.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                if k==3
                    Cluster_visualizationξ2(k)
                else
                    Cluster_visualizationξ22(k)
                end
                if k ==3
                    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                else
                    xticks([])
                end
                ax.text(.03, 0.85, L"ξ_{2} = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                #xlim(0, 1200)
                #ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_ξ2.png")
            
            ###
        #end

    ###

   
    ###-------------------------------------------- Mean_Lyapunov vs γ for diff E ranges ------------------------------------------------------
        
        #function plotting_Clas()
            ###Plots γ
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
            
                ns_job = [1, 50, 101]
                n_p = 50
                ps = range(0.01, 10.0, length=n_p)
                λ_bs, λ_as = fill(NaN, n_p), fill(NaN, n_p)
                
                for j in 1:n_p
                    try 
                        λ_b = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(j)_E_1_ICs_100_E_0.jld")["λ_mean"][:,1]
                        λ_a = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(j)_E_50_ICs_100_E_0.jld")["λ_mean"][:,1]
                        λ_bs[j] = mean(λ_b)
                        λ_as[j] = mean(λ_a)
                    catch
                        println("Missing $j")
                        continue
                    end
                end
                
                fig = figure(figsize=(6,6), layout="constrained");
                plot(ps, λ_bs, "o-", color = "red")
                plot(ps, λ_as, "o-", color = "blue")
                xlabel(L"γ", fontsize=20)
                ylabel(L"λ", fontsize=20)
                yticks([20,60,100],fontsize=15)
                ylim(20,100)
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_γ.png")
            ###

            ###Plots Δ
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
            
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1.,5.]
                Energies = [Float64[] for i in 1:3]
                
                for j in 1:2
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                for j in 3:3
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                #Tail data
                p = 0., K1, ξ11, ξ21, 0., K2, ξ12, ξ22, γ;
                n_p = 100
                p_ = range(.01, 13., length = n_p)
                ns_job = 1:2:101
                λ_tail = 1e-1
                step = 0.1
                Es_tail = fill(NaN, 100)
                no_data = []
                for job in 1:length(ns_job)
                    try 
                        E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_Δ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                        GS = zeros(2)
                        count = 1
                        #for i in ns_job[job]:(ns_job[job+1]-1)
                        #    roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
                        #    GS[count] = cE[1]
                        #    count+=1
                        #end
                        if E_tail[1] == 0.0
                            continue                        end 
                        Es_tail[ns_job[job]:ns_job[job+1]-1] = E_tail 
                    catch
                        println("Missing job $(job)")
                        push!(no_data,job)
                    end
                end
                println("no_data = $(no_data)")
                
                elements = [[0,1], [1,1], [2,1]]
                for k in 1:3
                    ax = fig.add_subplot(element(elements[k][1],elements[k][2]))
                    plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩_{E}");
                    for i in 1:length(λs_p[k])
                        scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                    end
                    ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    if k ==3
                        xlabel("E", fontsize = 20)
                        xticks([-120, -80, -40, 0], fontsize=15)
                    else
                        xticks([])
                    end
                    xlim(-130, 0)
                    ylim(-.05, 3.5)
                    yticks([])
                end

                ax = fig.add_subplot(element(3,1))
                plot(p_, Es_tail, "o", color = "red")
                xlabel(L"Δ", fontsize=20)
                ylim(20,100)
                yticks([])
            ###
        #end
        plotting_Clas()
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_and_TAIL.png")

        #Cluster visualization()
            ###Plots γ
            function Cluster_visualizationγ(j)
                k = j
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            function Cluster_visualizationγ2(j)
                k = j
                Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j];
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            
            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [20.0, 46.0,60.0]#[0.5, 1., 5.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                Cluster_visualizationγ2(k)
                #if k ==3
                #    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                #else
                #    xticks([])
                #end
                ax.text(.03, 0.85, L"γ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                #xlim(0, 3000)
                ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_γ_big2.png")

            # PLots Δ
            function Cluster_visualizationΔ(j)
                k = j
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1., 5.]
                Energies = [Float64[] for i in 1:3]
                p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                #Es = range(cE[1],0, length=n_E)
                Es = range(0,0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0 - cE[1],500.0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
            end
            function Cluster_visualizationΔ2(j)
                k = j
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1., 5.]
                Energies = [Float64[] for i in 1:3]
                p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                #Es = range(cE[1],0, length=n_E)
                Es = range(0,0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(cE[1], digits=3))_$(round(cE[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0 - cE[1],500.0-cE[1], length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1]+ cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1]+cE[1], digits=3))_$(round(Es[end]+cE[1], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
            end
            
            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [0.5, 1., 5.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                if k==3
                    Cluster_visualizationΔ2(k)
                else
                    Cluster_visualizationΔ(k)
                end
                if k ==3
                    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                else
                    xticks([])
                end
                ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                xlim(0, 1200)
                ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_Δ.png")
            ###Plots Δ
                K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5.,1.
            
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                ps = [0.5, 1.,5.]
                Energies = [Float64[] for i in 1:3]
                
                for j in 1:2
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                for j in 3:3
                    p = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ;
                    roots, cE, λs_p31, s_λ = crit_energies(p);
                    n_E = 100
                    Es = range(cE[1],cE[end], length=n_E)
                    Energies[j] = Es
                    #Es = range(-20,20, length=n_E)

                    job = 1
                    λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                    λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                    
                    job = 2
                    λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                    λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                    λs = vcat(λs, λs2)
                    λ_mean2 = vcat(λ_mean2, λ_mean22)
                    λs_p[j] = λs 
                    λmean_p[j] = λ_mean2
                end
                #Tail data
                p = 0., K1, ξ11, ξ21, 0., K2, ξ12, ξ22, γ;
                n_p = 100
                p_ = range(.01, 13., length = n_p)
                ns_job = 1:2:101
                λ_tail = 1e-1
                step = 0.1
                Es_tail = fill(NaN, 100)
                no_data = []
                for job in 1:length(ns_job)
                    try 
                        E_tail = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Erange/E_tail_p_$(p)_Δ_λ_tail_$(λ_tail)_$(ns_job[job])_$(ns_job[job+1])_ΔE_$(step).jld")["E_tail"][ns_job[job]:ns_job[job+1]-1];    
                        GS = zeros(2)
                        count = 1
                        #for i in ns_job[job]:(ns_job[job+1]-1)
                        #    roots, cE, λs, s_λs = crit_energies((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, p_[i]));
                        #    GS[count] = cE[1]
                        #    count+=1
                        #end
                        if E_tail[1] == 0.0
                            continue                        end 
                        Es_tail[ns_job[job]:ns_job[job+1]-1] = E_tail 
                    catch
                        println("Missing job $(job)")
                        push!(no_data,job)
                    end
                end
                println("no_data = $(no_data)")
                
                elements = [[0,1], [1,1], [2,1]]
                for k in 1:3
                    ax = fig.add_subplot(element(elements[k][1],elements[k][2]))
                    plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩_{E}");
                    for i in 1:length(λs_p[k])
                        scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                    end
                    ax.text(.03, 0.85, L"Δ = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    if k ==3
                        xlabel("E", fontsize = 20)
                        xticks([-120, -80, -40, 0], fontsize=15)
                    else
                        xticks([])
                    end
                    xlim(-130, 0)
                    ylim(-.05, 3.5)
                    yticks([])
                end

                ax = fig.add_subplot(element(3,1))
                plot(p_, Es_tail, "o", color = "red")
                xlabel(L"Δ", fontsize=20)
                ylim(20,100)
                yticks([])
            ###





            ### Plots ξ2
                ###Plots γ
            function Cluster_visualizationξ2(j)
                println(ps[j])
                k = j
                Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.0
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end
            function Cluster_visualizationξ22(j)
                println(ps[j])
                k = j
                Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.0
                ns_job = [1, 50, 101]
                λs_p = [[Float64[]] for i in 1:3]
                λmean_p = [Float64[] for i in 1:3]
                Energies = [Float64[] for i in 1:3]
                p = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
                roots, cE, λs_p31, s_λ = crit_energies(p,7);
                n_E = 100
                Es = range(cE[1],0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")[" λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .- cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                Es = range(0.,500.0, length=n_E)
                Energies[j] = Es
                #Es = range(-20,20, length=n_E)
                job = 1
                λs = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][1:49];
                λ_mean2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][1:49];
                job = 2
                λs2 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λs"][50:100];
                λ_mean22 = load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Mean_Lyapunov_Energies_$(p)_E_$(round(Es[1], digits=3))_$(round(Es[end], digits=3))_$(ns_job[job])_$(ns_job[job+1])_ICs_100.jld")["λ_mean2"][50:100];
                λs = vcat(λs, λs2)
                λ_mean2 = vcat(λ_mean2, λ_mean22)
                λs_p[j] = λs 
                λmean_p[j] = λ_mean2
                #Energies[k] = Energies[k] .-  cE[1]
                plot(Energies[k], λmean_p[k], "-", color="blue", markersize=5, label = L"⟨λ⟩");
                for i in 1:length(λs_p[k])
                    scatter(range(Energies[k][i], Energies[k][i], length=length(λs_p[k][i])), λs_p[k][i], color="black", alpha=0.5,s=1);
                end
                #xlim(0,200)
                
                #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov/Lyapunov_γ_$(ps[j]).png")
                #close()
            end

            fig = figure(figsize=(10,15), layout="constrained");
            gs = fig.add_gridspec(3,1);
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            ps = [2., 5., 10.]
            for k in 1:3
                ax = fig.add_subplot(element(k-1,0))
                if k==3
                    Cluster_visualizationξ2(k)
                else
                    Cluster_visualizationξ22(k)
                end
                if k ==3
                    xlabel("E", fontsize = 20)
                    #xticks([-120, -80, -40, 0], fontsize=15)
                else
                    xticks([])
                end
                ax.text(.03, 0.85, L"ξ_{2} = %$(ps[k])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                ylabel("λ", fontsize = 20)
                yticks([0, 2, 4,6], fontsize=15)
                #xlim(0, 1200)
                #ylim(-.05, 7.5)
            end
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_E_ξ2.png")
            
            ###
        #end

    ###

   
    #----------------------------------Poincare q2 = 0 - Energy based -----------------------------------------------
        #Single Energie
        
            function Poincare_q2_0(E, q_l, q_r, p; min_ics=100, max_attempts=100_000, timeout_seconds=10)
                ICs = generate_initial_conditions_q2_0(E, q_l, q_r, p; min_ics=min_ics, max_attempts=max_attempts, timeout_seconds=timeout_seconds)
                n_ICs = size(ICs)[1]
                println("Number of ICs: ", n_ICs)
                function linear_interp(x,x1,y1,x2,y2)
                    return ((x-x1)/(x2-x1))*y2 + ((x2-x)/(x2-x1))*y1
                end  
                function t_y0(t1,t2,y1,y2)
                    return (y2*t1 - y1*t2)/(y2-y1)
                end
                final_points = 0. #just to have a variable after the loop

                for i in 1:n_ICs
                    prob = ODEProblem(EqM!, ICs[i], (0.0, N*Δt),saveat = Δt, p)
                    sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
                    traj = sol.u
                    #getting just the q2 column
                    comp_sol = sol[3,1,:]
                    #Finding crossing points of q2 = 0
                    poi_all =  comp_sol[2:end-1] .* comp_sol[3:end] .<= 0
                    # just positive p2
                    poi_pos = sol[4,1,:][2:end-1] .>= 0
                    #Crossings at q2 = 0 with positive p2 
                    poi_pos = poi_pos .& poi_all

                    #In between times of zeros
                    t_ini = (0:Δt:N*Δt)[2:end-1][poi_pos] #times + for zeros
                    t_end = (0:Δt:N*Δt)[3:end][poi_pos] # times - for zeros
                    comp_ini = comp_sol[2:end-1][poi_pos] #q2 positive zero
                    comp_end = comp_sol[3:end][poi_pos] #q2 negatie zero
                    t_values=  @. t_y0(t_ini, t_end, comp_ini, comp_end) #Interpolation

                    dim_sols = size(t_values)[1]
                    solutions = zeros(dim_sols,4)
                    for component in [1,2,4] #p1, q1, p2
                        comp_ini = sol[component,1,:][2:end-1][poi_pos]
                        comp_end = sol[component,1,:][3:end][poi_pos]
                        interp_values = @. linear_interp( t_values, t_ini, comp_ini, t_end, comp_end)
                        solutions[:,component] = interp_values
                    end

                    if i == 1
                        final_points = copy(solutions)
                    else
                        final_points = vcat(final_points, solutions)
                    end
                end
                return final_points
            end
            function Poincare_q2_0_IC(E, q_l, q_r, p, IC; min_ics=100, max_attempts=100_000, timeout_seconds=10)
                function linear_interp(x,x1,y1,x2,y2)
                    return ((x-x1)/(x2-x1))*y2 + ((x2-x)/(x2-x1))*y1
                end  
                function t_y0(t1,t2,y1,y2)
                    return (y2*t1 - y1*t2)/(y2-y1)
                end
                final_points = 0. #just to have a variable after the loop

                    prob = ODEProblem(EqM!, IC, (0.0, N*Δt),saveat = Δt, p)
                    sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
                    traj = sol.u
                    #getting just the q2 column
                    comp_sol = sol[3,1,:]
                    #Finding crossing points of q2 = 0
                    poi_all =  comp_sol[2:end-1] .* comp_sol[3:end] .<= 0
                    # just positive p2
                    poi_pos = sol[4,1,:][2:end-1] .>= 0
                    #Crossings at q2 = 0 with positive p2 
                    poi_pos = poi_pos .& poi_all

                    #In between times of zeros
                    t_ini = (0:Δt:N*Δt)[2:end-1][poi_pos] #times + for zeros
                    t_end = (0:Δt:N*Δt)[3:end][poi_pos] # times - for zeros
                    comp_ini = comp_sol[2:end-1][poi_pos] #q2 positive zero
                    comp_end = comp_sol[3:end][poi_pos] #q2 negatie zero
                    t_values=  @. t_y0(t_ini, t_end, comp_ini, comp_end) #Interpolation

                    dim_sols = size(t_values)[1]
                    solutions = zeros(dim_sols,4)
                    for component in [1,2,4] #p1, q1, p2
                        comp_ini = sol[component,1,:][2:end-1][poi_pos]
                        comp_end = sol[component,1,:][3:end][poi_pos]
                        interp_values = @. linear_interp( t_values, t_ini, comp_ini, t_end, comp_end)
                        solutions[:,component] = interp_values
                    end

                    final_points = copy(solutions)
                return final_points, sol
            end

            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 5.0;
            crit_E = crit_energies(p,7)
            countor_energy(p, 10, false)
            E = 0.;
            ICs, w = Weighted_initial_conditions(E, p,-4.,4.)
            N = Int(2e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q1_r, q2_l,q_r = -5., 5., -5., 5.
            
            t = time()
            final_points = Poincare_q2_0(E, q_l, q1_r, p,min_ics=300, max_attempts=100_000, timeout_seconds=10);
            time() - t 
            fig = figure(figsize=(5, 5), layout="constrained")
            #gs = fig.add_gridspec(1,1)
            #element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            #slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            scat_plot = scatter(final_points[:,1], final_points[:,2], s = 0.1, color="black")
            xlabel(L"q_1",fontsize=17)
            ylabel(L"p_1",fontsize=17)
            xticks([-4,-2,0,2,4], fontsize = 15)
            yticks([-2,-1,0,1,2], fontsize = 15)

            min_idx = findall(x -> x > 0.1 && x < 0.2, λs)[6] #argmin(λs)
            max_idx = findall(x -> x > 0.9, λs)[4]
             max_idx = argmax(λs)

            ICs_i = ICs[min_idx]
            ICs_c = ICs[max_idx]#ICs[max_idx]
            final_points_i, sol_i = Poincare_q2_0_IC(E, q_l, q1_r, p, ICs_i,min_ics=200, max_attempts=100_000, timeout_seconds=10);
            final_points_c, sol_c = Poincare_q2_0_IC(E, q_l, q1_r, p, ICs_c,min_ics=200, max_attempts=100_000, timeout_seconds=10);
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            scat_plot = scatter(final_points_c[:,1], final_points_c[:,2], s = 0.1, color="red")
            scat_plot = scatter(final_points_i[:,1], final_points_i[:,2], s = 0.1, color="blue")
            #ax.set_ylim(4,-4)
            plot(sol_i.t[1:1000], sol_i[1, 1, :][1:1000], "-", color="blue", label="q1 - ICs min λ")
            plot(sol_c.t[1:500], sol_c[3, 1, :][1:500], "-", color="red", label="p1 - ICs min λ")

            #3D plot
            fig = figure(figsize=(10, 10), layout="constrained")
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=10, azim=280)
            λ_thr = 2.
            #scat_plot = ax.scatter3D(final_points[:,1][λ .< λ_thr], final_points[:,2][λ .< λ_thr], final_points[:,3][λ .< λ_thr], s = 0.1, color="black")
            scat_plot = ax.scatter3D(final_points[:,1], final_points[:,2], final_points[:,4], s = 0.1, color="black")
            ax.set_xlabel(L"q_1",fontsize=20)
            ax.set_ylabel(L"p_1",fontsize=20)
            ax.set_zlabel(L"p_2",fontsize=20)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/3DPoincare_q2_0_$(p)_E_$(E).png")
        
        #end
        
         #Multiple Energies
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
            N = Int(2e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q1_r, q2_l,q_r = -10, 10, -10, 10#-5., 5., -5., 5.
            n_p =5
            Es = [-1, -0.5, 0, 0.5,1]#[0.01, 1., 5.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            min_ics_list = [200, 200, 200, 200, 200]
            for i in 1:n_p
                p =  Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
                final_points = Poincare_q2_0(Es[i], q_l, q1_r, p, min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
            end

            function plot_Poinc()
                #fig = figure(figsize=(15, 5), layout="constrained")
                fig = figure(figsize=(25, 5), layout="constrained")
                gs = fig.add_gridspec(1,5)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-2,2)
                    yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    =#
                    ax.set_xlabel(L"q_1",fontsize=20)
                    title(L"E = %$(Es[i])",fontsize=20)
                end
                
            end
            plot_Poinc()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_q2_0_$(p)_E_$(Es)_γ.png")
        #####


        #Multiple parameters γ 
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.;
            N = Int(2e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q1_r, q2_l,q_r = -10, 10, -10, 10#-5., 5., -5., 5.
            n_p = 3
            ps = [1., 3., 5.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            E = 0. 
            min_ics_list = [100, 100, 100]
            for i in 1:n_p
                p =  Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[i]
                final_points = Poincare_q2_0(E, q_l, q1_r, p, min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
            end

            function plot_Poinc()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1),)
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-2,2)
                    yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    =#
                    ax.set_xlabel(L"q_1",fontsize=20)
                    title(L"γ = %$(ps[i])",fontsize=20)
                end
                
            end
            #plot_Poinc()
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_q2_0_$(p)_E_$(E)_γ.png")
            function plot_Poinc3D()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1),projection="3d")
                    ax.view_init(elev=10, azim=280)
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], final_points_list[i][:,4], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-2,2)
                    yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    =#
                    ax.set_xlabel(L"q_1",fontsize=20)
                    ax.set_ylabel(L"p_1",fontsize=20)
                    ax.set_zlabel(L"p_2",fontsize=20)
                    title(L"γ = %$(ps[i])",fontsize=20)
                end
                
            end
            plot_Poinc3D()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/3DPoincare_q2_0_$(p)_E_$(E)_γ_$(ps).png")
        #####

        
        #Multiple parameters Δ 
            K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1.;
            N = Int(1e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q1_r, q2_l,q_r = -50., 50., -50., 50.#-5., 5., -5., 5.
            n_p = 3
            ps = [10, 100., 200.]#[0.01, 1., 5.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            E = 0. 
            min_ics_list = [300, 100, 50]
            for i in 1:n_p
                p =  ps[i], K1, ξ11, ξ21, ps[i], K2, ξ12, ξ22, γ
                final_points = Poincare_q2_0(E, q_l, q1_r, p, min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
            end

            function plot_Poinc()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], s = 0.1, color="black")
                    
                    #ax.set_ylim(-2.5,2.5)
                    #yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        #yticks([])
                    end
    
                    if i ==1
                        ax.text(.38, 0.95, L"Δ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"Δ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    #ax.set_xlim(-5.5,5.5)
                    #xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    ax.set_xlabel(L"q_1",fontsize=20)
                end
                
            end
            plot_Poinc()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_q2_0_$(p)_E_$(E)_Δ_$(ps).png")
        #####

        
        #Multiple parameters ξ2 
            Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.;
            N = Int(1e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q_r = -10., 10.
            n_p = 3
            ps = [2., 5., 10.]#[0.01, 1., 5.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            E = 0. 
            min_ics_list = [200, 200, 200]
            for i in 1:n_p
                p =  Δ1, K1, ξ11, ps[i], Δ2, K2, ξ12, ps[i], γ
                final_points = Poincare_q2_0(E, q_l, q1_r, p, min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
            end

            function plot_Poinc()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], s = 0.1, color="black")
                    
                    ax.set_ylim(-2.5,2.5)
                    #yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        #yticks([])
                    end
                    title(L"ξ_{2} = %$(ps[i])", fontsize=20)
                    ax.set_xlim(-7,7)
                    #xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    ax.set_xlabel(L"q_1",fontsize=20)
                end
                
            end
            plot_Poinc()
    
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_q2_0_E_$(E)_ξ2_$(ps).png")
        #####

        ### More...
            N = Int(2e4)
            Δt = 1e-2 #smallest time inteval for the Lapunov calculation
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
            
            roots, cE = crit_energies(parameters);
            cE = [-48.2526,  -23.8816, -20.8769,  -10.9252,  -3.82856,  1.54256]
            q1_l,q1_r, q2_l,q2_r = -5., 5., -5., 5.
            final_points, λ_maxs = Poincare_Lyapunov2(parameters, N, Δt,q1_l,q1_r, q2_l,q2_r)
            fig = figure(figsize=(5, 10), layout="constrained")
            gs = fig.add_gridspec(2,1)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            scat_plot = ax.scatter(final_points[:,1], final_points[:,3], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"q_2",fontsize=12)

            ax = fig.add_subplot(element(1,0))
            scat_plot = ax.scatter(final_points[:,1], final_points[:,2], c = λ_maxs, s = 0.1, vmin=0., vmax = 3.)
            plt.colorbar(scat_plot, ax=ax, label="λ")
            #ax.set_ylim(4,-4)
            ax.set_xlabel(L"q_1",fontsize=12)
            ax.set_ylabel(L"p_1",fontsize=12)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICs_Delmar.png")
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_ICs_$(parameters).png")
                

            N = Int(2e4)
            Δt = 1e-2 #smallest time inteval for the Lapunov calculation
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.40825486855662213;
            
            roots, cE, λs, s_λs = crit_energies(parameters);
            s_λs
            #Classification fixed point
            #1 :: center, minimum 

        ###
    ####
    
    #----------------------------------Poincare p2=0 - Energy based -----------------------------------------------
        function Poincare_p2_0(E, q_l, q_r, p, p2_pos=false; min_ics=100, max_attempts=100_000, timeout_seconds=10)
                ICs = generate_initial_conditions_p2_0(E, q_l, q_r, p; min_ics=min_ics, max_attempts=max_attempts, timeout_seconds=timeout_seconds)
                n_ICs = size(ICs)[1]
                λ_maxs = zeros(n_ICs)
                
                println("Number of ICs: ", n_ICs)
                function linear_interp(x,x1,y1,x2,y2)
                    return ((x-x1)/(x2-x1))*y2 + ((x2-x)/(x2-x1))*y1
                end  
                function t_y0(t1,t2,y1,y2)
                    return (y2*t1 - y1*t2)/(y2-y1)
                end
                final_points = 0. #just to have a variable after the loop
                
                for i in 1:n_ICs
                    prob = ODEProblem(EqM!, ICs[i], (0.0, N*Δt),saveat = Δt, p)
                    sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
                    traj = sol.u

                    #getting just the p2 column
                    comp_sol = sol[4,1,:]
                    #Finding crossing points of p2 = 0
                    poi_all =  comp_sol[2:end-1] .* comp_sol[3:end] .<= 0
                    # just positive p2
                    if p2_pos == true
                        poi_pos = sol[4,1,:][2:end-1] .>= 0
                        #Crossings at q2 = 0 with positive p2 
                        poi_pos = poi_pos .& poi_all
                    else
                        poi_pos = poi_all
                    end
                    #In between times of zeros
                    t_ini = (0:Δt:N*Δt)[2:end-1][poi_pos] #times + for zeros
                    t_end = (0:Δt:N*Δt)[3:end][poi_pos] # times - for zeros
                    comp_ini = comp_sol[2:end-1][poi_pos] #p2 positive zero
                    comp_end = comp_sol[3:end][poi_pos] #p2 negatie zero
                    t_values=  @. t_y0(t_ini, t_end, comp_ini, comp_end) #Interpolation

                    dim_sols = size(t_values)[1]
                    solutions = zeros(dim_sols,4)
                    for component in [1,2,3] #q1,p1,q2 
                        comp_ini = sol[component,1,:][2:end-1][poi_pos]
                        comp_end = sol[component,1,:][3:end][poi_pos]
                        interp_values = @. linear_interp( t_values, t_ini, comp_ini, t_end, comp_end)
                        solutions[:,component] = interp_values
                    end

                    if i == 1
                        final_points = copy(solutions)
                    else
                        final_points = vcat(final_points, solutions)
                    end
                end
                return final_points
                
        end
        function Poincare_p2_0_λ(E, q_l, q_r, p, p2_pos=false; min_ics=100, max_attempts=100_000, timeout_seconds=10)
                ICs = generate_initial_conditions_p2_0(E, q_l, q_r, p; min_ics=min_ics, max_attempts=max_attempts, timeout_seconds=timeout_seconds)
                n_ICs = size(ICs)[1]
                λ_maxs = zeros(n_ICs)
                
                println("Number of ICs: ", n_ICs)
                function linear_interp(x,x1,y1,x2,y2)
                    return ((x-x1)/(x2-x1))*y2 + ((x2-x)/(x2-x1))*y1
                end  
                function t_y0(t1,t2,y1,y2)
                    return (y2*t1 - y1*t2)/(y2-y1)
                end
                final_points, λ_maxs = 0., 0. #just to have a variable after the loop
                
                for i in 1:n_ICs
                    prob = ODEProblem(EqM!, ICs[i], (0.0, N*Δt),saveat = Δt, p)
                    sol = solve(prob, Tsit5(), abstol=1e-7, reltol=1e-7, maxiters=N*1000)
                    traj = sol.u

                    λ_max  = Lyapunov_max(ICs[i], p,N, Δt, 1e-3)
                    #getting just the p2 column
                    comp_sol = sol[4,1,:]
                    #Finding crossing points of p2 = 0
                    poi_all =  comp_sol[2:end-1] .* comp_sol[3:end] .<= 0
                    # just positive p2
                    if p2_pos == true
                        poi_pos = sol[4,1,:][2:end-1] .>= 0
                        #Crossings at q2 = 0 with positive p2 
                        poi_pos = poi_pos .& poi_all
                    else
                        poi_pos = poi_all
                    end
                    #In between times of zeros
                    t_ini = (0:Δt:N*Δt)[2:end-1][poi_pos] #times + for zeros
                    t_end = (0:Δt:N*Δt)[3:end][poi_pos] # times - for zeros
                    comp_ini = comp_sol[2:end-1][poi_pos] #p2 positive zero
                    comp_end = comp_sol[3:end][poi_pos] #p2 negatie zero
                    t_values=  @. t_y0(t_ini, t_end, comp_ini, comp_end) #Interpolation

                    dim_sols = size(t_values)[1]
                    solutions = zeros(dim_sols,4)
                    for component in [1,2,3] #q1,p1,q2 
                        comp_ini = sol[component,1,:][2:end-1][poi_pos]
                        comp_end = sol[component,1,:][3:end][poi_pos]
                        interp_values = @. linear_interp( t_values, t_ini, comp_ini, t_end, comp_end)
                        solutions[:,component] = interp_values
                    end

                    if i == 1
                        final_points = copy(solutions)
                        λ_maxs = copy(λ_max*ones(dim_sols))
                    else
                        final_points = vcat(final_points, solutions)
                        λ_maxs = vcat(λ_maxs, λ_max*ones(dim_sols))
                    end
                end
                return final_points, λ_maxs
                
        end
        #Single Energie
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = 0., 1., 0., 5., 0., 1., 0., 5., 1.;
            roots, cE, λs, s_λ = crit_energies(p,7)
            countor_energy(p, 7, false)
            E = 0.;
            N = Int(4e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q_r = -5., 5.
            
            t = time()
            final_points, λ = Poincare_p2_0(E, q_l, q_r, p; min_ics=100, max_attempts=100_000, timeout_seconds=10);
            time() - t 
            
            fig = figure(figsize=(5, 5), layout="constrained")
            gs = fig.add_gridspec(1,1)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            #ax = fig.add_subplot(element(0,0))
            #title("δ = $(parameters[1]), ξ = $(parameters[3]), ϵ = $(parameters[6]), E = $(E)")
            #scat_plot = scatter(final_points[:,1], final_points[:,3], s = 0.1, color="black")
            fig = figure(figsize=(30, 30), layout="constrained")
            for i in 1:3
                for j in 1:3
                    ax = fig.add_subplot(3,3, (i-1)*3 + j, projection="3d")
                    ax.view_init(elev, azim=45)
                    scat_plot = ax.scatter3D(final_points[:,1], final_points[:,2], final_points[:,3], s = 0.1, color="black")
                    #ax.set_ylim(4,-4)
                    
                    ax.set_xlabel(L"q_1",fontsize=12)
                    ax.set_ylabel(L"p_1",fontsize=12)
                    ax.set_zlabel(L"q_2",fontsize=12)
                end
            end
            fig = figure(figsize=(20, 20), layout="constrained")
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=10, azim=280)
            λ_thr = 2.
            scat_plot = ax.scatter3D(final_points[:,1][λ .< λ_thr], final_points[:,2][λ .< λ_thr], final_points[:,3][λ .< λ_thr], s = 0.1, color="black")
            for j in 1:length(roots)
                ax.scatter3D(roots[j][1], roots[j][2], roots[j][3], s = 500., color="blue", marker="x")
            end
            ax.set_xlabel(L"q_1",fontsize=20)
            ax.set_ylabel(L"p_1",fontsize=20)
            ax.set_zlabel(L"q_2",fontsize=20)
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/3DPoincare_p2_0_$(p)_E_$(E).png")
        ###
        
        #Multiple parameters γ 
            
            Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5.;
            N = Int(2e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q1_r, q2_l,q_r = -5., 5., -5., 5.
            n_p = 3
            ps = [1., 3., 5.]#[0.01, 1., 5.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            E = 0. 
            min_ics_list = [100, 100, 100]
            for i in 1:n_p
                t = time()
                p =  Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[i]
                final_points = Poincare_p2_0(E, q_l, q1_r, p, false,  min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
                println(time() - t)
            end

            function plot_Poinc()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,3], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-2,2)
                    yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    =#
                    ax.set_xlabel(L"q_1",fontsize=20)
                    title(L"γ = %$(ps[i])",fontsize=20)
                end
                
            end
            #plot_Poinc()
            #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures//Poincare_p2_0_$(p)_E_$(E)_γ.png")
            function plot_Poinc3D()
               fig = figure(figsize=(30, 10), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1),projection="3d")
                    ax.view_init(elev=10, azim=280)
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], final_points_list[i][:,3], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-2,2)
                    yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    =#
                    ax.set_xlabel(L"q_1",fontsize=20)
                    ax.set_ylabel(L"p_1",fontsize=20)
                    ax.set_zlabel(L"q_2",fontsize=20)
                    title(L"γ = %$(ps[i])",fontsize=20)
                end
                
            end
            plot_Poinc3D()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/3DPoincare_p2_0_$(p)_E_$(E)_γ_$(ps).png")
        #####

        
        #Multiple parameters Δ 
            K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1.;
            N = Int(1e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q_r = -5., 5.
            n_p = 3
            ps = [0.01, 1., 5.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            E = 0. 
            min_ics_list = [200, 200, 200]
            for i in 1:n_p
                p =  ps[i], K1, ξ11, ξ21, ps[i], K2, ξ12, ξ22, γ
                final_points = Poincare_p2_0(E, q_l, q1_r, p, false,min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
            end

            function plot_Poinc()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,3], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-2,2)
                    yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    =#
                    ax.set_xlabel(L"q_1",fontsize=20)
                    title(L"Δ = %$(ps[i])",fontsize=20)
                end
                
            end
            plot_Poinc()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_q2_0_$(p)_E_$(E)_Δ_$(ps).png")
        #####
        
                
        #Multiple parameters ξ2
            Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1.;
            N = Int(1e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q_r = -10., 10.
            n_p = 3
            ps = [2., 5., 10.]
            final_points_list = [zeros(1,1) for i in 1:n_p]
            E = 200. 
            min_ics_list = [200, 200, 200]
            for i in 1:n_p
                 p =  Δ1, K1, ξ11, ps[i], Δ2, K2, ξ12, ps[i], γ
                final_points = Poincare_p2_0(E, q_l, q1_r, p, false,min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
            end

            function plot_Poinc()
               fig = figure(figsize=(15, 5), layout="constrained")
                gs = fig.add_gridspec(1,3)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_p
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,3], s = 0.1, color="black")
                
                    ax.set_ylim(-10,10)
                    #yticks([-2,-1, 0,1,2], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"p_1",fontsize=20)
                    else
                        yticks([])
                    end
                    #=
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    =#
                    ax.set_xlim(-10,10)
                    #xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    ax.set_xlabel(L"q_1",fontsize=20)
                    title(L"ξ_{2} = %$(ps[i])",fontsize=20)
                end
                
            end
            plot_Poinc()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare_q2_0_$(p)_E_$(E)_Δ_$(ps).png")
        #####
        
        #Multiple Energies
            p = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = 0., 1., 0., 5., 0., 1., 0., 5., 5.;
            N = Int(1e4);
            Δt = 1e-2; #smallest time inteval for the Lapunov calculation
            q_l,q_r = -5., 5.
            n_E = 5
            roots_, cE, λs, s_λ = crit_energies2(p, 100,q_r);
            #Es = range(cE[1]+1.,0., length=n_E)
            Es = [-1, -0.5, 0, 0.5, 1.]#range(0,500., length=n_E)
            final_points_list = [zeros(1,1) for i in 1:n_E]
            min_ics_list = [200, 200, 200, 200, 200]
            
            for i in 1:n_E
                t = time()
                final_points = Poincare_p2_0(Es[i], q_l, q1_r, p, false,  min_ics=min_ics_list[i]);
                final_points_list[i] = final_points
                println(time() - t)
            end
            function plot_Poinc()
               fig = figure(figsize=(25, 5), layout="constrained")
                gs = fig.add_gridspec(1,5)
                #gs = fig.add_gridspec(2,5)
                element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
                slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
                                
                for i in 1:n_E
                    ax = fig.add_subplot(element(0,i-1))
                    scat_plot = ax.scatter(final_points_list[i][:,1], final_points_list[i][:,2], s = 0.1, color="black")
                    #=
                    ax.set_ylim(-5,5)
                    yticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    if i == 1
                        ax.set_ylabel(L"q_2",fontsize=20)
                    else
                        yticks([])
                    end
                    #=
                    if i ==1
                        ax.text(.38, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")
                    else 
                        ax.text(.4, 0.95, L"γ = %$(ps[i])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                    end
                    =#
                    ax.set_xlim(-5.5,5.5)
                    xticks([-5,-2.5,0,2.5,5], fontsize = 15)
                    ax.set_xlabel(L"q_1",fontsize=20)
                    =#
                    title(L"E = %$(round(Es[i],digits=3))",fontsize=20)
                    
                end
                fig.suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = $((Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ))", fontsize=20);
    
                
            end
            plot_Poinc()
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Poincare/Poincare_p2_0_$(p)_E_$(Es).png")     
        #####
        
        


    ####

    
    #-----------------------FIG 1 - Coulpling and ESurface-------------------------------------------------
    
        function countor_energy_new(parameters, xx)
                roots, E_cl, λs, s_λ = crit_energies2(parameters,200,7)
                x = range(-xx, xx, length=1000);
                y = range(-xx, xx, length=1000);

                #Equivalent of meshgrid
                coordinates_x = repeat(x', length(x), 1);
                coordinates_y = repeat(y, 1, length(y));
                
                q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
                E_Contours = H_class([q1, p1, q2, p2],parameters) .- E_cl[1];

                Emin, Emax = 0, 150 #E_cl[1],10.
                CS = contourf(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 51), cmap="summer");
                #contour(coordinates_x, coordinates_y, E_Contours, range(Emin, -1, length = 8), colors="black"); # Only draw contour line for E = 0
                contour(coordinates_x, coordinates_y, E_Contours, range(Emin,Emax, length = 15), colors="black"); # Only draw contour line for E = 0
                custom_ticks = round.([Emin+5, (Emin+Emax)/2,Emax - 5 ],digits=1)
                #xlabel(L"q_1",fontsize=20)
                #ylabel(L"q_2",fontsize=20) #q1,q2

                cbar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
                cbar.ax.tick_params(axis="x", labelsize=15)
        end
        

        ps = [0.01, 1., 3., 5.]
        N_p = length(ps)
        θ1s = fill(NaN, N_p, 30)
        s_λs = fill("", N_p, 30)
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. # For γ
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1. # For ξ2
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1. # For Δ
        
        for j in 1:N_p
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, ps[j]    
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.,  1., 0., 2.8^2, 0., 10.3/10.4, 0., 2.5^2, ps[j]
            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            #parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
            #parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 

            roots, cE, λs, s_λ = crit_energies2(parameters,100,7);
            R = length(roots)
            θ1= zeros(R)
            #println(j)
            
            for i in 1:R
                θ1[i] = atan(roots[i][3], roots[i][1])
            end
            #sort(q1, by=real)'
            #sort(q11, by=real)' 
            sorted_indices = sortperm(θ1)
            roots, s_λ = roots[sorted_indices], s_λ[sorted_indices]
            
            for i in 1:R
                #println(j,i)
                θ1s[j, i] = atan(roots[i][3], roots[i][1])
                s_λs[j, i] = s_λ[i]         
            end 
        end
        function plot_γ()
            fig = figure(figsize=(20, 5),layout="constrained" )
            gs = fig.add_gridspec(1,4)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            for j in 1:N_p
                parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
                roots_, E_cl, λs, s_λ = crit_energies2(parameters,200,7);
                ax = fig.add_subplot(element(0,j-1))
                plots_rs = []
                for i in 1:length(roots_)
                    if s_λ[i] =="Saddle"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None",marker="X", markersize=7, color = "red", label="Unstable")
                    elseif s_λ[i] =="Saddle-focus"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="P", markersize=7, color = "darkred", label="Unstable-Stable")
                    else #center
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="o", markersize=7, color = "black", label="Stable")
                    end
                    push!(plots_rs, plot_r)
                end
                if j ==1
                    legend1 = legend(frameon=false, handles=plots_rs[[1]], loc="upper right", fontsize=15,scatterpoints=1)
                    ax.add_artist(legend1)
                    legend2 = legend(frameon=false, handles=plots_rs[[5,9]], loc="lower left", ncol=2, fontsize=15)
                end
                countor_energy_new(parameters, 7)
                if j == 1
                    yticks([-6,-3,0,3,6],fontsize=15)
                    ylabel(L"q_{2}", fontsize=20)
                else
                    yticks([])
                end
                xlabel(L"q_{1}", fontsize=20)
                xticks([-6,-3, 0,3, 6],fontsize=15)
                ax.text(.1, 0.95, L"γ = %$(ps[j])", transform=ax.transAxes, fontsize=20, verticalalignment="top")

                #ax.text(.5, 1.05, L"E", transform=ax.transAxes, fontsize=20)
            end






        end   
        plot_γ()
     
        savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points_stability_equivalent_γ2.png")
        #end
        #plot_γ()

        

        function plot_Δ()
            fig = figure(figsize=(20, 10), layout="constrained")
            gs = fig.add_gridspec(2,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            i=1
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue", label= "Saddle")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red", label= "Saddle-focus")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green", label= "Minimum")
            for i in 2:9
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green")
            end
            ax.set_xlabel(L"Δ",fontsize=12)
            ax.set_ylabel(L"θ",fontsize=12)
            legend(fontsize=10, shadow=true, loc = "upper right")
            
            j, k, l = 10, 75, 80
            plot(range(ps[j], ps[j], length = 10), range(-π, π, length = 10), "-", lw =2) 
            plot(range(ps[k], ps[k], length = 10), range(-π, π, length = 10), "-", lw =2 ) 
            plot(range(ps[l], ps[l], length = 10), range(-π, π, length = 10), "-", lw =2 ) 


            parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            ax = fig.add_subplot(element(0,1))
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7)
            title(L"Δ"*" = $(round(ps[j],digits=3))")
            
            
            parameters = ps[k], K1, ξ11, ξ21, ps[k], K2, ξ12, ξ22, γ 
            ax = fig.add_subplot(element(1,0))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 10)
            title(L"Δ"*" = $(round(ps[k],digits=3))")
            
            parameters = ps[l], K1, ξ11, ξ21, ps[l], K2, ξ12, ξ22, γ 
            ax = fig.add_subplot(element(1,1))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:9], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[10:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 10)
            title(L"Δ"*" = $(round(ps[l],digits=3))")
            
            suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(round.(parameters[1:8],digits=3))")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_equivalent_Δ.png")
        end
        plot_Δ()
    
        γ = 46.06
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. 
        parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
        xx= 13
        #function countor_energy2(parameters, xx, qq = true)
            x = range(-xx, xx, length=1000);
            y = range(-xx, xx, length=1000);

            #Equivalent of meshgrid
            coordinates_x = repeat(x', length(x), 1);
            coordinates_y = repeat(y, 1, length(y));
            q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2
            
            q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0;#q1,q2 
            
            E_Contours = H_class([q1, p1, q2, p2],parameters);
            E_Contours
            #roots_, E_cl, λs, s_λ = crit_energies2(parameters,1000, sqrt(10));
            
            Emin, Emax = 0 , 500
            CS = contourf(coordinates_x, coordinates_y, E_Contours, range( Emin, Emax , length = 100));
            contour(coordinates_x, coordinates_y, E_Contours, [0]);
            contour(coordinates_x, coordinates_y, E_Contours, range( Emin, Emax , length = 100),colors="black"); # Only draw contour line for E = 0
            #scatter(sqrt(γ),sqrt(γ))
            #scatter(-sqrt(γ),-sqrt(γ))
            xlabel(L"q_1",fontsize=12)
            ylabel(L"q_2",fontsize=12) #q1,q2
         
            cbar = colorbar(CS, label="E")


            plot_surface(coordinates_x, coordinates_y, E_Contours;
                       cmap="viridis", rstride=1, cstride=1, linewidth=0,
                       antialiased=true)
    ###############
        
    #-----------------------FIG 1 - Quantum Classical-------------------------------------------------
    

        ps = [0.01, 1., 3., 5.]
        N_p = length(ps)
        θ1s = fill(NaN, N_p, 30)
        s_λs = fill("", N_p, 30)
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. # For γ
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1. # For ξ2
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1. # For Δ
        
        for j in 1:N_p
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, ps[j]    
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.,  1., 0., 2.8^2, 0., 10.3/10.4, 0., 2.5^2, ps[j]
            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            #parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
            #parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 

            roots, cE, λs, s_λ = crit_energies2(parameters,100,7);
            R = length(roots)
            θ1= zeros(R)
            #println(j)
            
            for i in 1:R
                θ1[i] = atan(roots[i][3], roots[i][1])
            end
            #sort(q1, by=real)'
            #sort(q11, by=real)' 
            sorted_indices = sortperm(θ1)
            roots, s_λ = roots[sorted_indices], s_λ[sorted_indices]
            
            for i in 1:R
                #println(j,i)
                θ1s[j, i] = atan(roots[i][3], roots[i][1])
                s_λs[j, i] = s_λ[i]         
            end 
        end
        function plot_γ()
            
            fig = figure(figsize=(26, 13),layout="constrained" )
            font_s = 25
            gs = fig.add_gridspec(2,5, width_ratios=[1, 1, 1, 1, 0.25])
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            local cmap_ = "summer"
            local xx= x_lim = 7.09090909090909
            for j in 1:N_p
                parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
                roots_, E_cl, λs, s_λ = crit_energies2(parameters,200,7);
                ax = fig.add_subplot(element(0,j-1))
                plots_rs = []
                for i in 1:length(roots_)
                    if s_λ[i] =="Saddle"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None",marker="X", markersize=15, color = "red", label="Unstable")
                    elseif s_λ[i] =="Saddle-focus"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="P", markersize=15, color = "indianred", label="Unstable-Stable")
                    else #center
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="o", markersize=15, color = "black", label="Stable")
                    end
                    push!(plots_rs, plot_r)
                end
            
                x = range(-xx, xx, length=100);
                y = range(-xx, xx, length=100);

                #Equivalent of meshgrid
                coordinates_x = repeat(x', length(x), 1);
                coordinates_y = repeat(y, 1, length(y));
                
                q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
                E_Contours = H_class([q1, p1, q2, p2],parameters) .- E_cl[1];

                Emin, Emax = 0, -E_cl[1]+10 #E_cl[1],10.
                CS = contourf(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 101), cmap=cmap_)#, vmax = -E_cl[1]+10);
                #CS = contourf(coordinates_x, coordinates_y, E_Contours, cmap=cmap_, vmax = 0)
                #contour(coordinates_x, coordinates_y, E_Contours, range(Emin, -1, length = 8), colors="black"); # Only draw contour line for E = 0
                contour(coordinates_x, coordinates_y, E_Contours, range(Emin,-E_cl[1]+10, length = 9), colors="black"); # Only draw contour line for E = 0
                #custom_ticks = round.([Emin+5, (Emin+Emax)/2,Emax - 5 ],digits=1)
                #xlabel(L"q_1",fontsize=20)
                #ylabel(L"q_2",fontsize=20) #q1,q2
                custom_ticks = round.([Emin, (Emin+(-E_cl[1]+10))/2, -E_cl[1]+10],digits=1)
                cbar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
                cbar.ax.tick_params(axis="x", labelsize=font_s)
                cbar.ax.set_xlim(0, -E_cl[1]+10)

                 if j ==1
                    legend1 = legend(frameon=false, handles=plots_rs[[5]], loc="upper right",bbox_to_anchor=(1.05, 1.04), fontsize=font_s,scatterpoints=1)
                    ax.add_artist(legend1)
                    legend2 = legend(frameon=false, handles=plots_rs[[1,9]], loc="lower left",bbox_to_anchor=(-.05, -.04), ncol=2, fontsize=font_s)
                end
            

                #cbar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
                #cbar.ax.tick_params(axis="x", labelsize=15)
                if j == 1
                    yticks([-6,-3,0,3,6],fontsize=font_s)
                    ylabel(L"q_{2}", fontsize=font_s+5)
                else
                    yticks([])
                end
                xticks([])
                #xlim(-6,6)
                #ax.text(.01, 0.95, L"γ = %$(ps[j])", color="red",transform=ax.transAxes, fontsize=20, verticalalignment="top")


                ax = fig.add_subplot(element(1,j-1))
                N = 100;
                N_Q = 100
                IPR = zeros(N_Q,N_Q)
                for job in 1:10
                    IPR += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/PR_coh/IPR_p_$(parameters)_N_$(N)_N_Q_$(N_Q)_job_$(job).jld")["IPR_coh"];
                end
                PR = 1 ./ IPR
                q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
                PR_aux = (ones(118,118)*100)
                PR_aux[10:109,10:109] = PR
                cmap = matplotlib.cm.get_cmap("cividis").copy()
                cmap.set_over("white")
                im = imshow(PR_aux,origin="lower",cmap=cmap, extent=(-7.09090909090909,7.09090909090909, -7.09090909090909,7.09090909090909), vmax = 32)
                
                if j == 1
                    ylabel(L"q_2", fontsize=font_s+5)
                    yticks([-6,-3,0,3,6],fontsize=font_s)
                else
                    yticks([])
                end
               
                for i in 1:length(roots_)
                    if s_λ[i] =="Saddle"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None",marker="X", markersize=15, color = "red", label="Unstable")
                    elseif s_λ[i] =="Saddle-focus"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="P", markersize=15, color = "indianred", label="Unstable-Stable")
                    else #center
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="o", markersize=15, color = "black", label="Stable")
                    end
                    push!(plots_rs, plot_r)
                end
                contour(coordinates_x, coordinates_y, E_Contours, range(Emin,-E_cl[1]+10, length = 9), colors="black"); # Only draw contour line for E = 0
                
                xlabel(L"q_1", fontsize=font_s+5)
                xticks([-6,-3,0,3,6],fontsize=font_s)
                xlim(-7.09090909090909,7.09090909090909)
                
                #cbar = colorbar(im,label="PR")

                if j == N_p
                    #[x, y, width, height]
                    cax_ = fig.add_axes([0.95, 0.06, 0.01, 0.4]) 
                    custom_ticks = [0.0, 10.,20., 30.]
                    cbar = plt.colorbar(im, ticks=custom_ticks,cax = cax_)
                    cbar.ax.tick_params(labelsize=font_s)
                end

                #ax.text(.5, 1.05, L"E", transform=ax.transAxes, fontsize=20)
            end 
        end   
        plot_γ()
    
        #savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points_stability_equivalent_γ_3.png")
        savefig("C:/Users/edson/Desktop/ixed_points_stability_equivalent_γ_3.png")
        #end
        #plot_γ()

        

        function plot_Δ()
            fig = figure(figsize=(20, 10), layout="constrained")
            gs = fig.add_gridspec(2,2)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)

            ax = fig.add_subplot(element(0,0))
            i=1
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue", label= "Saddle")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red", label= "Saddle-focus")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green", label= "Minimum")
            for i in 2:9
                Saddle_idx = findall(==("Saddle"), s_λs[:,i])
                SaddleF_idx = findall(==("Saddle-focus"), s_λs[:,i])
                Center_idx = findall(==("Center"), s_λs[:,i])
                scatter(ps[Saddle_idx], θ1s[:,i][Saddle_idx], color="blue")
                scatter(ps[SaddleF_idx], θ1s[:,i][SaddleF_idx], color="red")
                scatter(ps[Center_idx], θ1s[:,i][Center_idx], color="green")
            end
            ax.set_xlabel(L"Δ",fontsize=12)
            ax.set_ylabel(L"θ",fontsize=12)
            legend(fontsize=10, shadow=true, loc = "upper right")
            
            j, k, l = 10, 75, 80
            plot(range(ps[j], ps[j], length = 10), range(-π, π, length = 10), "-", lw =2) 
            plot(range(ps[k], ps[k], length = 10), range(-π, π, length = 10), "-", lw =2 ) 
            plot(range(ps[l], ps[l], length = 10), range(-π, π, length = 10), "-", lw =2 ) 


            parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            ax = fig.add_subplot(element(0,1))
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 7)
            title(L"Δ"*" = $(round(ps[j],digits=3))")
            
            
            parameters = ps[k], K1, ξ11, ξ21, ps[k], K2, ξ12, ξ22, γ 
            ax = fig.add_subplot(element(1,0))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:6], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[7:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 10)
            title(L"Δ"*" = $(round(ps[k],digits=3))")
            
            parameters = ps[l], K1, ξ11, ξ21, ps[l], K2, ξ12, ξ22, γ 
            ax = fig.add_subplot(element(1,1))
            roots, E_cl, λs, s_λ = crit_energies(parameters);
            plots_rs = []
            for i in 1:length(roots)
                plot_r, = plot(roots[i][1], roots[i][3], marker="o", markersize=7, label="E = $(round(E_cl[i],digits=2)), θ = $(round(atan(roots[i][3], roots[i][1]),digits=2))")
                push!(plots_rs, plot_r)
            end
            plots_rs
            legend1 = legend(handles=plots_rs[1:9], loc="upper center", bbox_to_anchor=(0.5, 1.), ncol=3, fontsize=10)
            ax.add_artist(legend1)
            legend2 = legend(handles=plots_rs[10:length(roots)], loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, fontsize=10)
            countor_energy(parameters, 10)
            title(L"Δ"*" = $(round(ps[l],digits=3))")
            
            suptitle("Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = $(round.(parameters[1:8],digits=3))")
            savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Fixed_points/Fixed_points_stability_equivalent_Δ.png")
        end
        plot_Δ()
    
        γ = 46.06
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. 
        parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ
        xx= 13
        #function countor_energy2(parameters, xx, qq = true)
            x = range(-xx, xx, length=1000);
            y = range(-xx, xx, length=1000);

            #Equivalent of meshgrid
            coordinates_x = repeat(x', length(x), 1);
            coordinates_y = repeat(y, 1, length(y));
            q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2
            
            q1, p1, q2, p2 = coordinates_x,coordinates_y,0,0;#q1,q2 
            
            E_Contours = H_class([q1, p1, q2, p2],parameters);
            E_Contours
            #roots_, E_cl, λs, s_λ = crit_energies2(parameters,1000, sqrt(10));
            
            Emin, Emax = 0 , 500
            CS = contourf(coordinates_x, coordinates_y, E_Contours, range( Emin, Emax , length = 100));
            contour(coordinates_x, coordinates_y, E_Contours, [0]);
            contour(coordinates_x, coordinates_y, E_Contours, range( Emin, Emax , length = 100),colors="black"); # Only draw contour line for E = 0
            #scatter(sqrt(γ),sqrt(γ))
            #scatter(-sqrt(γ),-sqrt(γ))
            xlabel(L"q_1",fontsize=12)
            ylabel(L"q_2",fontsize=12) #q1,q2
         
            cbar = colorbar(CS, label="E")


            plot_surface(coordinates_x, coordinates_y, E_Contours;
                       cmap="viridis", rstride=1, cstride=1, linewidth=0,
                       antialiased=true)
    ###############

    ### ------------------------- Lyapunov map ------------------------------------------
        ps = [0.01, 1., 3., 5.]
        N_p = length(ps)
        θ1s = fill(NaN, N_p, 30)
        s_λs = fill("", N_p, 30)
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22 = 0., 1., 0., 5., 0., 1., 0., 5. # For γ
        #Δ1, K1, ξ11, Δ2, K2, ξ12, γ = 0., 1., 0., 0., 1., 0., 1. # For ξ2
        #K1, ξ11, ξ21, K2, ξ12, ξ22, γ = 1., 0., 5., 1., 0., 5., 1. # For Δ
        
        for j in 1:N_p
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, 0.04068266906056494, 1., 2.4030828241198847, 3.571424199455252, ps[j]    
            #Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = parameters = 0.,  1., 0., 2.8^2, 0., 10.3/10.4, 0., 2.5^2, ps[j]
            parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
            #parameters = Δ1, K1, ξ11, ps[j], Δ2, K2, ξ12, ps[j], γ
            #parameters = ps[j], K1, ξ11, ξ21, ps[j], K2, ξ12, ξ22, γ 

            roots, cE, λs, s_λ = crit_energies2(parameters,100,7);
            R = length(roots)
            θ1= zeros(R)
            #println(j)
            
            for i in 1:R
                θ1[i] = atan(roots[i][3], roots[i][1])
            end
            #sort(q1, by=real)'
            #sort(q11, by=real)' 
            sorted_indices = sortperm(θ1)
            roots, s_λ = roots[sorted_indices], s_λ[sorted_indices]
            
            for i in 1:R
                #println(j,i)
                θ1s[j, i] = atan(roots[i][3], roots[i][1])
                s_λs[j, i] = s_λ[i]         
            end 
        end

        function plot_γ()
            
            fig = figure(figsize=(20, 11),layout="constrained" )
            gs = fig.add_gridspec(2,4)
            element(i,j) = get(gs, (i,j)) # starts at 0 to N-1
            slice(i,j) = pycall(pybuiltin("slice"), PyObject, i,j)
            local cmap_ = "BuPu_r"
            local xx= x_lim = 6
            for j in 1:N_p
                p = parameters = Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, ps[j] 
                roots_, E_cl, λs, s_λ = crit_energies2(parameters,200,7);
                ax = fig.add_subplot(element(0,j-1))
                plots_rs = []
                for i in 1:length(roots_)
                    if s_λ[i] =="Saddle"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None",marker="X", markersize=7, color = "red", label="Unstable")
                    elseif s_λ[i] =="Saddle-focus"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="P", markersize=7, color = "darkred", label="Unstable-Stable")
                    else #center
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="o", markersize=7, color = "black", label="Stable")
                    end
                    push!(plots_rs, plot_r)
                end
                N_λ = 100
                λs = zeros(N_λ,N_λ)
                for job in 1:5
                    λs += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_map/Lyapunov_map_$(p)_job_$(job).jld")["λs"];
                end
                q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_λ),range(-x_lim,x_lim, length=N_λ),range(-x_lim,x_lim, length=N_λ),range(-x_lim,x_lim, length=N_λ)
                
                im = imshow(λs,origin="lower",cmap=cmap_,extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]),vmax = 1)
                x = range(-xx, xx, length=1000);
                y = range(-xx, xx, length=1000);
                coordinates_x = repeat(x', length(x), 1);
                coordinates_y = repeat(y, 1, length(y));
                q1, p1, q2, p2 = coordinates_x,0, coordinates_y,0;#q1,q2 
                E_Contours = H_class([q1, p1, q2, p2],p);
                roots, E_cl, λs, s_λ = crit_energies(p,7)
                Emin, Emax = E_cl[1],10.
                contour(coordinates_x, coordinates_y, E_Contours, range(Emin, Emax, length = 5), colors="black"); # Only draw contour line for E = 0
                #title(" Δ1, ξ21, γ = $(p[[1,4,9]])", fontsize=20)
                #xlabel(L"q_1", fontsize=20)
                #ylabel(L"q_2", fontsize=20)
                if j==N_p
                    cbar = colorbar(im)
                    cbar.set_label("λ", fontsize=20)
                end
                #=
                if j ==1
                    legend1 = legend(frameon=false, handles=plots_rs[[1]], loc="upper right", fontsize=15,scatterpoints=1)
                    ax.add_artist(legend1)
                    legend2 = legend(frameon=false, handles=plots_rs[[5,9]], loc="lower left", ncol=2, fontsize=15)
                end
                =#

                #cbar = colorbar(CS,location="top", ticks=custom_ticks, shrink=0.9)
                #cbar.ax.tick_params(axis="x", labelsize=15)
                if j == 1
                    yticks([-6,-3,0,3,6],fontsize=15)
                    ylabel(L"q_{2}", fontsize=20)
                else
                    yticks([])
                end
                xticks([])
                xlim(-6,6)
                ax.text(.1, 0.95, L"γ = %$(ps[j])", color="red",transform=ax.transAxes, fontsize=20, verticalalignment="top")


                ax = fig.add_subplot(element(1,j-1))
                N = 100;
                N_Q = 100
                IPR = zeros(N_Q,N_Q)
                for job in 1:10
                    IPR += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/data/PR_coh/IPR_p_$(parameters)_N_$(N)_N_Q_$(N_Q)_job_$(job).jld")["IPR_coh"];
                end
                PR = 1 ./ IPR 
                q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q),range(-x_lim,x_lim, length=N_Q)
                im = imshow(PR,origin="lower",cmap=cmap_,extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]),vmax = 20)
                
                if j == 1
                    ylabel(L"q_2", fontsize=20)
                    yticks([-6,-3,0,3,6],fontsize=15)
                else
                    yticks([])
                end
                for i in 1:length(roots_)
                    if s_λ[i] =="Saddle"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None",marker="X", markersize=7, color = "red", label="Unstable")
                    elseif s_λ[i] =="Saddle-focus"
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="P", markersize=7, color = "darkred", label="Unstable-Stable")
                    else #center
                        plot_r, = plot(roots_[i][1], roots_[i][3], linestyle="None", marker="o", markersize=7, color = "black", label="Stable")
                    end
                    push!(plots_rs, plot_r)
                end
                contour(coordinates_x, coordinates_y, E_Contours, range(Emin,Emax, length = 8), colors="black"); # Only draw contour line for E = 0
                
                xlabel(L"q_1", fontsize=20)
                xticks([-6,-3,0,3,6],fontsize=15)
                xlim(-6,6)
                
                if j ==N_p
                    cbar = colorbar(im)
                    cbar.set_label("PR", fontsize=20)
                end


                #ax.text(.5, 1.05, L"E", transform=ax.transAxes, fontsize=20)
            end 
        end   
        plot_γ()

         savefig("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/Figures/Lyapunov_PR_quantumclassical_γ.png")
        close()


        ###Reading Cluster
        Δ1, K1, ξ11, ξ21, Δ2, K2, ξ12, ξ22, γ = p = (0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 5.0, 1.0);
        
        N_λ = 100
        λs = zeros(N_λ,N_λ)
        for job in 1:5
            λs += load("C:/Users/edson/Desktop/Research/Kerr_system/Coupled_kerr/codes_Chemistry/Classical_Kerr/data/Lyapunov_map/Lyapunov_map_$(p)_job_$(job).jld")["λs"];
        end

        xx= x_lim = 6
        q1vals, p1vals, q2vals, p2vals = range(-x_lim,x_lim, length=N_λ),range(-x_lim,x_lim, length=N_λ),range(-x_lim,x_lim, length=N_λ),range(-x_lim,x_lim, length=N_λ)
        
        im = imshow(λs,origin="lower",cmap="summer",extent=(q1vals[1],q1vals[end], q2vals[1],q2vals[end]))
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