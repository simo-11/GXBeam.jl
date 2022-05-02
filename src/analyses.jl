"""
    static_analysis(assembly; kwargs...)

Perform a static analysis of the system of nonlinear beams contained in
`assembly`. Return the resulting system and a flag indicating whether the
iteration procedure converged.

# Keyword Arguments
 - `prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}()`:
        A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and values of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads = Dict{Int,DistributedLoads{Float64}}()`: A dictionary
        with keys corresponding to the elements to which distributed loads are
        applied and values of type [`DistributedLoads`](@ref) which describe
        the distributed loads on those elements.  If time varying, this input may
        be provided as a function of time.
 - `point_masses = Dict{Int,PointMass{Float64}}()`: A dictionary with keys 
        corresponding to the points to which point masses are attached and values 
        of type [`PointMass`](@ref) which contain the properties of the attached 
        point masses.  If time varying, this input may be provided as a function of time.
 - `gravity = [0,0,0]`: Gravity vector.  If time varying, this input may be provided as a 
        function of time.
 - `linear = false`: Set to `true` for a linear analysis
 - `linearization_state`: Linearization state variables.  Defaults to zeros.
 - `update_linearization_state`: Flag indicating whether to update the linearization state 
    variables for a linear analysis with the instantaneous state variables.
 - `method = :newton`: Method (as defined in NLsolve) to solve nonlinear system of equations
 - `linesearch = LineSearches.BackTracking(maxstep=1e6)`: Line search used to solve nonlinear 
        system of equations
 - `ftol = 1e-9`: tolerance for solving nonlinear system of equations
 - `iterations = 1000`: maximum iterations for solving the nonlinear system of equations
 - `tvec = 0`: Time vector/value. May be used in conjunction with time varying
        prescribed conditions and distributed loads to gradually increase
        displacements/loads.
 - `show_trace = false`: Flag indicating whether to show solution progress
 - `reset_state = true`: Flag indicating whether the state variables should be
        reset prior to performing the analysis.  This keyword argument is only valid
        for the pre-allocated version of this function.
"""
function static_analysis(assembly; kwargs...)

    system = System(assembly)

    return static_analysis!(system, assembly; kwargs..., reset_state=false)
end

"""
    static_analysis!(system, assembly; kwargs...)

Pre-allocated version of [`static_analysis`](@ref).
"""
function static_analysis!(system, assembly;
    prescribed_conditions=Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads=Dict{Int,DistributedLoads{Float64}}(),
    point_masses=Dict{Int,PointMass{Float64}}(),
    gravity=SVector(0,0,0),
    linear=false,
    linearization_state=nothing,
    update_linearization_state=false,
    method=:newton,
    linesearch=LineSearches.BackTracking(maxstep=1e6),
    ftol=1e-9,
    iterations=250,
    tvec=0.0,
    show_trace=false,
    reset_state=true)

    # reset state, if specified
    if reset_state
        reset_state!(system)
    end

    # unpack pre-allocated storage and pointers
    @unpack x, r, K, force_scaling, static_indices = system

    # use only a portion of the pre-allocated variables
    xs = get_static_state(system, x)
    rs = similar(r, length(xs))
    Ks = similar(K, length(xs), length(xs))

    # assume converged until proven otherwise
    converged = true
    
    # begin time stepping
    for t in tvec

        if show_trace
            println("Solving for t=$t")
        end

        # update stored time
        system.t = t

        # current parameters
        pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)
        dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(t)
        pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
        gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(tvec[it]))

        # solve the system of equations
        f! = (resid, x) -> static_system_residual!(resid, x, static_indices, force_scaling, 
            assembly, pcond, dload, pmass, gvec)

        j! = (jacob, x) -> static_system_jacobian!(jacob, x, static_indices, force_scaling, 
            assembly, pcond, dload, pmass, gvec)

        if linear
            # linear analysis
            if !update_linearization_state
                if isnothing(linearization_state)
                    xs .= 0
                else
                    xs .= linearization_state
                end
            end
            f!(rs, xs)
            j!(Ks, xs)

            # update the solution
            xs .-= safe_lu(Ks) \ rs

            # update convergence flag
            converged = true
        else
            # nonlinear analysis
            df = NLsolve.OnceDifferentiable(f!, j!, xs, rs, Ks)

            result = NLsolve.nlsolve(df, xs,
                show_trace=show_trace,
                linsolve=(x, A, b) -> ldiv!(x, safe_lu(A), b),
                method=method,
                linesearch=linesearch,
                ftol=ftol,
                iterations=iterations)

            # update the solution
            xs .= result.zero
            Ks .= df.DF

            # update convergence flag
            converged = result.f_converged
        end
    end

    # copy the static state vector into the dynamic state vector
    set_static_state!(system, xs)

    return system, converged
end

"""
    steady_state_analysis(assembly; kwargs...)

Perform a steady-state analysis for the system of nonlinear beams contained in
`assembly`.  Return the resulting system and a flag indicating whether the
iteration procedure converged.

# Keyword Arguments
 - `prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}()`:
        A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and values of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads = Dict{Int,DistributedLoads{Float64}}()`: A dictionary
        with keys corresponding to the elements to which distributed loads are
        applied and values of type [`DistributedLoads`](@ref) which describe
        the distributed loads on those elements.  If time varying, this input may
        be provided as a function of time.
 - `point_masses = Dict{Int,PointMass{Float64}}()`: A dictionary with keys 
        corresponding to the points to which point masses are attached and values 
        of type [`PointMass`](@ref) which contain the properties of the attached 
        point masses.  If time varying, this input may be provided as a function of time.
 - `gravity = [0,0,0]`: Gravity vector.  If time varying, this input may be provided as a 
        function of time.            
 - `linear = false`: Set to `true` for a linear analysis
 - `linearization_state`: Linearization state variables.  Defaults to zeros.
 - `update_linearization_state`: Flag indicating whether to update the linearization state 
    variables for a linear analysis with the current state variables.
 - `method = :newton`: Method (as defined in NLsolve) to solve nonlinear system of equations
 - `linesearch = LineSearches.LineSearches.BackTracking(maxstep=1e6)`: Line search used to 
    solve the nonlinear system of equations
 - `ftol = 1e-9`: tolerance for solving the nonlinear system of equations
 - `iterations = 1000`: maximum iterations for solving the nonlinear system of equations
 - `origin = zeros(3)`: Global frame origin vector. If time varying, this input
    may be provided as a function of time.
 - `linear_velocity = zeros(3)`: Global frame linear velocity vector. If time
    varying, this vector may be provided as a function of time.
 - `angular_velocity = zeros(3)`: Global frame angular velocity vector. If time
    varying, this vector may be provided as a function of time.
 - `linear_acceleration = zeros(3)`: Global frame linear acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `angular_acceleration = zeros(3)`: Global frame angular acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `tvec = 0.0`: Time vector/value. May be used in conjunction with time varying
    prescribed conditions, distributed loads, and global motion to gradually
    increase displacements/loads.
 - `show_trace = false`: Flag indicating whether to show solution progress
 - `reset_state = true`: Flag indicating whether the state variables should be
    reset prior to performing the analysis.  This keyword argument is only valid
    for the pre-allocated version of this function.
"""
function steady_state_analysis(assembly; kwargs...)

    system = System(assembly)

    return steady_state_analysis!(system, assembly; kwargs..., reset_state=true)
end

"""
    steady_state_analysis!(system, assembly; kwargs...)

Pre-allocated version of [`steady_state_analysis`](@ref).
"""
function steady_state_analysis!(system, assembly;
    prescribed_conditions=Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads=Dict{Int,DistributedLoads{Float64}}(),
    point_masses=Dict{Int,PointMass{Float64}}(),
    gravity=SVector(0,0,0),
    linear=false,
    linearization_state=nothing,
    update_linearization_state=false,
    method=:newton,
    linesearch=LineSearches.BackTracking(maxstep=1e6),
    ftol=1e-9,
    iterations=250,
    origin=(@SVector zeros(3)),
    linear_velocity=(@SVector zeros(3)),
    angular_velocity=(@SVector zeros(3)),
    linear_acceleration=(@SVector zeros(3)),
    angular_acceleration=(@SVector zeros(3)),
    tvec=0.0,
    show_trace=false,
    reset_state=true,
    expanded=false,
    structural_damping=false,
    )

    # reset state, if specified
    if reset_state
        reset_state!(system)
    end

    # unpack pre-allocated storage
    @unpack x, r, K, force_scaling, dynamic_indices, expanded_indices = system

    # construct expanded state vector (if necessary)
    if expanded
        x = get_expanded_state(system, assembly, prescribed_conditions)
        r = similar(r, length(x))
        K = similar(K, length(x), length(x))
    end

    # assume converged until proven otherwise
    converged = true

    # begin time stepping
    for t in tvec

        if show_trace
            println("Solving for t=$t")
        end

        # update stored time
        system.t = t

        # current parameters
        pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)
        dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(t)
        pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
        gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(tvec[it]))
        x0 = typeof(origin) <: AbstractVector ? SVector{3}(origin) : SVector{3}(origin(t))
        v0 = typeof(linear_velocity) <: AbstractVector ? SVector{3}(linear_velocity) : SVector{3}(linear_velocity(t))
        ω0 = typeof(angular_velocity) <: AbstractVector ? SVector{3}(angular_velocity) : SVector{3}(angular_velocity(t))
        a0 = typeof(linear_acceleration) <: AbstractVector ? SVector{3}(linear_acceleration) : SVector{3}(linear_acceleration(t))
        α0 = typeof(angular_acceleration) <: AbstractVector ? SVector{3}(angular_acceleration) : SVector{3}(angular_acceleration(t))

        # residual and jacobian function
        if expanded
            f! = (resid, x) -> expanded_system_residual!(resid, x, expanded_indices, force_scaling, 
                structural_damping, assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0)

            j! = (jacob, x) -> expanded_system_jacobian!(jacob, x, expanded_indices, force_scaling, 
                structural_damping, assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0)
        else
            f! = (resid, x) -> steady_state_system_residual!(resid, x, dynamic_indices, force_scaling, 
                structural_damping, assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0)

            j! = (jacob, x) -> steady_state_system_jacobian!(jacob, x, dynamic_indices, force_scaling, 
                structural_damping, assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0)
        end

        # solve the system of equations
        if linear
            # set up a linear analysis
            if !update_linearization_state
                if isnothing(linearization_state)
                    x .= 0
                else
                    x .= linearization_state
                end
            end
            f!(r, x)
            j!(K, x)

            # update the solution               
            x .-= safe_lu(K) \ r

            # update the convergence flag
            converged = true
        else
            # nonlinear analysis
            df = NLsolve.OnceDifferentiable(f!, j!, x, r, K)

            result = NLsolve.nlsolve(df, x,
                show_trace=show_trace,
                linsolve=(x, A, b) -> ldiv!(x, safe_lu(A), b),
                method=method,
                linesearch=linesearch,
                ftol=ftol,
                iterations=iterations)

            # update the solution
            x .= result.zero
            K .= df.DF

            # update the convergence flag
            converged = result.f_converged
        end
    end

    # insert expanded state variables into system state vector
    if expanded
        set_expanded_state!(system, assembly, prescribed_conditions, x)
    end

    return system, converged
end

"""
    eigenvalue_analysis(assembly; kwargs...)

Compute the eigenvalues and eigenvectors of the system of nonlinear beams
contained in `assembly`.  Return the modified system, eigenvalues, eigenvectors,
and a convergence flag indicating whether the corresponding steady-state analysis
converged.

# Keyword Arguments
 - `prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}()`:
        A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and values of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads = Dict{Int,DistributedLoads{Float64}}()`: A dictionary
        with keys corresponding to the elements to which distributed loads are
        applied and values of type [`DistributedLoads`](@ref) which describe
        the distributed loads on those elements.  If time varying, this input may
        be provided as a function of time.
 - `point_masses = Dict{Int,PointMass{Float64}}()`: A dictionary with keys 
        corresponding to the points to which point masses are attached and values 
        of type [`PointMass`](@ref) which contain the properties of the attached 
        point masses.  If time varying, this input may be provided as a function of time.
 - `structural_damping=false`: Flag indicating whether structural damping should be enabled
 - `gravity = [0,0,0]`: Gravity vector.  If time varying, this input may be provided as a 
        function of time.            
 - `linear = false`: Set to `true` for a linear analysis
 - `linearization_state`: Linearization state variables.  Defaults to zeros.
 - `update_linearization_state`: Flag indicating whether to update the linearization state 
    variables for a linear analysis with the current state variables.
 - `method = :newton`: Method (as defined in NLsolve) to solve nonlinear system of equations
 - `linesearch = LineSearches.LineSearches.BackTracking(maxstep=1e6)`: Line search used to 
    solve nonlinear system of equations
 - `ftol = 1e-9`: tolerance for solving the nonlinear system of equations
 - `iterations = 1000`: maximum iterations for solving the nonlinear system of equations
 - `show_trace = false`: Flag indicating whether to show solution progress
 - `reset_state = true`: Flag indicating whether the state variables should be
    reset prior to performing the steady-state analysis.  This keyword argument
    is only valid for the pre-allocated version of this function.
 - `find_steady_state = reset_state && !linear`: Flag indicating whether the
    steady state solution should be found prior to performing the eigenvalue analysis.
 - `origin = zeros(3)`: Global frame origin.
    If time varying, this vector may be provided as a function of time.
 - `linear_velocity = zeros(3)`: Global frame linear velocity vector.
    If time varying, this vector may be provided as a function of time.
 - `angular_velocity = zeros(3)`: Global frame angular velocity vector.
    If time varying, this vector may be provided as a function of time.
 - `linear_acceleration = zeros(3)`: Global frame linear acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `angular_acceleration = zeros(3)`: Global frame angular acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `tvec`: Time vector. May be used in conjunction with time varying
    prescribed conditions, distributed loads, and global motion to gradually
    increase displacements/loads during the steady-state analysis.
 - `nev = 6`: Number of eigenvalues to compute
"""
function eigenvalue_analysis(assembly; kwargs...)

    system = System(assembly)

    return eigenvalue_analysis!(system, assembly; kwargs..., reset_state=true)
end

"""
    eigenvalue_analysis!(system, assembly; kwargs...)

Pre-allocated version of `eigenvalue_analysis`.  Uses the state variables stored in
`system` as an initial guess for iterating to find the steady state solution.
"""
function eigenvalue_analysis!(system, assembly;
    prescribed_conditions=Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads=Dict{Int,DistributedLoads{Float64}}(),
    point_masses=Dict{Int,PointMass{Float64}}(),
    structural_damping=false,
    gravity=SVector(0,0,0),
    expanded=false,
    linear=false,
    linearization_state=nothing,
    update_linearization_state=false,
    method=:newton,
    linesearch=LineSearches.BackTracking(maxstep=1e6),
    ftol=1e-9,
    iterations=250,
    show_trace=false,
    reset_state=true,
    find_steady_state=!linear && reset_state,
    origin=(@SVector zeros(3)),
    linear_velocity=(@SVector zeros(3)),
    angular_velocity=(@SVector zeros(3)),
    linear_acceleration=(@SVector zeros(3)),
    angular_acceleration=(@SVector zeros(3)),
    tvec=0.0,
    nev=6,
    )

    if reset_state
        reset_state!(system)
    end

    # perform steady state analysis (if nonlinear)
    if find_steady_state
        if show_trace
            println("Finding a Steady-State Solution")
        end

        system, converged = steady_state_analysis!(system, assembly;
            prescribed_conditions=prescribed_conditions,
            distributed_loads=distributed_loads,
            point_masses=point_masses,
            gravity=gravity,
            expanded=expanded,
            linear=linear,
            linearization_state=linearization_state,
            update_linearization_state=update_linearization_state,
            method=method,
            linesearch=linesearch,
            ftol=ftol,
            iterations=iterations,
            origin=origin,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            angular_acceleration=angular_acceleration,
            tvec=tvec,
            show_trace=show_trace,
            reset_state=reset_state,
            )
    else
        # set linearization state variables
        if linear && !update_linearization_state
            if isnothing(linearization_state)
                system.x .= 0
            else
                system.x .= linearization_state
            end
        end
        # converged by default
        converged = true
    end

    if show_trace
        println("Solving Eigensystem")
    end

    # unpack state vector, stiffness, and mass matrices
    @unpack x, K, M, force_scaling, dynamic_indices, expanded_indices, t = system

    # current parameters
    pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)
    dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(t)
    pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
    gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(tvec[it]))
    x0 = typeof(origin) <: AbstractVector ? SVector{3}(origin) : SVector{3}(origin(t))
    v0 = typeof(linear_velocity) <: AbstractVector ? SVector{3}(linear_velocity) : SVector{3}(linear_velocity(t))
    ω0 = typeof(angular_velocity) <: AbstractVector ? SVector{3}(angular_velocity) : SVector{3}(angular_velocity(t))
    a0 = typeof(linear_acceleration) <: AbstractVector ? SVector{3}(linear_acceleration) : SVector{3}(linear_acceleration(t))
    α0 = typeof(angular_acceleration) <: AbstractVector ? SVector{3}(angular_acceleration) : SVector{3}(angular_acceleration(t))

    if expanded

        # construct expanded state vector
        x = get_expanded_state(system, assembly, prescribed_conditions)
        K = similar(K, length(x), length(x))
        M = similar(K, length(x), length(x))

        # solve for the system stiffness matrix
        expanded_system_jacobian!(K, x, dynamic_indices, force_scaling, structural_damping,
            assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0)

        # solve for the system mass matrix
        expanded_system_mass_matrix!(M, dynamic_indices, force_scaling, assembly, 
            prescribed_conditions, point_masses)

    else

        # solve for the system stiffness matrix
        steady_state_system_jacobian!(K, x, dynamic_indices, force_scaling, structural_damping,
            assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0)

        # solve for the system mass matrix
        system_mass_matrix!(M, x, dynamic_indices, force_scaling, 
            assembly, prescribed_conditions, point_masses)

    end

    # construct linear map
    T = eltype(system)
    nx = length(x)
    Kfact = safe_lu(K)
    f! = (b, x) -> ldiv!(b, Kfact, M * x)
    fc! = (b, x) -> mul!(b, M', Kfact' \ x)
    A = LinearMap{T}(f!, fc!, nx, nx; ismutating=true)

    # compute eigenvalues and eigenvectors
    λ, V = partialeigen(partialschur(A; nev=min(nx, nev), which=LM())[1])

    # sort eigenvalues by magnitude
    perm = sortperm(λ, by=(λ) -> (abs(λ), imag(λ)), rev=true)
    λ .= λ[perm]
    V .= V[:,perm]

    # eigenvalues are actually -1/λ, no modification necessary for eigenvectors
    λ .= -1 ./ λ

    return system, λ, V, converged
end

"""
    initial_condition_analysis(assembly, t0; kwargs...)

Perform an analysis to obtain a consistent set of initial conditions.  Return the
final system with the new initial conditions.

# Keyword Arguments
 - `prescribed_conditions: A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and values of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads = Dict{Int,DistributedLoads{Float64}}()`: A dictionary
        with keys corresponding to the elements to which distributed loads are
        applied and values of type [`DistributedLoads`](@ref) which describe
        the distributed loads on those elements.  If time varying, this input may
        be provided as a function of time.
 - `point_masses = Dict{Int,PointMass{Float64}}()`: A dictionary with keys 
        corresponding to the points to which point masses are attached and values 
        of type [`PointMass`](@ref) which contain the properties of the attached 
        point masses.  If time varying, this input may be provided as a function of time.
 - `structural_damping=true`: Flag indicating whether structural damping should be enabled
 - `gravity = [0,0,0]`: Gravity vector.  If time varying, this input may be provided as a 
        function of time.
 - `linear = false`: Set to `true` for a linear analysis
 - `linearization_state`: Linearization state variables.  Defaults to zeros.
 - `method = :newton`: Method (as defined in NLsolve) to solve nonlinear system of equations
 - `linesearch = LineSearches.LineSearches.BackTracking(maxstep=1e6)`: Line search used to solve nonlinear system of equations
 - `ftol = 1e-9`: tolerance for solving nonlinear system of equations
 - `iterations = 1000`: maximum iterations for solving the nonlinear system of equations
 - `show_trace = false`: Flag indicating whether to show solution progress
 - `reset_state = true`: Flag indicating whether the state variables should be
    reset prior to performing the analysis.  This keyword argument is only valid
    for the pre-allocated version of this function.
 - `origin = zeros(3)`: Global frame origin.
    If time varying, this vector may be provided as a function of time.
 - `linear_velocity = zeros(3)`: Global frame linear velocity vector.
    If time varying, this vector may be provided as a function of time.
 - `angular_velocity = zeros(3)`: Global frame angular velocity vector.
    If time varying, this vector may be provided as a function of time.
 - `linear_acceleration = zeros(3)`: Global frame linear acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `angular_acceleration = zeros(3)`: Global frame angular acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `u0=fill(zeros(3), length(assembly.points))`: Initial linear displacement of each point
 - `theta0=fill(zeros(3), length(assembly.points))`: Initial angular displacement of each point
 - `udot0=fill(zeros(3), length(assembly.points))`: Initial linear displacement rate of each point
 - `thetadot0=fill(zeros(3), length(assembly.points))`: Initial angular displacement rate of each point
"""
function initial_condition_analysis(assembly, t0; kwargs...)

    system = System(assembly)

    return initial_condition_analysis!(system, assembly, t0; kwargs...)
end

"""
    initial_condition_analysis!(system, assembly, t0; kwargs...)

Pre-allocated version of `initial_condition_analysis`.
"""
function initial_condition_analysis!(system, assembly, t0;
    prescribed_conditions=Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads=Dict{Int,DistributedLoads{Float64}}(),
    point_masses=Dict{Int,PointMass{Float64}}(),
    structural_damping=true,
    gravity=SVector(0,0,0),
    linear=false,
    linearization_state=nothing,
    method=:newton,
    linesearch=LineSearches.BackTracking(maxstep=1e6),
    ftol=1e-9,
    iterations=250,
    show_trace=false,
    reset_state=true,
    origin=(@SVector zeros(3)),
    linear_velocity=(@SVector zeros(3)),
    angular_velocity=(@SVector zeros(3)),
    linear_acceleration=(@SVector zeros(3)),
    angular_acceleration=(@SVector zeros(3)),
    u0=fill((@SVector zeros(3)), length(assembly.points)),
    theta0=fill((@SVector zeros(3)), length(assembly.points)),
    udot0=fill((@SVector zeros(3)), length(assembly.points)),
    thetadot0=fill((@SVector zeros(3)), length(assembly.points)),
    )

    if reset_state
        reset_state!(system)
    end

    # unpack pre-allocated storage and pointers for system
    @unpack x, r, K, force_scaling, dynamic_indices, udot, θdot, Vdot, Ωdot = system

    if show_trace
        println("Solving for t=$(t0)")
    end
    
    # set current time
    system.t = t0

    # set current parameters
    pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t0)
    dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(t0)
    pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
    gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(tvec[it]))
    x0 = typeof(origin) <: AbstractVector ? SVector{3}(origin) : SVector{3}(origin(t0))
    v0 = typeof(linear_velocity) <: AbstractVector ? SVector{3}(linear_velocity) : SVector{3}(linear_velocity(t0))
    ω0 = typeof(angular_velocity) <: AbstractVector ? SVector{3}(angular_velocity) : SVector{3}(angular_velocity(t0))
    a0 = typeof(linear_acceleration) <: AbstractVector ? SVector{3}(linear_acceleration) : SVector{3}(linear_acceleration(t))
    α0 = typeof(angular_acceleration) <: AbstractVector ? SVector{3}(angular_acceleration) : SVector{3}(angular_acceleration(t))

    # construct residual and jacobian functions
    f! = (resid, x) -> initial_condition_system_residual!(resid, x, dynamic_indices, 
        force_scaling, structural_damping, assembly, pcond, dload, pmass, gvec, 
        x0, v0, ω0, a0, α0, u0, theta0, udot0, thetadot0)

    j! = (jacob, x) -> initial_condition_system_jacobian!(jacob, x, dynamic_indices, 
        force_scaling, structural_damping, assembly, pcond, dload, pmass, gvec, 
        x0, v0, ω0, a0, α0, u0, theta0, udot0, thetadot0)

    # solve system of equations
    if linear
        # set up a linear analysis
        if isnothing(linearization_state)
            x .= 0
        else
            x .= linearization_state
        end
        f!(r, x)
        j!(K, x)

        # update the solution
        x .-= safe_lu(K) \ r
    
        # set convergence flag
        converged = true
    else
        # perform a nonlinear analysis
        df = OnceDifferentiable(f!, j!, x, r, K)

        result = NLsolve.nlsolve(df, x,
            show_trace=show_trace,
            linsolve=(x, A, b) -> ldiv!(x, safe_lu(A), b),
            method=method,
            linesearch=linesearch,
            ftol=ftol,
            iterations=iterations)

        # update the solution
        x .= result.zero
        K .= df.DF

        # set convergence flag
        converged = result.f_converged
    end

    # save calculated variables
    for ipoint = 1:length(assembly.points)
        udot[ipoint], θdot[ipoint] = udot0[ipoint], thetadot0[ipoint]
        Vdot[ipoint], Ωdot[ipoint] = point_displacement_rates(x, ipoint, dynamic_indices.icol_point, pcond)
    end

    # restore original state vector
    set_state_variables!(system, pcond; u=u0, theta=theta0)

    return system, converged
end

"""
    time_domain_analysis(assembly, tvec; kwargs...)

Perform a time-domain analysis for the system of nonlinear beams contained in
`assembly` using the time vector `tvec`.  Return the final system, a post-processed
solution history, and a convergence flag indicating whether the iterations
converged for each time step.

# Keyword Arguments
 - `prescribed_conditions: A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and values of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads: A dictionary with keys corresponding to the elements to
        which distributed loads are applied and values of type
        [`DistributedLoads`](@ref) which describe the distributed loads at those
        points.  If time varying, this input may be provided as a function of
        time.
 - `point_masses = Dict{Int,PointMass{Float64}}()`: A dictionary with keys 
        corresponding to the points to which point masses are attached and values 
        of type [`PointMass`](@ref) which contain the properties of the attached 
        point masses.  If time varying, this input may be provided as a function of time.
 - `structural_damping = true`: Flag indicating whether structural damping should be enabled
 - `gravity = [0,0,0]`: Gravity vector.  If time varying, this input may be provided as a 
        function of time.
 - `linear = false`: Set to `true` for a linear analysis
 - `linearization_state`: Linearization state variables.  Defaults to zeros.
 - `update_linearization_state`: Flag indicating whether to update the linearization state 
    variables for a linear analysis with the current state variables.
 - `method = :newton`: Method (as defined in NLsolve) to solve nonlinear system of equations
 - `linesearch = LineSearches.LineSearches.BackTracking(maxstep=1e6)`: Line search used to solve nonlinear system of equations
 - `ftol = 1e-9`: tolerance for solving nonlinear system of equations
 - `iterations = 1000`: maximum iterations for solving the nonlinear system of equations
 - `show_trace = false`: Flag indicating whether to show solution progress
 - `reset_state = true`: Flag indicating whether the state variables should be
    reset prior to performing the analysis.  This keyword argument is only valid
    for the pre-allocated version of this function.
 - `initialize = true`: Flag indicating whether a consistent set of initial
    conditions should be found using [`initial_condition_analysis`](@ref). If
    `false`, the keyword arguments `u0`, `theta0`, `udot0` and `thetadot0` will
    be ignored and the system state vector will be used as the initial state
    variables.
 - `origin`: Global frame origin vector. If time varying, this input
    may be provided as a function of time.
 - `linear_velocity`: Global frame linear velocity vector. If time
    varying, this vector may be provided as a function of time.
 - `angular_velocity`: Global frame angular velocity vector. If time
    varying, this vector may be provided as a function of time.
 - `linear_acceleration = zeros(3)`: Global frame linear acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `angular_acceleration = zeros(3)`: Global frame angular acceleration vector. If time
    varying, this vector may be provided as a function of time.
 - `u0=fill(zeros(3), length(assembly.points))`: Initial displacment of each beam element,
 - `theta0=fill(zeros(3), length(assembly.points))`: Initial angular displacement of each beam element,
 - `udot0=fill(zeros(3), length(assembly.points))`: Initial time derivative with respect to `u`
 - `thetadot0=fill(zeros(3), length(assembly.points))`: Initial time derivative with respect to `theta`
 - `save=1:length(tvec)`: Steps at which to save the time history
"""
function time_domain_analysis(assembly, tvec; kwargs...)

    system = System(assembly)

    return time_domain_analysis!(system, assembly, tvec; kwargs...)
end

"""
    time_domain_analysis!(system, assembly, tvec; kwargs...)

Pre-allocated version of [`time_domain_analysis`](@ref).
"""
function time_domain_analysis!(system, assembly, tvec;
    prescribed_conditions=Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads=Dict{Int,DistributedLoads{Float64}}(),
    point_masses=Dict{Int,PointMass{Float64}}(),
    structural_damping=true,
    gravity=SVector(0,0,0),
    linear=false,
    linearization_state=nothing,
    update_linearization_state=false,
    method=:newton,
    linesearch=LineSearches.BackTracking(maxstep=1e6),
    ftol=1e-9,
    iterations=250,
    show_trace=false,
    reset_state=true,
    initialize=true,
    origin=(@SVector zeros(3)),
    linear_velocity=(@SVector zeros(3)),
    angular_velocity=(@SVector zeros(3)),
    linear_acceleration=(@SVector zeros(3)),
    angular_acceleration=(@SVector zeros(3)),
    u0=fill((@SVector zeros(3)), length(assembly.points)),
    theta0=fill((@SVector zeros(3)), length(assembly.points)),
    udot0=fill((@SVector zeros(3)), length(assembly.points)),
    thetadot0=fill((@SVector zeros(3)), length(assembly.points)),
    save=1:length(tvec)
    )

    if reset_state
        reset_state!(system)
    end

    # perform initial condition analysis
    if initialize
        system, converged = initial_condition_analysis!(system, assembly, tvec[1];
            prescribed_conditions=prescribed_conditions,
            distributed_loads=distributed_loads,
            point_masses=point_masses,
            structural_damping=structural_damping,
            gravity=gravity,
            linear=linear,
            linearization_state=linearization_state,
            method=method,
            linesearch=linesearch,
            ftol=ftol,
            iterations=iterations,
            show_trace=show_trace,
            reset_state=false,
            origin=origin,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            angular_acceleration=angular_acceleration,
            u0=u0,
            theta0=theta0,
            udot0=udot0,
            thetadot0=thetadot0,
            )
    else
        # converged by default
        converged = true
    end

    # unpack pre-allocated storage and pointers for system
    @unpack x, r, K, force_scaling, dynamic_indices, udot, θdot, Vdot, Ωdot = system

    # initialize storage for each time step
    isave = 1
    history = Vector{AssemblyState{eltype(system)}}(undef, length(save))

    # add initial state to the solution history
    if isave in save
        pcond = typeof(prescribed_conditions) <: AbstractDict ?
            prescribed_conditions : prescribed_conditions(tvec[1])
        history[isave] = AssemblyState(system, assembly, prescribed_conditions=pcond)
        isave += 1
    end

    # --- Begin Time Domain Simulation --- #

    for it = 2:length(tvec)

        if show_trace
            println("Solving for t=$(tvec[it])")
        end

        # update current time
        system.t = tvec[it]

        # current time step size
        dt = tvec[it] - tvec[it-1]

        # current parameters
        pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(tvec[it])
        dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(tvec[it])
        pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
        gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(tvec[it]))
        x0 = typeof(origin) <: AbstractVector ? SVector{3}(origin) : SVector{3}(origin(tvec[it]))
        v0 = typeof(linear_velocity) <: AbstractVector ? SVector{3}(linear_velocity) : SVector{3}(linear_velocity(tvec[it]))
        ω0 = typeof(angular_velocity) <: AbstractVector ? SVector{3}(angular_velocity) : SVector{3}(angular_velocity(tvec[it]))
        a0 = typeof(linear_acceleration) <: AbstractVector ? SVector{3}(linear_acceleration) : SVector{3}(linear_acceleration(t))
        α0 = typeof(angular_acceleration) <: AbstractVector ? SVector{3}(angular_acceleration) : SVector{3}(angular_acceleration(t))

        # set current initialization parameters
        for ipoint = 1:length(assembly.points)
            # extract beam element state variables
            u, θ = point_displacement(x, ipoint, dynamic_indices.icol_point, pcond)
            V, Ω = point_velocities(x, ipoint, dynamic_indices.icol_point)
            # calculate state rate initialization terms, use storage for state rates
            udot[ipoint] = 2 / dt * u + udot[ipoint]
            θdot[ipoint] = 2 / dt * θ + θdot[ipoint]
            Vdot[ipoint] = 2 / dt * V + Vdot[ipoint]
            Ωdot[ipoint] = 2 / dt * Ω + Ωdot[ipoint]
        end

        # solve for the state variables at the next time step
        f! = (r, x) -> newmark_system_residual!(r, x, dynamic_indices, force_scaling, 
            structural_damping, assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0,
            udot, θdot, Vdot, Ωdot, dt)

        j! = (K, x) -> newmark_system_jacobian!(K, x, dynamic_indices, force_scaling, 
            structural_damping, assembly, pcond, dload, pmass, gvec, x0, v0, ω0, a0, α0,
            udot, θdot, Vdot, Ωdot, dt)

        # solve system of equations
        if linear
            # linear analysis
            if !update_linearization_state
                if isnothing(linearization_state)
                    x .= 0
                else
                    x .= linearization_state
                end
            end
            f!(r, x)
            j!(K, x)
            x .-= safe_lu(K) \ r
        else
            df = OnceDifferentiable(f!, j!, x, r, K)

            result = NLsolve.nlsolve(df, x,
                show_trace=show_trace,
                linsolve=(x, A, b) -> ldiv!(x, safe_lu(A), b),
                method=method,
                linesearch=linesearch,
                ftol=ftol,
                iterations=iterations)

            x .= result.zero
            K .= df.DF
        end

        # set new state rates
        for ipoint = 1:length(assembly.points)
            u, θ = point_displacement(x, ipoint, dynamic_indices.icol_point, pcond)
            V, Ω = point_velocities(x, ipoint, dynamic_indices.icol_point)
            udot[ipoint] = 2/dt*u - udot[ipoint]
            θdot[ipoint] = 2/dt*θ - θdot[ipoint]
            Vdot[ipoint] = 2/dt*V - Vdot[ipoint]
            Ωdot[ipoint] = 2/dt*Ω - Ωdot[ipoint]
        end

        # add state to history
        if it in save
            history[isave] = AssemblyState(system, assembly, prescribed_conditions=pcond)
            isave += 1
        end

        # stop early if unconverged
        if !linear && !result.f_converged
            if show_trace
                println("Solution failed to converge")
            end
            history = history[1:it]
            converged = false
            break
        end

    end

    return system, history, converged
end
