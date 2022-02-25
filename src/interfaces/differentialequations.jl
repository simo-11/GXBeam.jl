"""
    ODEProblem(system::GXBeam.System, assembly, tspan; kwargs...)

Construct a `ODEProblem` for the system of nonlinear beams
contained in `assembly` which may be used with the DifferentialEquations package.

Keyword Arguments:
 - `prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}()`:
        A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and elements of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads = Dict{Int,DistributedLoads{Float64}}()`: A dictionary
        with keys corresponding to the elements to which distributed loads are
        applied and elements of type [`DistributedLoads`](@ref) which describe
        the distributed loads on those elements.  If time varying, this input may
        be provided as a function of time.
 - `point_masses = Dict{Int,Vector{PointMass{Float64}}}()`: A dictionary with keys 
        corresponding to the points at which point masses are attached and values 
        containing vectors of objects of type [`PointMass`](@ref) which describe 
        the point masses attached at those points.  If time varying, this input may
        be provided as a function of time.
 - `gravity`: Gravity vector. If time varying, this input may be provided as a 
        function of time.
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
"""
function SciMLBase.ODEProblem(system::System, assembly, tspan; kwargs...)

    N = length(system.x)
    nelem = length(assembly.elements)

    # use initial state from `system`
    u0 = similar(system.x, N + 12*nelem)
    u0[1:N] .= system.x
    for ielem = 1:nelem
        icol = N+12*(ielem-1)
        u0[icol+1:icol+3] = system.udot[ielem]
        u0[icol+4:icol+6] = system.θdot[ielem]
        u0[icol+7:icol+9] = system.Vdot[ielem]
        u0[icol+10:icol+12] = system.Ωdot[ielem]
    end

    # create ODEFunction
    func = SciMLBase.ODEFunction(system, assembly; kwargs...)

    return SciMLBase.ODEProblem{true}(func, u0, tspan)
end

"""
    ODEFunction(system::GXBeam.System, assembly; 
    )

Construct a `ODEFunction` for the system of nonlinear beams
contained in `assembly` which may be used with the DifferentialEquations package.

**Note that this function defines a non-constant mass matrix, which is not directly 
supported by DifferentialEquations.jl.  This function is therefore experimental**

The parameters associated with the resulting ODEFunction are defined by the tuple
`(prescribed_conditions, distributed_loads, point_masses, origin, linear_velocity, 
angular_velocity, linear_acceleration, angular_acceleration)` where each parameter is defined as follows:
 - `prescribed_conditions`: A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and elements of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads`: A dictionary with keys corresponding to the elements to
        which distributed loads are applied and elements of type
        [`DistributedLoads`](@ref) which describe the distributed loads at those
        points.  If time varying, this input may be provided as a function of
        time.
 - `point_masses = Dict{Int,Vector{PointMass{Float64}}}()`: A dictionary with keys 
        corresponding to the points at which point masses are attached and values 
        containing vectors of objects of type [`PointMass`](@ref) which describe 
        the point masses attached at those points.  If time varying, this input may
        be provided as a function of time.
 - `gravity`: Gravity vector. If time varying, this input may be provided as a 
        function of time.
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
"""
function SciMLBase.ODEFunction(system::System, assembly;
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads = Dict{Int,DistributedLoads{Float64}}(),
    point_masses = Dict{Int,Vector{PointMass{Float64}}}(),
    gravity = (@SVector zeros(3)),
    origin = (@SVector zeros(3)),
    linear_velocity = (@SVector zeros(3)),
    angular_velocity = (@SVector zeros(3)),
    linear_acceleration = (@SVector zeros(3)),
    angular_acceleration = (@SVector zeros(3)),
    )

    # check to make sure the system isn't static
    @assert !system.static

    # unpack system pointers
    N = length(system.x)
    nelem = length(assembly.elements)
    npoint = length(assembly.points)
    irow_point = system.irow_point
    irow_elem = system.irow_elem
    irow_elem1 = system.irow_elem1
    irow_elem2 = system.irow_elem2
    icol_point = system.icol_point
    icol_elem = system.icol_elem

    # unpack scaling parameters
    force_scaling = system.force_scaling

    # DAE function
    f = function(du, u, p, t)

        # get current parameters
        pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)
        dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(t)
        pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
        gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(t))
        x0 = typeof(origin) <: AbstractVector ? SVector{3}(origin) : SVector{3}(origin(t))
        v0 = typeof(linear_velocity) <: AbstractVector ? SVector{3}(linear_velocity) : SVector{3}(linear_velocity(t))
        ω0 = typeof(angular_velocity) <: AbstractVector ? SVector{3}(angular_velocity) : SVector{3}(angular_velocity(t))
        a0 = typeof(linear_acceleration) <: AbstractVector ? SVector{3}(linear_acceleration) : SVector{3}(linear_acceleration(t))
        α0 = typeof(angular_acceleration) <: AbstractVector ? SVector{3}(angular_acceleration) : SVector{3}(angular_acceleration(t))

        # add contributions to residual equations from the beam elements
        for ielem = 1:nelem
    
            # get pointers for element
            icol = icol_elem[ielem]
            irow_e = irow_elem[ielem]
            irow_e1 = irow_elem1[ielem]
            irow_p1 = irow_point[assembly.start[ielem]]
            irow_e2 = irow_elem2[ielem]
            irow_p2 = irow_point[assembly.stop[ielem]]
    
            # set state rates for element
            udot = SVector(u[N+12*(ielem-1)+1], u[N+12*(ielem-1)+2], u[N+12*(ielem-1)+3])
            θdot = SVector(u[N+12*(ielem-1)+4], u[N+12*(ielem-1)+5], u[N+12*(ielem-1)+6])
            Vdot = SVector(u[N+12*(ielem-1)+7], u[N+12*(ielem-1)+8], u[N+12*(ielem-1)+9])
            Ωdot = SVector(u[N+12*(ielem-1)+10], u[N+12*(ielem-1)+11], u[N+12*(ielem-1)+12])
    
            dynamic_element_residual!(du, u, ielem, assembly.elements[ielem],
                 dload, pmass, gvec, force_scaling, icol, irow_e, irow_e1, irow_p1, irow_e2, irow_p2,
                 x0, v0, ω0, a0, α0, udot, θdot, Vdot, Ωdot)
    
        end
    
        # add contributions to the residual equations from the prescribed point conditions
        for ipoint = 1:npoint
    
            # skip if the unknowns have been eliminated from the system of equations
            if icol_point[ipoint] <= 0
                continue
            end
    
            icol = icol_point[ipoint]
            irow_p = irow_point[ipoint]
    
            point_residual!(du, u, ipoint, assembly, pcond, force_scaling, icol, irow_p, 
                irow_elem1, irow_elem2)
        end

        for i = N+1:N+12*nelem
            du[i] = u[i]
        end
    
        return du
    end

    # construct mass matrix
    mass_matrix = spzeros(N+12*nelem, N+12*nelem)
    mass_matrix .= 0
    for ielem = 1:nelem
        irow = N+12*(ielem-1)+1
        icol = icol_elem[ielem]
        mass_matrix[irow, icol] = 1
        mass_matrix[irow+1, icol+1] = 1
        mass_matrix[irow+2, icol+2] = 1
        mass_matrix[irow+3, icol+3] = 1
        mass_matrix[irow+4, icol+4] = 1
        mass_matrix[irow+5, icol+5] = 1
        mass_matrix[irow+6, icol+12] = 1
        mass_matrix[irow+7, icol+13] = 1
        mass_matrix[irow+8, icol+14] = 1
        mass_matrix[irow+9, icol+15] = 1
        mass_matrix[irow+10, icol+16] = 1
        mass_matrix[irow+11, icol+17] = 1
    end

    # construct jacobian
    jac = function(J, u, p, t)

        J .= 0

        # get current parameters
        pcond = typeof(prescribed_conditions) <: AbstractDict ? prescribed_conditions : prescribed_conditions(t)
        dload = typeof(distributed_loads) <: AbstractDict ? distributed_loads : distributed_loads(t)
        pmass = typeof(point_masses) <: AbstractDict ? point_masses : point_masses(t)
        gvec = typeof(gravity) <: AbstractVector ? SVector{3}(gravity) : SVector{3}(gravity(t))
        x0 = typeof(origin) <: AbstractVector ? SVector{3}(origin) : SVector{3}(origin(t))
        v0 = typeof(linear_velocity) <: AbstractVector ? SVector{3}(linear_velocity) : SVector{3}(linear_velocity(t))
        ω0 = typeof(angular_velocity) <: AbstractVector ? SVector{3}(angular_velocity) : SVector{3}(angular_velocity(t))
        a0 = typeof(linear_acceleration) <: AbstractVector ? SVector{3}(linear_acceleration) : SVector{3}(linear_acceleration(t))
        α0 = typeof(angular_acceleration) <: AbstractVector ? SVector{3}(angular_acceleration) : SVector{3}(angular_acceleration(t))

        # add contributions to residual equations from the beam elements
        for ielem = 1:nelem
    
            # get pointers for element
            icol1 = icol_elem[ielem]
            icol2 = N+12*(ielem-1)+1
            irow_e = irow_elem[ielem]
            irow_p1 = irow_point[assembly.start[ielem]]
            irow_p2 = irow_point[assembly.stop[ielem]]
    
            # use storage for the jacobian to calculate the mass matrix
            element_mass_matrix!(J, u, ielem, assembly.elements[ielem], pmass, 
                force_scaling, icol1, irow_e, irow_p1, irow_p2)

            # move jacobian entries into appropriate slots
            for irange in (irow_e:irow_e+5, irow_p1:irow_p1+5, irow_p2:irow_p2+5)
                for i in irange
                    J[i, icol2] = J[i, icol1]
                    J[i, icol2+1] = J[i, icol1+1]
                    J[i, icol2+2] = J[i, icol1+2]
                    J[i, icol2+3] = J[i, icol1+3]
                    J[i, icol2+4] = J[i, icol1+4]
                    J[i, icol2+5] = J[i, icol1+5]
                    J[i, icol2+6] = J[i, icol1+12]
                    J[i, icol2+7] = J[i, icol1+13]
                    J[i, icol2+8] = J[i, icol1+14]
                    J[i, icol2+9] = J[i, icol1+15]
                    J[i, icol2+10] = J[i, icol1+16]
                    J[i, icol2+11] = J[i, icol1+17]
                end
            end
        end

        # add contributions to residual equations from the beam elements
        for ielem = 1:nelem
    
            # get pointers for element
            icol = icol_elem[ielem]
            irow_e = irow_elem[ielem]
            irow_e1 = irow_elem1[ielem]
            irow_p1 = irow_point[assembly.start[ielem]]
            irow_e2 = irow_elem2[ielem]
            irow_p2 = irow_point[assembly.stop[ielem]]
    
            # set state rates for element
            udot = SVector(u[N+12*(ielem-1)+1], u[N+12*(ielem-1)+2], u[N+12*(ielem-1)+3])
            θdot = SVector(u[N+12*(ielem-1)+4], u[N+12*(ielem-1)+5], u[N+12*(ielem-1)+6])
            Vdot = SVector(u[N+12*(ielem-1)+7], u[N+12*(ielem-1)+8], u[N+12*(ielem-1)+9])
            Ωdot = SVector(u[N+12*(ielem-1)+10], u[N+12*(ielem-1)+11], u[N+12*(ielem-1)+12])
    
            # compute the jacobian
            dynamic_element_jacobian!(J, u, ielem, assembly.elements[ielem],
                 dload, pmass, gvec, force_scaling, icol, irow_e, irow_e1, irow_p1, irow_e2, irow_p2,
                 x0, v0, ω0, a0, α0, udot, θdot, Vdot, Ωdot)

        end
    
        # add contributions to the residual equations from the prescribed point conditions
        for ipoint = 1:npoint
    
            # skip if the unknowns have been eliminated from the system of equations
            if icol_point[ipoint] <= 0
                continue
            end
    
            icol = icol_point[ipoint]
            irow_p = irow_point[ipoint]
    
            point_jacobian!(J, u, ipoint, assembly, pcond,
                force_scaling, icol, irow_p, irow_elem1, irow_elem2)
        end

        for i = N+1:N+12*nelem
            J[i,i] = 1
        end
    
        return J
    end

    jac_prototype = similar(system.K, N+12*nelem, N+12*nelem)

    return SciMLBase.ODEFunction{true,true}(f; mass_matrix, jac, jac_prototype)
end


"""
    DAEProblem(system::GXBeam.System, assembly, tspan; kwargs...)

Construct a `DAEProblem` for the system of nonlinear beams
contained in `assembly` which may be used with the DifferentialEquations package.

A consistent set of initial conditions may be obtained prior to constructing the
DAEProblem using [`initial_condition_analysis!`](@ref) or by constructing a
DAEProblem after a time domain analysis.

Keyword Arguments:
 - `prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}()`:
        A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and elements of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads = Dict{Int,DistributedLoads{Float64}}()`: A dictionary
        with keys corresponding to the elements to which distributed loads are
        applied and elements of type [`DistributedLoads`](@ref) which describe
        the distributed loads on those elements.  If time varying, this input may
        be provided as a function of time.
 - `point_masses = Dict{Int,Vector{PointMass{Float64}}}()`: A dictionary with keys 
        corresponding to the points at which point masses are attached and values 
        containing vectors of objects of type [`PointMass`](@ref) which describe 
        the point masses attached at those points.  If time varying, this input may
        be provided as a function of time.
 - `gravity = zeros(3)`: Gravity vector. If time varying, this input may be provided as a 
        function of time.
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
"""
function SciMLBase.DAEProblem(system::System, assembly, tspan;
    prescribed_conditions = Dict{Int,PrescribedConditions{Float64}}(),
    distributed_loads = Dict{Int,DistributedLoads{Float64}}(),
    point_masses = Dict{Int,Vector{PointMass{Float64}}}(),
    gravity = (@SVector zeros(3)),
    origin = (@SVector zeros(3)),
    linear_velocity = (@SVector zeros(3)),
    angular_velocity = (@SVector zeros(3)),
    linear_acceleration = (@SVector zeros(3)),
    angular_acceleration = (@SVector zeros(3)),
    )

    # create SciMLBase.DAEFunction
    func = SciMLBase.DAEFunction(system, assembly)

    # use initial state from `system`
    u0 = copy(system.x)

    # use initial state rates from `system`
    du0 = zero(u0)
    for (ielem, icol) in enumerate(system.icol_elem)
        du0[icol:icol+2] = system.udot[ielem]
        du0[icol+3:icol+5] = system.θdot[ielem]
        du0[icol+12:icol+14] = system.Vdot[ielem]
        du0[icol+15:icol+17] = system.Ωdot[ielem]
    end

    # set parameters
    p = (prescribed_conditions, distributed_loads, point_masses, gravity, origin, 
       linear_velocity, angular_velocity, linear_acceleration, angular_acceleration)

    # get differential variables
    differential_vars = get_differential_vars(system, assembly)

    return SciMLBase.DAEProblem{true}(func, du0, u0, tspan, p; differential_vars)
end

"""
    DAEFunction(system::GXBeam.System, assembly)

Construct a `DAEFunction` for the system of nonlinear beams
contained in `assembly` which may be used with the DifferentialEquations package.

The parameters associated with the resulting SciMLBase.DAEFunction are defined by the tuple
`(prescribed_conditions, distributed_loads, point_masses, origin, linear_velocity, 
angular_velocity, linear_acceleration, angular_acceleration)`
where each parameter is defined as follows:
 - `prescribed_conditions`: A dictionary with keys corresponding to the points at
        which prescribed conditions are applied and elements of type
        [`PrescribedConditions`](@ref) which describe the prescribed conditions
        at those points.  If time varying, this input may be provided as a
        function of time.
 - `distributed_loads`: A dictionary with keys corresponding to the elements to
        which distributed loads are applied and elements of type [`DistributedLoads`](@ref) 
        which describe the distributed loads on those elements.  If time varying, this 
        input may be provided as a function of time.
 - `point_masses = Dict{Int,Vector{PointMass{Float64}}}()`: A dictionary with keys 
        corresponding to the points at which point masses are attached and values 
        containing vectors of objects of type [`PointMass`](@ref) which describe 
        the point masses attached at those points.  If time varying, this input may
        be provided as a function of time.
 - `gravity`: Gravity vector. If time varying, this input may be provided as a 
        function of time.
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
"""
function SciMLBase.DAEFunction(system::System, assembly)

    # check to make sure the system isn't static
    @assert !system.static

    # unpack system pointers
    irow_point = system.irow_point
    irow_elem = system.irow_elem
    irow_elem1 = system.irow_elem1
    irow_elem2 = system.irow_elem2
    icol_point = system.icol_point
    icol_elem = system.icol_elem

    # unpack scaling parameters
    force_scaling = system.force_scaling

    # DAE function
    f = function(resid, du, u, p, t)

        # get current parameters
        prescribed_conditions = typeof(p[1]) <: AbstractDict ? p[1] : p[1](t)
        distributed_loads = typeof(p[2]) <: AbstractDict ? p[2] : p[2](t)
        point_masses = typeof(p[3]) <: AbstractDict ? p[3] : p[3](t)
        gvec = typeof(p[4]) <: AbstractVector ? SVector{3}(p[4]) : SVector{3}(p[4](t))
        x0 = typeof(p[5]) <: AbstractVector ? SVector{3}(p[5]) : SVector{3}(p[5](t))
        v0 = typeof(p[6]) <: AbstractVector ? SVector{3}(p[6]) : SVector{3}(p[6](t))
        ω0 = typeof(p[7]) <: AbstractVector ? SVector{3}(p[7]) : SVector{3}(p[7](t))
        a0 = typeof(p[8]) <: AbstractVector ? SVector{3}(p[8]) : SVector{3}(p[8](t))
        α0 = typeof(p[9]) <: AbstractVector ? SVector{3}(p[9]) : SVector{3}(p[9](t))

        # calculate residual
        dynamic_system_residual!(resid, du, u, assembly, prescribed_conditions,
            distributed_loads, point_masses, gvec, force_scaling, irow_point, irow_elem, 
            irow_elem1, irow_elem2, icol_point, icol_elem, x0, v0, ω0, a0, α0)

        return resid
    end

    # jacobian function with respect to states/state rates
    jac = function(J, du, u, p, gamma, t)

        # zero out all jacobian entries
        J .= 0.0

        # get current parameters
        prescribed_conditions = typeof(p[1]) <: AbstractDict ? p[1] : p[1](t)
        distributed_loads = typeof(p[2]) <: AbstractDict ? p[2] : p[2](t)
        point_masses = typeof(p[3]) <: AbstractDict ? p[3] : p[3](t)
        gvec = typeof(p[4]) <: AbstractVector ? SVector{3}(p[4]) : SVector{3}(p[4](t))
        x0 = typeof(p[5]) <: AbstractVector ? SVector{3}(p[5]) : SVector{3}(p[5](t))
        v0 = typeof(p[6]) <: AbstractVector ? SVector{3}(p[6]) : SVector{3}(p[6](t))
        ω0 = typeof(p[7]) <: AbstractVector ? SVector{3}(p[7]) : SVector{3}(p[7](t))
        a0 = typeof(p[8]) <: AbstractVector ? SVector{3}(p[8]) : SVector{3}(p[8](t))
        α0 = typeof(p[9]) <: AbstractVector ? SVector{3}(p[9]) : SVector{3}(p[9](t))

        # calculate jacobian
        dynamic_system_jacobian!(J, du, u, assembly, prescribed_conditions,
            distributed_loads, point_masses, gvec, force_scaling, irow_point, irow_elem, 
            irow_elem1, irow_elem2, icol_point, icol_elem, x0, v0, ω0, a0, α0)

        # add gamma multiplied by the mass matrix
        system_mass_matrix!(J, gamma, u, assembly, point_masses, force_scaling,
            irow_point, irow_elem, irow_elem1, irow_elem2, icol_point, icol_elem)

        return J
    end

    # jacobian prototype (use dense since sparse isn't working)
    jac_prototype = system.K

    # TODO: figure out how to use a sparse matrix here.
    # It's failing with a singular exception during the LU factorization.
    # Using `jac_prototype` also causes errors

    return SciMLBase.DAEFunction{true,true}(f) # TODO: re-add jacobian here once supported
end

function get_differential_vars(system::System, assembly::Assembly)
    differential_vars = fill(false, length(system.x))
    for (ielem, icol) in enumerate(system.icol_elem)
       icol = system.icol_elem[ielem]
       differential_vars[icol:icol+2] .= true # u (for the beam element)
       differential_vars[icol+3:icol+5] .= true # θ (for the beam element)
       for i = 1:6
           if !iszero(assembly.elements[ielem].mu[i])
              differential_vars[icol+5+i] = true
           end
       end
       differential_vars[icol+12:icol+14] .= true # V (for the beam element)
       differential_vars[icol+15:icol+17] .= true # Ω (for the beam element)
    end
    return differential_vars
end
