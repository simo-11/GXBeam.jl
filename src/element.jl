"""
    Element{TF}

Composite type that defines a beam element's properties

# Fields
 - `L`: Length of the beam element
 - `x`: Location of the beam element (the center of the beam element)
 - `compliance`: Compliance matrix for the beam element
 - `mass`: Mass matrix for the beam element
 - `Cab`: Transformation matrix from the undeformed beam element frame to the body frame
 - `mu`: Beam element damping coefficients
"""
struct Element{TF}
    L::TF
    x::SVector{3,TF}
    compliance::SMatrix{6,6,TF,36}
    mass::SMatrix{6,6,TF,36}
    Cab::SMatrix{3,3,TF,9}
    mu::SVector{6,TF}
end

"""
    Element(L, x, compliance, mass, Cab, mu)

Construct a beam element

# Arguments
- `L`: Length of the beam element
- `x`: Location of the beam element (the center of the beam element)
- `compliance`: Beam element compliance matrix
- `mass`: Beam element mass matrix
- `Cab`: Transformation matrix from the undeformed beam element frame to the body frame
- `mu`: Beam element damping coefficients
"""
function Element(L, x, compliance, mass, Cab, mu)
    TF = promote_type(typeof(L), eltype(x), eltype(compliance), eltype(mass), eltype(Cab), eltype(mu))
    return Element{TF}(L, x, compliance, mass, Cab, mu)
end

"""
    element_strain(element, F, M)

Calculate the strain of a beam element given the resultant forces and moments applied on
the element expressed in the deformed beam element frame
"""
@inline function element_strain(element, F, M)
    C = element.compliance
    S11 = C[SVector{3}(1:3), SVector{3}(1:3)]
    S12 = C[SVector{3}(1:3), SVector{3}(4:6)]
    return S11*F + S12*M
end

"""
    element_curvature(element, F, M)

Calculate the curvature of a beam element given the resultant force and moments applied on
the element expressed in the deformed beam element frame
"""
@inline function element_curvature(element, F, M)
    C = element.compliance
    S21 = C[SVector{3}(4:6), SVector{3}(1:3)]
    S22 = C[SVector{3}(4:6), SVector{3}(4:6)]
    return S21*F + S22*M
end

"""
    element_linear_momentum(element, v, ω)

Calculate the linear momentum of a beam element given the linear and angular velocity of
the element expressed in the deformed beam element frame
"""
@inline function element_linear_momentum(element, v, ω)
    M = element.mass
    mass11 = M[SVector{3}(1:3), SVector{3}(1:3)]
    mass12 = M[SVector{3}(1:3), SVector{3}(4:6)]
    return mass11*v + mass12*ω
end

"""
    element_angular_momentum(element, v, ω)

Calculate the angular momentum of a beam element given the linear and angular velocity of
the element expressed in the deformed beam element frame
"""
@inline function element_angular_momentum(element, v, ω)
    M = element.mass
    mass21 = M[SVector{3}(4:6), SVector{3}(1:3)]
    mass22 = M[SVector{3}(4:6), SVector{3}(4:6)]
    return mass21*v + mass22*ω
end

"""
    element_states(x, ielem, icol_elem, force_scaling)

Extract the state variables `F` and `M` of element `ielem` from the state variable vector.
"""
@inline function element_states(x, ielem, icol_elem, force_scaling)

    icol = icol_elem[ielem]

    F = SVector(x[icol  ], x[icol+1], x[icol+2]) .* force_scaling
    M = SVector(x[icol+3], x[icol+4], x[icol+5]) .* force_scaling

    return F, M
end

"""
    static_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity)

Calculate/extract the element properties needed to construct the residual for a static 
analysis
"""
@inline function static_element_properties(x, indices, force_scaling, assembly, ielem, 
    prescribed_conditions, gravity)

    # unpack element parameters
    @unpack L, Cab, compliance, mass = assembly.elements[ielem]

    # compliance submatrices
    S11 = compliance[SVector{3}(1:3), SVector{3}(1:3)]
    S12 = compliance[SVector{3}(1:3), SVector{3}(4:6)]
    S21 = compliance[SVector{3}(4:6), SVector{3}(1:3)]
    S22 = compliance[SVector{3}(4:6), SVector{3}(4:6)]

    # mass submatrices
    mass11 = mass[SVector{3}(1:3), SVector{3}(1:3)]
    mass12 = mass[SVector{3}(1:3), SVector{3}(4:6)]
    mass21 = mass[SVector{3}(4:6), SVector{3}(1:3)]
    mass22 = mass[SVector{3}(4:6), SVector{3}(4:6)]

    # forces and moments
    F, M = element_states(x, ielem, indices.icol_elem, force_scaling)

    # strain and curvature
    γ = S11*F + S12*M
    κ = S21*F + S22*M

    # linear and angular displacement
    u1, θ1 = point_displacement(x, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    u2, θ2 = point_displacement(x, assembly.stop[ielem], indices.icol_point, prescribed_conditions)
    u = (u1 + u2)/2
    θ = (θ1 + θ2)/2

    # rotation matrices
    C = get_C(θ)
    Ct = C'
    CtCab = C'*Cab
    Qinv = get_Qinv(θ)
   
    # linear and angular acceleration
    a = -gravity
    α = @SVector zeros(3)

    return (; L, C, Ct, Cab, CtCab, Qinv, S11, S12, S21, S22, mass11, mass12, mass21, mass22, 
        u1, u2, θ1, θ2, u, θ, F, M, γ, κ, a, α)
end

"""
    steady_state_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

Calculate/extract the element properties needed to construct the residual for a steady 
state analysis
"""
@inline function steady_state_element_properties(x, indices, force_scaling, assembly, ielem, 
    prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    properties = static_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity)

    @unpack u, θ, a, α = properties

    # distance from the rotation center
    Δx = assembly.elements[ielem].x - x0

    # undeformed reference frame linear and angular velocity
    v = v0 + cross(ω0, Δx)
    ω = ω0

    # linear and angular velocity (prescribed)
    V1, Ω1 = point_velocities(x, assembly.start[ielem], indices.icol_point)
    V2, Ω2 = point_velocities(x, assembly.stop[ielem], indices.icol_point)
    V = (V1 + V2)/2
    Ω = (Ω1 + Ω2)/2

    # linear and angular acceleration
    a += a0 + cross(α0, Δx) + cross(α0, u)
    α += α0

    return (; properties..., V1, V2, Ω1, Ω2, V, Ω, v, ω, a, α) 
end

"""
    initial_condition_element_properties(x, indices, force_scaling, assembly, ielem,
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0,
        u0, θ0, udot0, θdot0)

Calculate/extract the element properties needed to construct the residual for a time-domaing
analysis initialization
"""
@inline function initial_condition_element_properties(x, indices, force_scaling, assembly, ielem,
    prescribed_conditions, gravity, x0, v0, ω0, a0, α0, u0, θ0, udot0, θdot0)

    # unpack element parameters
    @unpack L, Cab, compliance, mass = assembly.elements[ielem]

    # compliance submatrices
    S11 = compliance[SVector{3}(1:3), SVector{3}(1:3)]
    S12 = compliance[SVector{3}(1:3), SVector{3}(4:6)]
    S21 = compliance[SVector{3}(4:6), SVector{3}(1:3)]
    S22 = compliance[SVector{3}(4:6), SVector{3}(4:6)]

    # mass matrix submatrices
    mass11 = mass[SVector{3}(1:3), SVector{3}(1:3)]
    mass12 = mass[SVector{3}(1:3), SVector{3}(4:6)]
    mass21 = mass[SVector{3}(4:6), SVector{3}(1:3)]
    mass22 = mass[SVector{3}(4:6), SVector{3}(4:6)]

    # forces and moments
    F, M = element_states(x, ielem, indices.icol_elem, force_scaling)

    # strain and curvature
    γ = S11*F + S12*M
    κ = S21*F + S22*M

    # linear and angular displacements
    u1 = SVector{3}(u0[assembly.start[ielem]])
    θ1 = SVector{3}(θ0[assembly.start[ielem]])

    u2 = SVector{3}(u0[assembly.stop[ielem]])
    θ2 = SVector{3}(θ0[assembly.stop[ielem]])

    u = (u1 + u2)/2
    θ = (θ1 + θ2)/2

    # transformation matrices
    C = get_C(θ)
    Ct = C'
    CtCab = C'*Cab
    Qinv = get_Qinv(θ)

    # distance from the rotation center
    Δx = assembly.elements[ielem].x - x0

    # undeformed reference frame linear and angular velocity
    v = v0 + cross(ω0, Δx)
    ω = ω0

    # linear and angular velocity
    V1, Ω1 = point_velocities(x, assembly.start[ielem], indices.icol_point)
    V2, Ω2 = point_velocities(x, assembly.stop[ielem], indices.icol_point)
    V = (V1 + V2)/2
    Ω = (Ω1 + Ω2)/2

    # linear and angular velocity rates
    V1dot, Ω1dot = point_displacement_rates(x, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    V2dot, Ω2dot = point_displacement_rates(x, assembly.stop[ielem], indices.icol_point, prescribed_conditions)
    Vdot = (V1dot + V2dot)/2
    Ωdot = (Ω1dot + Ω2dot)/2

    # linear and angular acceleration
    a = a0 + cross(α0, Δx) + cross(α0, u) + Vdot - gravity
    α = α0 + Ωdot

    return (; L, C, Ct, Cab, CtCab, Qinv, S11, S12, S21, S22, mass11, mass12, mass21, mass22, 
        u1, u2, θ1, θ2, F, M, u, θ, V, Ω, γ, κ, v, ω, a, α) 
end

"""
    newmark_element_properties(x, indices, force_scaling, assembly, ielem,
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0,
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

Calculate/extract the element properties needed to construct the residual for a newmark-
scheme time stepping analysis
"""
@inline function newmark_element_properties(x, indices, force_scaling, assembly, ielem,
    prescribed_conditions, gravity, x0, v0, ω0, a0, α0, 
    udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    properties = steady_state_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    @unpack V1, V2, Ω1, Ω2, u, θ, a, α = properties

    # transformation matrices
    Q = get_Q(θ)

    # velocity rates
    V1dot = 2/dt*V1 - SVector{3}(Vdot_init[assembly.start[ielem]])
    Ω1dot = 2/dt*Ω1 - SVector{3}(Ωdot_init[assembly.start[ielem]])

    V2dot = 2/dt*V2 - SVector{3}(Vdot_init[assembly.stop[ielem]])
    Ω2dot = 2/dt*Ω2 - SVector{3}(Ωdot_init[assembly.stop[ielem]])

    Vdot = (V1dot + V2dot)/2
    Ωdot = (Ω1dot + Ω2dot)/2

    # linear and angular acceleration
    a += Vdot
    α += Ωdot

    return (; properties..., Q, a, α) 
end

"""
    dynamic_element_properties(dx, x, indices, force_scaling, assembly, ielem,
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

Calculate/extract the element properties needed to construct the residual for a dynamic 
analysis
"""
@inline function dynamic_element_properties(dx, x, indices, force_scaling, assembly, ielem,
    prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    properties = steady_state_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    @unpack u, θ, V, Ω, a, α = properties

    # rotation parameter matrices
    Q = get_Q(θ)

    # displacement rates
    V1dot, Ω1dot = point_velocity_rates(dx, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    V2dot, Ω2dot = point_velocity_rates(dx, assembly.stop[ielem], indices.icol_point, prescribed_conditions)
    Vdot = (V1dot + V2dot)/2
    Ωdot = (Ω1dot + Ω2dot)/2

    # linear and angular acceleration
    a += Vdot
    α += Ωdot

    return (; properties..., Q, a, α) 
end

"""
    static_element_resultants(properties, distributed_loads, ielem)

Calculate the resultant loads applied at each end of a beam element for a static analysis.
"""
@inline function static_element_resultants(properties, distributed_loads, ielem)
  
    @unpack L, Ct, CtCab, mass11, mass12, mass21, mass22, F, M, γ, κ, a, α = properties

    # loads due to internal loads

    tmp = CtCab*F
    F1 = tmp
    F2 = tmp

    tmp1 = CtCab*M
    tmp2 = L/2*CtCab*cross(e1 + γ, F)
    M1 = tmp1 + tmp2
    M2 = tmp1 - tmp2

    # # add loads due to linear and angular momentum rates (and gravity)
    Pdot = CtCab*mass11*CtCab'*a + CtCab*mass12*CtCab'*α
    Hdot = CtCab*mass21*CtCab'*a + CtCab*mass22*CtCab'*α

    tmp = L/2*Pdot
    F1 -= tmp
    F2 += tmp

    tmp = L/2*Hdot
    M1 -= tmp
    M2 += tmp

    # add contributions from distributed loads
    if haskey(distributed_loads, ielem)
        dload = distributed_loads[ielem]
        F1 += dload.f1 + Ct*dload.f1_follower
        F2 -= dload.f2 + Ct*dload.f2_follower
        M1 += dload.m1 + Ct*dload.m1_follower
        M2 -= dload.m2 + Ct*dload.m2_follower
    end    

    return (; F1, F2, M1, M2)
end

"""
    dynamic_element_resultants(properties, distributed_loads, ielem)

Calculate the resultant loads applied at each end of a beam element for a dynamic 
analysis.
"""
@inline function dynamic_element_resultants(properties, distributed_loads, ielem)

    resultants = static_element_resultants(properties, distributed_loads, ielem)

    @unpack F1, F2, M1, M2 = resultants

    @unpack L, CtCab, mass11, mass12, mass21, mass22, V, Ω, v, ω = properties

    # linear and angular momentum
    P = CtCab*mass11*CtCab'*V + CtCab*mass12*CtCab'*Ω
    H = CtCab*mass21*CtCab'*V + CtCab*mass22*CtCab'*Ω

    # resultant forces
    tmp = L/2*cross(ω, P) 
    F1 -= tmp
    F2 += tmp

    # resultant moments
    tmp = L/2*(cross(ω, H) + cross(V, P))
    M1 -= tmp
    M2 += tmp

    return (; resultants..., F1, F2, M1, M2)
end

"""
    compatability_residuals(properties)

Calculate the compatability residuals for the beam element
"""
@inline function compatability_residuals(properties)
   
    @unpack L, Cab, CtCab, Qinv, γ, κ, u1, u2, θ1, θ2 = properties

    Δu = L*(CtCab*(e1 + γ) - Cab*e1)

    Δθ = L*Qinv*Cab*κ

    ru = u2 - u1 - Δu
    rθ = θ2 - θ1 - Δθ

    return (; ru, rθ)
end

"""
    insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

Insert the residual entries corresponding to a beam element into the system residual vector.
"""
@inline function insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, 
    compatability, resultants)

    @unpack ru, rθ = compatability
    @unpack F1, F2, M1, M2 = resultants

    # compatability equations
    irow = indices.irow_elem[ielem]
    resid[irow:irow+2] .= ru
    resid[irow+3:irow+5] .= rθ

    # equilibrium equations for the start of the beam element
    irow = indices.irow_point[assembly.start[ielem]]
    resid[irow:irow+2] .-= F1 ./ force_scaling
    resid[irow+3:irow+5] .-= M1 ./ force_scaling

    # equilibrium equations for the end of the beam element
    irow = indices.irow_point[assembly.stop[ielem]]
    resid[irow:irow+2] .+= F2 ./ force_scaling
    resid[irow+3:irow+5] .+= M2 ./ force_scaling

    return resid
end

"""
    static_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
        prescribed_conditions, distributed_loads, gravity)

Calculate and insert the residual entries corresponding to a beam element for a static 
analysis into the system residual vector.
"""
@inline function static_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
    prescribed_conditions, distributed_loads, gravity)

    properties = static_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity)

    compatability = compatability_residuals(properties)

    resultants = static_element_resultants(properties, distributed_loads, ielem)

    insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, compatability, resultants)

    return resid
end

"""
    steady_state_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
        prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

Calculate and insert the residual entries corresponding to a beam element for a steady state 
analysis into the system residual vector.
"""
@inline function steady_state_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
    prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

    properties = steady_state_element_properties(x, indices, force_scaling, assembly, ielem,
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    compatability = compatability_residuals(properties)

    resultants = dynamic_element_resultants(properties, distributed_loads, ielem)

    insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, compatability, resultants)

    return resid
end

"""
    initial_condition_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
        prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0, 
        u0, θ0, udot0, θdot0)

Calculate and insert the residual entries corresponding to a beam element for the 
initialization of a time domain simulation into the system residual vector.
"""
@inline function initial_condition_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
    prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
    u0, θ0, udot0, θdot0)

    properties = initial_condition_element_properties(x, indices, force_scaling, assembly, ielem,
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0, u0, θ0, udot0, θdot0)

    compatability = compatability_residuals(properties)

    resultants = dynamic_element_resultants(properties, distributed_loads, ielem)

    insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, compatability, resultants)

    return resid
end

"""
    newmark_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
        prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

Calculate and insert the residual entries corresponding to a beam element for a 
newmark-scheme time marching analysis into the system residual vector.
"""
@inline function newmark_element_residual!(resid, x, indices, force_scaling, assembly, ielem,  
    prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
    udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    properties = newmark_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0, 
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    compatability = compatability_residuals(properties)

    resultants = dynamic_element_resultants(properties, distributed_loads, ielem)

    insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, compatability, resultants)

    return resid
end

"""
    dynamic_element_residual!(resid, dx, x, indices, force_scaling, assembly, ielem,  
        prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

Calculate and insert the residual entries corresponding to a beam element for a dynamic
analysis into the system residual vector.
"""
@inline function dynamic_element_residual!(resid, dx, x, indices, force_scaling, assembly, ielem,  
    prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

    properties = dynamic_element_properties(dx, x, indices, force_scaling, assembly, ielem,
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    compatability = compatability_residuals(properties)

    resultants = dynamic_element_resultants(properties, distributed_loads, ielem)

    insert_element_residuals!(resid, indices, force_scaling, assembly, ielem, compatability, resultants)

    return resid
end

"""
    static_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity)

Calculate/extract the element properties needed to calculate the jacobian entries 
corresponding to a beam element for a static analysis
"""
@inline function static_element_jacobian_properties(properties, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, gravity)

    @unpack θ, C, S11, S12, S21, S22 = properties

    # strain and curvature
    γ_F = S11
    γ_M = S12
    κ_F = S21
    κ_M = S22

    # linear and angular displacement
    u1_u1, θ1_θ1 = point_displacement_jacobians(x, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    u2_u2, θ2_θ2 = point_displacement_jacobians(x, assembly.stop[ielem], indices.icol_point, prescribed_conditions)

    # rotation matrices
    C_θ1, C_θ2, C_θ3 = get_C_θ(C, θ)
    Ct_θ1, Ct_θ2, Ct_θ3 = C_θ1', C_θ2', C_θ3'
    Qinv_θ1, Qinv_θ2, Qinv_θ3 = get_Qinv_θ(θ)

    return (; properties..., u1_u1, u2_u2, θ1_θ1, θ2_θ2, C_θ1, C_θ2, C_θ3, Ct_θ1, Ct_θ2, Ct_θ3, 
        Qinv_θ1, Qinv_θ2, Qinv_θ3, γ_F, γ_M, κ_F, κ_M)
end

"""
    steady_state_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0))

Calculate/extract the element properties needed to calculate the jacobian entries 
corresponding to a element for a steady state analysis
"""
@inline function steady_state_element_jacobian_properties(properties, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    properties = static_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity)

    # linear and angular acceleration
    a_u = tilde(α0)

    return (; properties..., a_u)
end

"""
    initial_condition_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity)

Calculate/extract the element properties needed to calculate the jacobian entries 
corresponding to a element for a Newmark scheme time marching analysis
"""
@inline function initial_condition_element_jacobian_properties(properties, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, gravity)

    @unpack Cab, CtCab, S11, S12, S21, S22, v, ω = properties

    # strain and curvature
    γ_F = S11
    γ_M = S12
    κ_F = S21
    κ_M = S22

    # linear and angular velocity rates
    V1dot_V1dot, Ω1dot_Ω1dot = point_displacement_jacobians(x, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    V2dot_V2dot, Ω2dot_Ω2dot = point_displacement_jacobians(x, assembly.stop[ielem], indices.icol_point, prescribed_conditions)

    return (; properties..., γ_F, γ_M, κ_F, κ_M, V1dot_V1dot, Ω1dot_Ω1dot, V2dot_V2dot, Ω2dot_Ω2dot)
end

"""
    newmark_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0,
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

Calculate/extract the element properties needed to calculate the jacobian entries 
corresponding to a element for a Newmark scheme time marching analysis
"""
@inline function newmark_element_jacobian_properties(properties, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0,
    udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    properties = steady_state_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    @unpack Q, θ = properties

    # transformation matrices
    Q_θ1, Q_θ2, Q_θ3 = get_Q_θ(Q, θ)

    # velocity rates
    Vdot_V = 2/dt*I3
    Ωdot_Ω = 2/dt*I3

    # linear and angular acceleration
    a_V = Vdot_V
    α_Ω = Ωdot_Ω

    return (; properties..., Q_θ1, Q_θ2, Q_θ3, a_V, α_Ω) 
end

"""
    dynamic_element_jacobian_properties(properties, dx, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

Calculate/extract the element properties needed to calculate the jacobian entries 
corresponding to a element for a dynamic analysis
"""
@inline function dynamic_element_jacobian_properties(properties, dx, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    properties = steady_state_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    @unpack Q, θ = properties

    # transformation matrices
    Q_θ1, Q_θ2, Q_θ3 = get_Q_θ(Q, θ)

    return (; properties..., Q_θ1, Q_θ2, Q_θ3)
end

"""
    mass_matrix_element_jacobian_properties(assembly, ielem, x, indices, force_scaling, 
        prescribed_conditions)

Calculate/extract the element properties needed to calculate the mass matrix jacobian entries 
corresponding to a beam element
"""
@inline function mass_matrix_element_jacobian_properties(assembly, ielem, x, indices, force_scaling, 
    prescribed_conditions)

    # element properties
    @unpack L, Cab, compliance, mass = assembly.elements[ielem]
   
    # mass submatrices
    mass11 = mass[SVector{3}(1:3), SVector{3}(1:3)]
    mass12 = mass[SVector{3}(1:3), SVector{3}(4:6)]
    mass21 = mass[SVector{3}(4:6), SVector{3}(1:3)]
    mass22 = mass[SVector{3}(4:6), SVector{3}(4:6)]

    # linear and angular displacement
    u1, θ1 = point_displacement(x, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    u2, θ2 = point_displacement(x, assembly.stop[ielem], indices.icol_point, prescribed_conditions)
    u = (u1 + u2)/2
    θ = (θ1 + θ2)/2

    # transformation matrices
    C = get_C(θ)
    Ct = C'
    CtCab = Ct*Cab

    # linear and angular velocity rates
    V1dot_V1dot, Ω1dot_Ω1dot = point_displacement_jacobians(x, assembly.start[ielem], indices.icol_point, prescribed_conditions)
    V2dot_V2dot, Ω2dot_Ω2dot = point_displacement_jacobians(x, assembly.stop[ielem], indices.icol_point, prescribed_conditions)

    return (; L, C, Ct, Cab, CtCab, mass11, mass12, mass21, mass22, 
        V1dot_V1dot, Ω1dot_Ω1dot, V2dot_V2dot, Ω2dot_Ω2dot)
end

@inline function compatability_jacobians(properties)
   
    @unpack L, Cab, CtCab, Qinv, γ, κ, u1_u1, u2_u2, θ1_θ1, θ2_θ2, γ_F, γ_M, κ_F, κ_M, 
        Ct_θ1, Ct_θ2, Ct_θ3, Qinv_θ1, Qinv_θ2, Qinv_θ3, = properties

    Δu_θ = L*mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*(e1 + γ))
    Δu_F = L*CtCab*γ_F
    Δu_M = L*CtCab*γ_M

    Δθ_θ = L*mul3(Qinv_θ1, Qinv_θ2, Qinv_θ3, Cab*κ)
    Δθ_F = L*Qinv*Cab*κ_F
    Δθ_M = L*Qinv*Cab*κ_M

    ru_u1 = -u1_u1
    ru_u2 = u2_u2
    ru_θ1 = -1/2*Δu_θ*θ1_θ1
    ru_θ2 = -1/2*Δu_θ*θ2_θ2
    ru_F = -Δu_F
    ru_M = -Δu_M

    rθ_θ1 = -θ1_θ1 - 1/2*Δθ_θ*θ1_θ1
    rθ_θ2 = θ2_θ2 - 1/2*Δθ_θ*θ2_θ2
    rθ_F = -Δθ_F
    rθ_M = -Δθ_M

    return (; ru_u1, ru_u2, ru_θ1, ru_θ2, ru_F, ru_M, rθ_θ1, rθ_θ2, rθ_F, rθ_M)
end

@inline function initial_condition_compatability_jacobians(properties)
   
    @unpack L, Cab, CtCab, Qinv, γ_F, γ_M, κ_F, κ_M = properties

    Δu_F = L*CtCab*γ_F
    Δu_M = L*CtCab*γ_M

    Δθ_F = L*Qinv*Cab*κ_F
    Δθ_M = L*Qinv*Cab*κ_M

    ru_F = -Δu_F
    ru_M = -Δu_M

    rθ_F = -Δθ_F
    rθ_M = -Δθ_M

    return (; ru_F, ru_M, rθ_F, rθ_M)
end

"""
    static_element_resultant_jacobians(properties, distributed_loads, ielem)

Calculate the jacobians for the resultant loads applied at each end of a beam element 
for a static analysis.
"""
@inline function static_element_resultant_jacobians(properties, distributed_loads, ielem)

    jacobians = _static_element_resultant_jacobians(properties, distributed_loads, ielem)

    return finalize_static_element_resultant_jacobians(properties, jacobians)
end

@inline function _static_element_resultant_jacobians(properties, distributed_loads, ielem)

    @unpack L, Cab, CtCab, mass11, mass12, mass21, mass22, a, α, F, M, γ, κ, 
        C_θ1, C_θ2, C_θ3, Ct_θ1, Ct_θ2, Ct_θ3, γ_F, γ_M = properties

    # loads due to internal loads

    F1_u = @SMatrix zeros(3,3)
    F2_u = @SMatrix zeros(3,3)

    tmp = mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*F)
    F1_θ = tmp
    F2_θ = tmp

    F1_F = CtCab
    F2_F = CtCab

    M1_u = @SMatrix zeros(3,3)
    M2_u = @SMatrix zeros(3,3)

    tmp1 = mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*M)
    tmp2 = L/2*mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*cross(e1 + γ, F))
    M1_θ = tmp1 + tmp2
    M2_θ = tmp1 - tmp2

    tmp = L/2*CtCab*(tilde(e1 + γ) - tilde(F)*γ_F)
    M1_F = tmp
    M2_F = -tmp

    tmp = L/2*CtCab*tilde(F)*γ_M
    M1_M = -tmp + CtCab
    M2_M = tmp + CtCab

    # add loads due to linear and angular momentum rates (and gravity)
    Pdot_θ = mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*(mass11*CtCab'*a + mass12*CtCab'*α)) + 
        CtCab*mass11*Cab'*mul3(C_θ1, C_θ2, C_θ3, a) + 
        CtCab*mass12*Cab'*mul3(C_θ1, C_θ2, C_θ3, α)
    
    Hdot_θ = mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*(mass21*CtCab'*a + mass22*CtCab'*α)) + 
        CtCab*mass21*Cab'*mul3(C_θ1, C_θ2, C_θ3, a) + 
        CtCab*mass22*Cab'*mul3(C_θ1, C_θ2, C_θ3, α)
    
    tmp = L/2*Pdot_θ
    F1_θ -= tmp
    F2_θ += tmp

    tmp = L/2*Hdot_θ
    M1_θ -= tmp
    M2_θ += tmp

    # add distributed loads
    if haskey(distributed_loads, ielem)
        dload = distributed_loads[ielem]
        F1_θ += mul3(Ct_θ1, Ct_θ2, Ct_θ3, dload.f1_follower)
        F2_θ -= mul3(Ct_θ1, Ct_θ2, Ct_θ3, dload.f2_follower)
        M1_θ += mul3(Ct_θ1, Ct_θ2, Ct_θ3, dload.m1_follower)
        M2_θ -= mul3(Ct_θ1, Ct_θ2, Ct_θ3, dload.m2_follower)
    end

    return (; F1_u, F1_θ, F1_F, F2_u, F2_θ, F2_F,
        M1_u, M1_θ, M1_F, M1_M, M2_u, M2_θ, M2_F, M2_M)

end

@inline function finalize_static_element_resultant_jacobians(properties, jacobians)

    @unpack u1_u1, u2_u2, θ1_θ1, θ2_θ2 = properties

    @unpack F1_u, F1_θ, F2_u, F2_θ, M1_u, M1_θ, M2_u, M2_θ = jacobians

    F1_u1 = 1/2*F1_u*u1_u1
    F1_u2 = 1/2*F1_u*u2_u2
    F1_θ1 = 1/2*F1_θ*θ1_θ1
    F1_θ2 = 1/2*F1_θ*θ2_θ2

    F2_u1 = 1/2*F2_u*u1_u1
    F2_u2 = 1/2*F2_u*u2_u2
    F2_θ1 = 1/2*F2_θ*θ1_θ1
    F2_θ2 = 1/2*F2_θ*θ2_θ2

    M1_u1 = 1/2*M1_u*u1_u1
    M1_u2 = 1/2*M1_u*u2_u2
    M1_θ1 = 1/2*M1_θ*θ1_θ1
    M1_θ2 = 1/2*M1_θ*θ2_θ2

    M2_u1 = 1/2*M2_u*u1_u1
    M2_u2 = 1/2*M2_u*u2_u2
    M2_θ1 = 1/2*M2_θ*θ1_θ1
    M2_θ2 = 1/2*M2_θ*θ2_θ2

    return (; jacobians..., F1_u1, F1_u2, F1_θ1, F1_θ2, F2_u1, F2_u2, F2_θ1, F2_θ2,
        M1_u1, M1_u2, M1_θ1, M1_θ2, M2_u1, M2_u2, M2_θ1, M2_θ2,)
end

"""
    steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)

Calculate the jacobians for the resultant loads applied at each end of a beam element 
for a steady state analysis.
"""
@inline function steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)

    jacobians = _steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)

    return finalize_dynamic_element_resultant_jacobians(properties, jacobians)
end

@inline function _steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)

    jacobians = _static_element_resultant_jacobians(properties, distributed_loads, ielem)

    @unpack L, Cab, CtCab, mass11, mass12, mass21, mass22, V, Ω, v, ω, 
        C_θ1, C_θ2, C_θ3, Ct_θ1, Ct_θ2, Ct_θ3, a_u = properties

    @unpack F1_u, F1_θ, F2_u, F2_θ, M1_u, M1_θ, M2_u, M2_θ = jacobians

    # add loads due to linear and angular momentum rates
    Pdot_u = CtCab*mass11*CtCab'*a_u
    Hdot_u = CtCab*mass21*CtCab'*a_u

    tmp = L/2*Pdot_u
    F1_u -= tmp
    F2_u += tmp

    tmp = L/2*Hdot_u
    M1_u -= tmp
    M2_u += tmp

    # add loads due to linear and angular momentum
    P = CtCab*mass11*CtCab'*V + CtCab*mass12*CtCab'*Ω

    P_θ = mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*(mass11*CtCab'*V + mass12*CtCab'*Ω)) + 
        CtCab*mass11*Cab'*mul3(C_θ1, C_θ2, C_θ3, V) + 
        CtCab*mass12*Cab'*mul3(C_θ1, C_θ2, C_θ3, Ω)
    P_V = CtCab*mass11*CtCab'
    P_Ω = CtCab*mass12*CtCab'

    H_θ = mul3(Ct_θ1, Ct_θ2, Ct_θ3, Cab*(mass21*CtCab'*V + mass22*CtCab'*Ω)) + 
        CtCab*mass21*Cab'*mul3(C_θ1, C_θ2, C_θ3, V) + 
        CtCab*mass22*Cab'*mul3(C_θ1, C_θ2, C_θ3, Ω)
    H_V = CtCab*mass21*CtCab'
    H_Ω = CtCab*mass22*CtCab'

    tmp = L/2*tilde(ω)*P_θ
    F1_θ -= tmp
    F2_θ += tmp

    tmp = L/2*tilde(ω)*P_V
    F1_V = -tmp
    F2_V = tmp

    tmp = L/2*tilde(ω)*P_Ω
    F1_Ω = -tmp
    F2_Ω = tmp

    tmp = L/2*(tilde(ω)*H_θ + tilde(V)*P_θ)
    M1_θ -= tmp
    M2_θ += tmp

    tmp = L/2*(tilde(ω)*H_V + tilde(V)*P_V - tilde(P))
    M1_V = -tmp
    M2_V = tmp

    tmp = L/2*(tilde(ω)*H_Ω + tilde(V)*P_Ω)
    M1_Ω = -tmp
    M2_Ω = tmp

    return (; jacobians..., F1_u, F1_θ, F1_V, F1_Ω, F2_u, F2_θ, F2_V, F2_Ω, M1_u, M1_θ, M1_V, M1_Ω, M2_u, M2_θ, M2_V, M2_Ω)
end

@inline function finalize_dynamic_element_resultant_jacobians(properties, jacobians)

    jacobians = finalize_static_element_resultant_jacobians(properties, jacobians)

    @unpack F1_V, F1_Ω, F2_V, F2_Ω, M1_V, M1_Ω, M2_V, M2_Ω = jacobians

    F1_V1 = 1/2*F1_V
    F1_V2 = 1/2*F1_V
    F1_Ω1 = 1/2*F1_Ω
    F1_Ω2 = 1/2*F1_Ω

    F2_V1 = 1/2*F2_V
    F2_V2 = 1/2*F2_V
    F2_Ω1 = 1/2*F2_Ω
    F2_Ω2 = 1/2*F2_Ω

    M1_V1 = 1/2*M1_V
    M1_V2 = 1/2*M1_V
    M1_Ω1 = 1/2*M1_Ω
    M1_Ω2 = 1/2*M1_Ω

    M2_V1 = 1/2*M2_V
    M2_V2 = 1/2*M2_V
    M2_Ω1 = 1/2*M2_Ω
    M2_Ω2 = 1/2*M2_Ω

    return (; jacobians..., F1_V1, F1_V2, F1_Ω1, F1_Ω2, F2_V1, F2_V2, F2_Ω1, F2_Ω2,
        M1_V1, M1_V2, M1_Ω1, M1_Ω2, M2_V1, M2_V2, M2_Ω1, M2_Ω2,)
end

"""
    initial_condition_element_resultant_jacobians(properties)

Calculate the jacobians for the resultant loads applied at each end of a beam element 
for the initialization of a time domain analysis.
"""
@inline function initial_condition_element_resultant_jacobians(properties)

    @unpack L, CtCab, mass11, mass12, mass21, mass22, F, M, γ, κ, V, Ω, v, ω, γ_F, γ_M = properties

    # loads due to internal forces/moments
    F1_F = CtCab
    F2_F = CtCab

    tmp = L/2*CtCab*(tilde(e1 + γ) - tilde(F)*γ_F)
    M1_F = tmp
    M2_F = -tmp

    tmp = L/2*CtCab*tilde(F)*γ_M
    M1_M = -tmp + CtCab
    M2_M = tmp + CtCab

    # add loads due to linear and angular momentum rates (and gravity)
    Pdot_Vdot = CtCab*mass11*CtCab'
    Pdot_Ωdot = CtCab*mass12*CtCab'
    Hdot_Vdot = CtCab*mass21*CtCab'
    Hdot_Ωdot = CtCab*mass22*CtCab'
    
    tmp = L/2*Pdot_Vdot
    F1_Vdot = -tmp
    F2_Vdot = tmp

    tmp = L/2*Pdot_Ωdot
    F1_Ωdot = -tmp
    F2_Ωdot = tmp

    tmp = L/2*Hdot_Vdot
    M1_Vdot = -tmp 
    M2_Vdot = tmp

    tmp = L/2*Hdot_Ωdot
    M1_Ωdot = -tmp
    M2_Ωdot = tmp

    # add loads due to linear and angular momentum
    P = CtCab*mass11*CtCab'*V + CtCab*mass12*CtCab'*Ω

    P_V = CtCab*mass11*CtCab'
    P_Ω = CtCab*mass12*CtCab'

    H_V = CtCab*mass21*CtCab'
    H_Ω = CtCab*mass22*CtCab'

    tmp = L/2*tilde(ω)*P_V
    F1_V = -tmp
    F2_V = tmp

    tmp = L/2*tilde(ω)*P_Ω
    F1_Ω = -tmp
    F2_Ω = tmp

    tmp = L/2*(tilde(ω)*H_V + tilde(V)*P_V - tilde(P))
    M1_V = -tmp
    M2_V = tmp

    tmp = L/2*(tilde(ω)*H_Ω + tilde(V)*P_Ω)
    M1_Ω = -tmp
    M2_Ω = tmp

    # split jacobians between the two beam end points

    @unpack V1dot_V1dot, V2dot_V2dot, Ω1dot_Ω1dot, Ω2dot_Ω2dot = properties

    F1_V1dot = 1/2*F1_Vdot*V1dot_V1dot
    F1_V2dot = 1/2*F1_Vdot*V2dot_V2dot
    F1_Ω1dot = 1/2*F1_Ωdot*Ω1dot_Ω1dot
    F1_Ω2dot = 1/2*F1_Ωdot*Ω2dot_Ω2dot

    F2_V1dot = 1/2*F2_Vdot*V1dot_V1dot
    F2_V2dot = 1/2*F2_Vdot*V2dot_V2dot
    F2_Ω1dot = 1/2*F2_Ωdot*Ω1dot_Ω1dot
    F2_Ω2dot = 1/2*F2_Ωdot*Ω2dot_Ω2dot

    M1_V1dot = 1/2*M1_Vdot*V1dot_V1dot
    M1_V2dot = 1/2*M1_Vdot*V2dot_V2dot
    M1_Ω1dot = 1/2*M1_Ωdot*Ω1dot_Ω1dot
    M1_Ω2dot = 1/2*M1_Ωdot*Ω2dot_Ω2dot

    M2_V1dot = 1/2*M2_Vdot*V1dot_V1dot
    M2_V2dot = 1/2*M2_Vdot*V2dot_V2dot
    M2_Ω1dot = 1/2*M2_Ωdot*Ω1dot_Ω1dot
    M2_Ω2dot = 1/2*M2_Ωdot*Ω2dot_Ω2dot

    F1_V1 = 1/2*F1_V
    F1_V2 = 1/2*F1_V
    F1_Ω1 = 1/2*F1_Ω
    F1_Ω2 = 1/2*F1_Ω

    F2_V1 = 1/2*F2_V
    F2_V2 = 1/2*F2_V
    F2_Ω1 = 1/2*F2_Ω
    F2_Ω2 = 1/2*F2_Ω

    M1_V1 = 1/2*M1_V
    M1_V2 = 1/2*M1_V
    M1_Ω1 = 1/2*M1_Ω
    M1_Ω2 = 1/2*M1_Ω

    M2_V1 = 1/2*M2_V
    M2_V2 = 1/2*M2_V
    M2_Ω1 = 1/2*M2_Ω
    M2_Ω2 = 1/2*M2_Ω

    return (; 
        F1_V1dot, F1_V2dot, F1_Ω1dot, F1_Ω2dot, F1_F, F1_V1, F1_V2, F1_Ω1, F1_Ω2,
        F2_V1dot, F2_V2dot, F2_Ω1dot, F2_Ω2dot, F2_F, F2_V1, F2_V2, F2_Ω1, F2_Ω2,
        M1_V1dot, M1_V2dot, M1_Ω1dot, M1_Ω2dot, M1_F, M1_M, M1_V1, M1_V2, M1_Ω1, M1_Ω2,
        M2_V1dot, M2_V2dot, M2_Ω1dot, M2_Ω2dot, M2_F, M2_M, M2_V1, M2_V2, M2_Ω1, M2_Ω2
        )
end

"""
    newmark_element_resultant_jacobians(properties, distributed_loads, ielem)

Calculate the jacobians for the resultant loads applied at each end of a beam element 
for a Newmark scheme time marching analysis.
"""
@inline function newmark_element_resultant_jacobians(properties, distributed_loads, ielem)

    jacobians = _newmark_element_resultant_jacobians(properties, distributed_loads, ielem)

    return finalize_dynamic_element_resultant_jacobians(properties, jacobians)
end

@inline function _newmark_element_resultant_jacobians(properties, distributed_loads, ielem)

    jacobians = _steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)

    @unpack L, CtCab, mass11, mass12, mass21, mass22, a_V, α_Ω = properties
    
    @unpack F1_V, F1_Ω, F2_V, F2_Ω, M1_V, M1_Ω, M2_V, M2_Ω = jacobians

    # add loads due to linear and angular momentum rates (and gravity)
    Pdot_a = CtCab*mass11*CtCab'
    Pdot_α = CtCab*mass12*CtCab'
    Hdot_a = CtCab*mass21*CtCab'
    Hdot_α = CtCab*mass22*CtCab'

    tmp = L/2*Pdot_a*a_V
    F1_V -= tmp
    F2_V += tmp

    tmp = L/2*Pdot_α*α_Ω
    F1_Ω -= tmp
    F2_Ω += tmp

    tmp = L/2*Hdot_a*a_V
    M1_V -= tmp
    M2_V += tmp

    tmp = L/2*Hdot_α*α_Ω
    M1_Ω -= tmp
    M2_Ω += tmp

    return (; jacobians..., F1_V, F1_Ω, F2_V, F2_Ω, M1_V, M1_Ω, M2_V, M2_Ω)
end

"""
    dynamic_element_resultant_jacobians(properties, distributed_loads, ielem)

Calculate the jacobians for the resultant loads applied at each end of a beam element 
for a dynamic analysis.
"""
@inline function dynamic_element_resultant_jacobians(properties, distributed_loads, ielem)

    return steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)
end

"""
    mass_matrix_element_resultant_jacobians(properties)

Calculate the mass matrix jacobians for the resultant loads applied at each end of a beam element 
"""
@inline function mass_matrix_element_resultant_jacobians(properties)

    @unpack L, CtCab, mass11, mass12, mass21, mass22 = properties

    # add loads due to linear and angular momentum rates (and gravity)
    Pdot_Vdot = CtCab*mass11*CtCab'
    Pdot_Ωdot = CtCab*mass12*CtCab'
    Hdot_Vdot = CtCab*mass21*CtCab'
    Hdot_Ωdot = CtCab*mass22*CtCab'

    tmp = L/2*Pdot_Vdot
    F1_Vdot = -tmp
    F2_Vdot = tmp

    tmp = L/2*Pdot_Ωdot
    F1_Ωdot = -tmp
    F2_Ωdot = tmp

    tmp = L/2*Hdot_Vdot
    M1_Vdot = -tmp
    M2_Vdot = tmp

    tmp = L/2*Hdot_Ωdot
    M1_Ωdot = -tmp
    M2_Ωdot = tmp

    @unpack V1dot_V1dot, Ω1dot_Ω1dot, V2dot_V2dot, Ω2dot_Ω2dot = properties

    F1_V1dot = 1/2*F1_Vdot*V1dot_V1dot
    F1_V2dot = 1/2*F1_Vdot*V2dot_V2dot
    F1_Ω1dot = 1/2*F1_Ωdot*Ω1dot_Ω1dot
    F1_Ω2dot = 1/2*F1_Ωdot*Ω2dot_Ω2dot

    F2_V1dot = 1/2*F2_Vdot*V1dot_V1dot
    F2_V2dot = 1/2*F2_Vdot*V2dot_V2dot
    F2_Ω1dot = 1/2*F2_Ωdot*Ω1dot_Ω1dot
    F2_Ω2dot = 1/2*F2_Ωdot*Ω2dot_Ω2dot

    M1_V1dot = 1/2*M1_Vdot*V1dot_V1dot
    M1_V2dot = 1/2*M1_Vdot*V2dot_V2dot
    M1_Ω1dot = 1/2*M1_Ωdot*Ω1dot_Ω1dot
    M1_Ω2dot = 1/2*M1_Ωdot*Ω2dot_Ω2dot

    M2_V1dot = 1/2*M2_Vdot*V1dot_V1dot
    M2_V2dot = 1/2*M2_Vdot*V2dot_V2dot
    M2_Ω1dot = 1/2*M2_Ωdot*Ω1dot_Ω1dot
    M2_Ω2dot = 1/2*M2_Ωdot*Ω2dot_Ω2dot

    return (; F1_V1dot, F1_V2dot, F1_Ω1dot, F1_Ω2dot, 
        F2_V1dot, F2_V2dot, F2_Ω1dot, F2_Ω2dot, 
        M1_V1dot, M1_V2dot, M1_Ω1dot, M1_Ω2dot, 
        M2_V1dot, M2_V2dot, M2_Ω1dot, M2_Ω2dot)
end


@inline function insert_static_element_jacobians!(jacob, indices, force_scaling, 
    assembly, ielem, compatability, resultants)

    @unpack ru_u1, ru_u2, ru_θ1, ru_θ2, ru_F, ru_M, rθ_θ1, rθ_θ2, rθ_F, rθ_M = compatability

    @unpack F1_u1, F1_u2, F1_θ1, F1_θ2, F1_F,
        F2_u1, F2_u2, F2_θ1, F2_θ2, F2_F,
        M1_u1, M1_u2, M1_θ1, M1_θ2, M1_F, M1_M,
        M2_u1, M2_u2, M2_θ1, M2_θ2, M2_F, M2_M = resultants

    icol = indices.icol_elem[ielem]
    icol1 = indices.icol_point[assembly.start[ielem]]
    icol2 = indices.icol_point[assembly.stop[ielem]]

    # compatability equations
    irow = indices.irow_elem[ielem]

    jacob[irow:irow+2, icol1:icol1+2] .= ru_u1
    jacob[irow:irow+2, icol1+3:icol1+5] .= ru_θ1

    jacob[irow:irow+2, icol2:icol2+2] .= ru_u2
    jacob[irow:irow+2, icol2+3:icol2+5] .= ru_θ2

    jacob[irow:irow+2, icol:icol+2] .= ru_F .* force_scaling
    jacob[irow:irow+2, icol+3:icol+5] .= ru_M .* force_scaling

    jacob[irow+3:irow+5, icol1+3:icol1+5] .= rθ_θ1

    jacob[irow+3:irow+5, icol2+3:icol2+5] .= rθ_θ2

    jacob[irow+3:irow+5, icol:icol+2] .= rθ_F .* force_scaling
    jacob[irow+3:irow+5, icol+3:icol+5] .= rθ_M .* force_scaling
    
    # equilibrium equations for the start of the beam element
    irow1 = indices.irow_point[assembly.start[ielem]]

    jacob[irow1:irow1+2, icol1:icol1+2] .-= F1_u1 ./ force_scaling
    jacob[irow1:irow1+2, icol1+3:icol1+5] .-= F1_θ1 ./ force_scaling

    jacob[irow1:irow1+2, icol2:icol2+2] .-= F1_u2 ./ force_scaling
    jacob[irow1:irow1+2, icol2+3:icol2+5] .-= F1_θ2 ./ force_scaling

    jacob[irow1:irow1+2, icol:icol+2] .= -F1_F

    jacob[irow1+3:irow1+5, icol1:icol1+2] .-= M1_u1 ./ force_scaling
    jacob[irow1+3:irow1+5, icol1+3:icol1+5] .-= M1_θ1 ./ force_scaling

    jacob[irow1+3:irow1+5, icol2:icol2+2] .-= M1_u2 ./ force_scaling
    jacob[irow1+3:irow1+5, icol2+3:icol2+5] .-= M1_θ2 ./ force_scaling

    jacob[irow1+3:irow1+5, icol:icol+2] .= -M1_F
    jacob[irow1+3:irow1+5, icol+3:icol+5] .= -M1_M

    # equilibrium equations for the end of the beam element
    irow2 = indices.irow_point[assembly.stop[ielem]]

    jacob[irow2:irow2+2, icol1:icol1+2] .+= F2_u1 ./ force_scaling
    jacob[irow2:irow2+2, icol1+3:icol1+5] .+= F2_θ1 ./ force_scaling

    jacob[irow2:irow2+2, icol2:icol2+2] .+= F2_u2 ./ force_scaling
    jacob[irow2:irow2+2, icol2+3:icol2+5] .+= F2_θ2 ./ force_scaling

    jacob[irow2:irow2+2, icol:icol+2] .= F2_F

    jacob[irow2+3:irow2+5, icol1:icol1+2] .+= M2_u1 ./ force_scaling
    jacob[irow2+3:irow2+5, icol1+3:icol1+5] .+= M2_θ1 ./ force_scaling

    jacob[irow2+3:irow2+5, icol2:icol2+2] .+= M2_u2 ./ force_scaling
    jacob[irow2+3:irow2+5, icol2+3:icol2+5] .+= M2_θ2 ./ force_scaling

    jacob[irow2+3:irow2+5, icol:icol+2] .= M2_F
    jacob[irow2+3:irow2+5, icol+3:icol+5] .= M2_M

    return jacob
end

@inline function insert_initial_condition_element_jacobians!(jacob, indices, force_scaling, 
    assembly, ielem, compatability, resultants)

    @unpack ru_F, ru_M, rθ_F, rθ_M = compatability

    @unpack F1_V1dot, F1_V2dot, F1_Ω1dot, F1_Ω2dot, F1_F, F1_V1, F1_V2, F1_Ω1, F1_Ω2,
        F2_V1dot, F2_V2dot, F2_Ω1dot, F2_Ω2dot, F2_F, F2_V1, F2_V2, F2_Ω1, F2_Ω2,
        M1_V1dot, M1_V2dot, M1_Ω1dot, M1_Ω2dot, M1_F, M1_M, M1_V1, M1_V2, M1_Ω1, M1_Ω2,
        M2_V1dot, M2_V2dot, M2_Ω1dot, M2_Ω2dot, M2_F, M2_M, M2_V1, M2_V2, M2_Ω1, M2_Ω2 = resultants

    icol = indices.icol_elem[ielem]
    icol1 = indices.icol_point[assembly.start[ielem]]
    icol2 = indices.icol_point[assembly.stop[ielem]]

    # compatability equations
    irow = indices.irow_elem[ielem]

    jacob[irow:irow+2, icol:icol+2] .= ru_F .* force_scaling
    jacob[irow:irow+2, icol+3:icol+5] .= ru_M .* force_scaling

    jacob[irow+3:irow+5, icol:icol+2] .= rθ_F .* force_scaling
    jacob[irow+3:irow+5, icol+3:icol+5] .= rθ_M .* force_scaling
    
    # equilibrium equations for the start of the beam element
    irow1 = indices.irow_point[assembly.start[ielem]]

    jacob[irow1:irow1+2, icol1:icol1+2] .-= F1_V1dot ./ force_scaling
    jacob[irow1:irow1+2, icol1+3:icol1+5] .-= F1_Ω1dot ./ force_scaling

    jacob[irow1:irow1+2, icol2:icol2+2] .-= F1_V2dot ./ force_scaling
    jacob[irow1:irow1+2, icol2+3:icol2+5] .-= F1_Ω2dot ./ force_scaling

    jacob[irow1:irow1+2, icol:icol+2] .= -F1_F

    jacob[irow1:irow1+2, icol1+6:icol1+8] .-= F1_V1 ./ force_scaling
    jacob[irow1:irow1+2, icol1+9:icol1+11] .-= F1_Ω1 ./ force_scaling

    jacob[irow1:irow1+2, icol2+6:icol2+8] .-= F1_V2 ./ force_scaling
    jacob[irow1:irow1+2, icol2+9:icol2+11] .-= F1_Ω2 ./ force_scaling

    jacob[irow1+3:irow1+5, icol1:icol1+2] .-= M1_V1dot ./ force_scaling
    jacob[irow1+3:irow1+5, icol1+3:icol1+5] .-= M1_Ω1dot ./ force_scaling

    jacob[irow1+3:irow1+5, icol2:icol2+2] .-= M1_V2dot ./ force_scaling
    jacob[irow1+3:irow1+5, icol2+3:icol2+5] .-= M1_Ω2dot ./ force_scaling

    jacob[irow1+3:irow1+5, icol:icol+2] .= -M1_F
    jacob[irow1+3:irow1+5, icol+3:icol+5] .= -M1_M

    jacob[irow1+3:irow1+5, icol1+6:icol1+8] .-= M1_V1 ./ force_scaling
    jacob[irow1+3:irow1+5, icol1+9:icol1+11] .-= M1_Ω1 ./ force_scaling

    jacob[irow1+3:irow1+5, icol2+6:icol2+8] .-= M1_V2 ./ force_scaling
    jacob[irow1+3:irow1+5, icol2+9:icol2+11] .-= M1_Ω2 ./ force_scaling

    # equilibrium equations for the end of the beam element
    irow2 = indices.irow_point[assembly.stop[ielem]]

    jacob[irow2:irow2+2, icol1:icol1+2] .+= F2_V1dot ./ force_scaling
    jacob[irow2:irow2+2, icol1+3:icol1+5] .+= F2_Ω1dot ./ force_scaling

    jacob[irow2:irow2+2, icol2:icol2+2] .+= F2_V2dot ./ force_scaling
    jacob[irow2:irow2+2, icol2+3:icol2+5] .+= F2_Ω2dot ./ force_scaling

    jacob[irow2:irow2+2, icol:icol+2] .= F2_F

    jacob[irow2:irow2+2, icol1+6:icol1+8] .+= F2_V1 ./ force_scaling
    jacob[irow2:irow2+2, icol1+9:icol1+11] .+= F2_Ω1 ./ force_scaling

    jacob[irow2:irow2+2, icol2+6:icol2+8] .+= F2_V2 ./ force_scaling
    jacob[irow2:irow2+2, icol2+9:icol2+11] .+= F2_Ω2 ./ force_scaling

    jacob[irow2+3:irow2+5, icol1:icol1+2] .+= M2_V1dot ./ force_scaling
    jacob[irow2+3:irow2+5, icol1+3:icol1+5] .+= M2_Ω1dot ./ force_scaling

    jacob[irow2+3:irow2+5, icol2:icol2+2] .+= M2_V2dot ./ force_scaling
    jacob[irow2+3:irow2+5, icol2+3:icol2+5] .+= M2_Ω2dot ./ force_scaling

    jacob[irow2+3:irow2+5, icol:icol+2] .= M2_F
    jacob[irow2+3:irow2+5, icol+3:icol+5] .= M2_M

    jacob[irow2+3:irow2+5, icol1+6:icol1+8] .+= M2_V1 ./ force_scaling
    jacob[irow2+3:irow2+5, icol1+9:icol1+11] .+= M2_Ω1 ./ force_scaling

    jacob[irow2+3:irow2+5, icol2+6:icol2+8] .+= M2_V2 ./ force_scaling
    jacob[irow2+3:irow2+5, icol2+9:icol2+11] .+= M2_Ω2 ./ force_scaling

    return jacob
end

@inline function insert_dynamic_element_jacobians!(jacob, indices, force_scaling,
    assembly, ielem, compatability, resultants)

    insert_static_element_jacobians!(jacob, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

    @unpack F1_V1, F1_V2, F1_Ω1, F1_Ω2,
        F2_V1, F2_V2, F2_Ω1, F2_Ω2,
        M1_V1, M1_V2, M1_Ω1, M1_Ω2,
        M2_V1, M2_V2, M2_Ω1, M2_Ω2 = resultants

    icol1 = indices.icol_point[assembly.start[ielem]]
    icol2 = indices.icol_point[assembly.stop[ielem]]
    
    # equilibrium equations for the start of the beam element
    irow1 = indices.irow_point[assembly.start[ielem]]

    jacob[irow1:irow1+2, icol1+6:icol1+8] .-= F1_V1 ./ force_scaling
    jacob[irow1:irow1+2, icol1+9:icol1+11] .-= F1_Ω1 ./ force_scaling

    jacob[irow1:irow1+2, icol2+6:icol2+8] .-= F1_V2 ./ force_scaling
    jacob[irow1:irow1+2, icol2+9:icol2+11] .-= F1_Ω2 ./ force_scaling

    jacob[irow1+3:irow1+5, icol1+6:icol1+8] .-= M1_V1 ./ force_scaling
    jacob[irow1+3:irow1+5, icol1+9:icol1+11] .-= M1_Ω1 ./ force_scaling

    jacob[irow1+3:irow1+5, icol2+6:icol2+8] .-= M1_V2 ./ force_scaling
    jacob[irow1+3:irow1+5, icol2+9:icol2+11] .-= M1_Ω2 ./ force_scaling

    # equilibrium equations for the end of the beam element
    irow2 = indices.irow_point[assembly.stop[ielem]]

    jacob[irow2:irow2+2, icol1+6:icol1+8] .+= F2_V1 ./ force_scaling
    jacob[irow2:irow2+2, icol1+9:icol1+11] .+= F2_Ω1 ./ force_scaling

    jacob[irow2:irow2+2, icol2+6:icol2+8] .+= F2_V2 ./ force_scaling
    jacob[irow2:irow2+2, icol2+9:icol2+11] .+= F2_Ω2 ./ force_scaling

    jacob[irow2+3:irow2+5, icol1+6:icol1+8] .+= M2_V1 ./ force_scaling
    jacob[irow2+3:irow2+5, icol1+9:icol1+11] .+= M2_Ω1 ./ force_scaling

    jacob[irow2+3:irow2+5, icol2+6:icol2+8] .+= M2_V2 ./ force_scaling
    jacob[irow2+3:irow2+5, icol2+9:icol2+11] .+= M2_Ω2 ./ force_scaling

    return jacob
end

@inline function insert_element_mass_matrix_jacobians!(jacob, gamma, indices, force_scaling,
    assembly, ielem, resultants)

    @unpack F1_V1dot, F1_V2dot, F1_Ω1dot, F1_Ω2dot, 
        F2_V1dot, F2_V2dot, F2_Ω1dot, F2_Ω2dot, 
        M1_V1dot, M1_V2dot, M1_Ω1dot, M1_Ω2dot, 
        M2_V1dot, M2_V2dot, M2_Ω1dot, M2_Ω2dot = resultants

    icol1 = indices.icol_point[assembly.start[ielem]]
    icol2 = indices.icol_point[assembly.stop[ielem]]
    
    # equilibrium equations for the start of the beam element
    irow1 = indices.irow_point[assembly.start[ielem]]

    jacob[irow1:irow1+2, icol1+6:icol1+8] .-= F1_V1dot .* gamma ./ force_scaling
    jacob[irow1:irow1+2, icol1+9:icol1+11] .-= F1_Ω1dot .* gamma ./ force_scaling

    jacob[irow1:irow1+2, icol2+6:icol2+8] .-= F1_V2dot .* gamma ./ force_scaling
    jacob[irow1:irow1+2, icol2+9:icol2+11] .-= F1_Ω2dot .* gamma ./ force_scaling

    jacob[irow1+3:irow1+5, icol1+6:icol1+8] .-= M1_V1dot .* gamma ./ force_scaling
    jacob[irow1+3:irow1+5, icol1+9:icol1+11] .-= M1_Ω1dot .* gamma ./ force_scaling

    jacob[irow1+3:irow1+5, icol2+6:icol2+8] .-= M1_V2dot .* gamma ./ force_scaling
    jacob[irow1+3:irow1+5, icol2+9:icol2+11] .-= M1_Ω2dot .* gamma ./ force_scaling

    # equilibrium equations for the end of the beam element
    irow2 = indices.irow_point[assembly.stop[ielem]]

    jacob[irow2:irow2+2, icol1+6:icol1+8] .+= F2_V1dot .* gamma ./ force_scaling
    jacob[irow2:irow2+2, icol1+9:icol1+11] .+= F2_Ω1dot .* gamma ./ force_scaling

    jacob[irow2:irow2+2, icol2+6:icol2+8] .+= F2_V2dot .* gamma ./ force_scaling
    jacob[irow2:irow2+2, icol2+9:icol2+11] .+= F2_Ω2dot .* gamma ./ force_scaling

    jacob[irow2+3:irow2+5, icol1+6:icol1+8] .+= M2_V1dot .* gamma ./ force_scaling
    jacob[irow2+3:irow2+5, icol1+9:icol1+11] .+= M2_Ω1dot .* gamma ./ force_scaling

    jacob[irow2+3:irow2+5, icol2+6:icol2+8] .+= M2_V2dot .* gamma ./ force_scaling
    jacob[irow2+3:irow2+5, icol2+9:icol2+11] .+= M2_Ω2dot .* gamma ./ force_scaling

    return jacob
end

"""
    static_element_jacobian!(jacob, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, distributed_loads, gravity)

Calculate and insert the jacobian entries corresponding to a beam element for a static 
analysis into the system jacobian matrix.
"""
@inline function static_element_jacobian!(jacob, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, distributed_loads, gravity)

    properties = static_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity)

    properties = static_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity)

    compatability = compatability_jacobians(properties)

    resultants = static_element_resultant_jacobians(properties, distributed_loads, ielem)

    insert_static_element_jacobians!(jacob, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

    return jacob
end

"""
    steady_state_element_jacobian!(jacob, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

Calculate and insert the jacobian entries corresponding to a beam element for a steady state 
analysis into the system jacobian matrix.
"""
@inline function steady_state_element_jacobian!(jacob, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

    properties = steady_state_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    properties = steady_state_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    compatability = compatability_jacobians(properties)

    resultants = steady_state_element_resultant_jacobians(properties, distributed_loads, ielem)

    insert_dynamic_element_jacobians!(jacob, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

    return jacob
end

"""
    initial_condition_element_jacobian!(jacob, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
        u0, θ0, udot0, θdot0)

Calculate and insert the jacobian entries corresponding to a beam element for the 
initialization of a time domain analysis into the system jacobian matrix.
"""
@inline function initial_condition_element_jacobian!(jacob, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
    u0, θ0, udot0, θdot0)

    properties = initial_condition_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0, u0, θ0, udot0, θdot0)

    properties = initial_condition_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity)

    compatability = initial_condition_compatability_jacobians(properties)

    resultants = initial_condition_element_resultant_jacobians(properties)

    insert_initial_condition_element_jacobians!(jacob, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

    return jacob
end

"""
    newmark_element_jacobian!(jacob, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

Calculate and insert the jacobian entries corresponding to a beam element for a Newmark-scheme
time marching analysis into the system jacobian matrix.
"""
@inline function newmark_element_jacobian!(jacob, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0,
    udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    properties = newmark_element_properties(x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0, 
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    properties = newmark_element_jacobian_properties(properties, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0,
        udot_init, θdot_init, Vdot_init, Ωdot_init, dt)

    compatability = compatability_jacobians(properties)

    resultants = newmark_element_resultant_jacobians(properties, distributed_loads, ielem)

    insert_dynamic_element_jacobians!(jacob, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

    return jacob
end

"""
    dynamic_element_jacobian!(jacob, dx, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, distributed_loads, gravity, 
        x0, v0, ω0, a0, α0)

Calculate and insert the jacobian entries corresponding to a beam element for a dynamic
analysis into the system jacobian matrix.
"""
@inline function dynamic_element_jacobian!(jacob, dx, x, indices, force_scaling, 
    assembly, ielem, prescribed_conditions, distributed_loads, gravity, x0, v0, ω0, a0, α0)

    properties = dynamic_element_properties(dx, x, indices, force_scaling, assembly, ielem, 
        prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    properties = dynamic_element_jacobian_properties(properties, dx, x, indices, force_scaling, 
        assembly, ielem, prescribed_conditions, gravity, x0, v0, ω0, a0, α0)

    compatability = compatability_jacobians(properties)

    resultants = dynamic_element_resultant_jacobians(properties, distributed_loads, ielem)

    insert_dynamic_element_jacobians!(jacob, indices, force_scaling, assembly, ielem, 
        compatability, resultants)

    return jacob
end

"""
    element_mass_matrix!(jacob, gamma, x, indices, force_scaling, assembly, 
        ielem, prescribed_conditions)

Calculate and insert the mass_matrix jacobian entries corresponding to a beam element into 
the system jacobian matrix.
"""
@inline function element_mass_matrix!(jacob, gamma, x, indices, force_scaling, assembly, 
    ielem, prescribed_conditions)

    properties = mass_matrix_element_jacobian_properties(assembly, ielem, x, indices, 
        force_scaling, prescribed_conditions)

    resultants = mass_matrix_element_resultant_jacobians(properties)
    
    insert_element_mass_matrix_jacobians!(jacob, gamma, indices, force_scaling, assembly, ielem, 
        resultants)

    return jacob
end