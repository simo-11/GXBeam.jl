# unit vectors and identity matrix
const e1 = SVector(1, 0, 0)
const e2 = SVector(0, 1, 0)
const e3 = SVector(0, 0, 1)
const I3 = @SMatrix [1 0 0; 0 1 0; 0 0 1]
const Z3 = @SMatrix zeros(3,3)

"""
    tilde(x)

Construct the cross product operator matrix
"""
@inline tilde(x) = @SMatrix [0 -x[3] x[2]; x[3] 0 -x[1]; -x[2] x[1] 0]

"""
    wiener_milenkovic(c)

Construct a Wiener-Milenkovic transformation matrix, given the three Wiener-Milenkovic
parameters in `c`.

Note that the corresponding rotation matrix is the transpose of this
transformation matrix.
"""
@inline function wiener_milenkovic(c)

    c0 = 2 - c'*c/8

    C = 1/(4-c0)^2*(@SMatrix [
        c0^2 + c[1]^2 - c[2]^2 - c[3]^2  2*(c[1]*c[2] + c0*c[3])          2*(c[1]*c[3]-c0*c[2]);
        2*(c[1]*c[2] - c0*c[3])          c0^2 - c[1]^2 + c[2]^2 - c[3]^2  2*(c[2]*c[3] + c0*c[1]);
        2*(c[1]*c[3] + c0*c[2])          2*(c[2]*c[3] - c0*c[1])          c0^2 - c[1]^2 - c[2]^2 + c[3]^2
        ]
    )

    return C
end

"""
    transform_properties(K, T)

Applies the transformation `T` to the stiffness or mass matrix `K`
"""
transform_properties(X, T) = [T' Z3; Z3 T'] * X * [T Z3; Z3 T]

"""
    rotation_parameter_scaling(θ)

Extract a scaling parameter which may be multiplied by the angular parameters
to yield the Wiener-Milenkovic rotation parameters.  Use of this scaling
parameter allows deflections greater than 360 degrees.
"""
@inline function rotation_parameter_scaling(θ)

    θ = real(θ)

    θnorm = sqrt(θ'*θ)

    scaling = 1

    m = div(div(θnorm, 4) + 1, 2) # number of times to add/subtract

    if θnorm > 4
        scaling -= 8*m/θnorm
    end

    return scaling
end

@inline function rotation_parameter_scaling_θ(θ)
    
    return rotation_parameter_scaling_θ(rotation_parameter_scaling(θ), θ)
end

@inline function rotation_parameter_scaling_θ(scaling, θ)
    
    return ifelse(isone(scaling), zero(θ), (1 - scaling)*θ/(θ'*θ))
end

@inline function rotation_parameter_scaling_θ_θ(θ)
    
    return rotation_parameter_scaling_θ_θ(rotation_parameter_scaling(θ), θ)
end

@inline function rotation_parameter_scaling_θ_θ(scaling, θ)
    
    return ifelse(isone(scaling), (@SMatrix zeros(eltype(θ),3,3)), (1 - scaling)/(θ'*θ)*(I - 3*θ*θ'/(θ'*θ)))
end
        
"""
    get_C(θ)

Returns the transformation matrix `C` given the three angular parameters in `θ`.
"""
@inline function get_C(θ)

    scaling = rotation_parameter_scaling(θ)

    c = scaling*θ

    return wiener_milenkovic(c)
end

"""
    get_C_θ([C, ] θ)

Calculate the derivative of the Wiener-Milenkovic transformation matrix `C` with
respect to each of the rotation parameters in `θ`.
"""
@inline get_C_θ(c) = get_C_θ(get_C(c), c)

@inline function get_C_θ(C, θ)

    scaling = rotation_parameter_scaling(θ)
    scaling_θ = rotation_parameter_scaling_θ(scaling, θ)

    c = scaling*θ
    c_θ = scaling_θ*θ' + scaling*I3

    c0 = 2 - c'*c/8
    c0_c = -c/4

    tmp = 1/(4-c0)^2
    tmp_c = 2/(4-c0)^3*c0_c

    C_c1 = tmp_c[1]*C/tmp + tmp*(@SMatrix [
        2*c0*c0_c[1] + 2*c[1]    2*(c[2] + c0_c[1]*c[3]) 2*(c[3] - c0_c[1]*c[2]);
         2*(c[2] - c0_c[1]*c[3])  2*c0*c0_c[1] - 2*c[1]   2*(c0_c[1]*c[1] + c0);
         2*(c[3] + c0_c[1]*c[2]) -2*(c0_c[1]*c[1] + c0)   2*c0*c0_c[1] - 2*c[1]
        ]
    )

    C_c2 = tmp_c[2]*C/tmp + tmp*(@SMatrix [
        2*c0*c0_c[2] - 2*c[2]   2*(c[1] + c0_c[2]*c[3]) -2*(c0_c[2]*c[2] + c0);
        2*(c[1] - c0_c[2]*c[3]) 2*c0*c0_c[2] + 2*c[2]    2*(c[3] + c0_c[2]*c[1]);
         2*(c0_c[2]*c[2] + c0)   2*(c[3] - c0_c[2]*c[1])  2*c0*c0_c[2] - 2*c[2]
        ]
    )

    C_c3 = tmp_c[3]*C/tmp + tmp*(@SMatrix [
         2*c0*c0_c[3] - 2*c[3]   2*(c0_c[3]*c[3] + c0)    2*(c[1] - c0_c[3]*c[2]);
        -2*(c0_c[3]*c[3] + c0)   2*c0*c0_c[3] - 2*c[3]    2*(c[2] + c0_c[3]*c[1]);
         2*(c[1] + c0_c[3]*c[2]) 2*(c[2] - c0_c[3]*c[1])  2*c0*c0_c[3] + 2*c[3]
        ]
    )

    C_θ1 = C_c1*c_θ[1,1] + C_c2*c_θ[2,1] + C_c3*c_θ[3,1]
    C_θ2 = C_c1*c_θ[1,2] + C_c2*c_θ[2,2] + C_c3*c_θ[3,2]
    C_θ3 = C_c1*c_θ[1,3] + C_c2*c_θ[2,3] + C_c3*c_θ[3,3]

    return C_θ1, C_θ2, C_θ3
end

"""
    get_C_t([C, ] θ, θ_t)

Calculate the derivative of the Wiener-Milenkovic transformation matrix `C` with
respect to the scalar parameter `t`. `θ_t` is the derivative of the angular parameter
`θ` with respect to `t`.
"""
@inline get_C_t(θ, θ_t) = get_C_t(get_C(θ), θ, θ_t)

@inline function get_C_t(C, θ, θ_t)

    scaling = rotation_parameter_scaling(θ)
    scaling_θ = rotation_parameter_scaling_θ(scaling, θ)

    c = scaling*θ
    c_t = scaling_θ'*θ_t*θ + scaling*θ_t

    c0 = 2 - c'*c/8
    c0dot = -c'*c_t/4
    tmp = 1/(4-c0)^2
    tmpdot = 2/(4-c0)^3*c0dot

    Cdot = tmpdot*C/tmp + tmp*(@SMatrix [
        2*c0*c0dot + 2*c[1]*c_t[1] - 2*c[2]*c_t[2] - 2*c[3]*c_t[3] (
        2*(c_t[1]*c[2] + c[1]*c_t[2] + c0dot*c[3] + c0*c_t[3])) (
        2*(c_t[1]*c[3] + c[1]*c_t[3] - c0dot*c[2] - c0*c_t[2]));
        2*(c_t[1]*c[2] + c[1]*c_t[2] - c0dot*c[3] - c0*c_t[3]) (
        2*c0*c0dot - 2*c[1]*c_t[1] + 2*c[2]*c_t[2] - 2*c[3]*c_t[3]) (
        2*(c_t[2]*c[3] + c[2]*c_t[3] + c0dot*c[1] + c0*c_t[1]));
        2*(c_t[1]*c[3] + c[1]*c_t[3] + c0dot*c[2] + c0*c_t[2]) (
        2*(c_t[2]*c[3] + c[2]*c_t[3] - c0dot*c[1] - c0*c_t[1])) (
        2*c0*c0dot - 2*c[1]*c_t[1] - 2*c[2]*c_t[2] + 2*c[3]*c_t[3])]
    )

    return Cdot
end

"""
    get_C_t_θ(θ, θ_t)

Calculate the derivative of the time derivative of the Wiener-Milenkovic
transformation matrix `C` with respect to `θ`.
"""
@inline function get_C_t_θ(θ, θ_t)

    θ1, θ2, θ3 = θ
    C_t_θ1 = ForwardDiff.derivative(θ1 -> get_C_t(SVector(θ1, θ2, θ3), θ_t), θ1)
    C_t_θ2 = ForwardDiff.derivative(θ2 -> get_C_t(SVector(θ1, θ2, θ3), θ_t), θ2)
    C_t_θ3 = ForwardDiff.derivative(θ3 -> get_C_t(SVector(θ1, θ2, θ3), θ_t), θ3)

    return C_t_θ1, C_t_θ2, C_t_θ3
end

"""
    get_C_t_θdot(θ)

Calculate the derivative of the time derivative of the Wiener-Milenkovic
transformation matrix `C` with respect to each of the time derivatives of `θ`.
"""
@inline function get_C_t_θdot(θ)

    Cdot_θdot1 = ForwardDiff.derivative(cdot1 -> GXBeam.get_C_t(θ, SVector(cdot1, 0, 0)), 0)
    Cdot_θdot2 = ForwardDiff.derivative(cdot2 -> GXBeam.get_C_t(θ, SVector(0, cdot2, 0)), 0)
    Cdot_θdot3 = ForwardDiff.derivative(cdot3 -> GXBeam.get_C_t(θ, SVector(0, 0, cdot3)), 0)
    
    return Cdot_θdot1, Cdot_θdot2, Cdot_θdot3
end

"""
    get_Q(θ)

Calculate the matrix Q as defined in the paper "Geometrically nonlinear analysis
of composite beams using Wiener-Milenković parameters" by Qi Wang and Wenbin Yu
given the rotational parameters in `θ`.
"""
@inline function get_Q(θ)

    scaling = rotation_parameter_scaling(θ)

    c = scaling*θ

    c0 = 2 - c'*c/8

    return 1/(4-c0)^2*((4 - 1/4*c'*c)*I - 2*tilde(c) + 1/2*c*c')
end

"""
    get_Q_θ(θ)
    get_Q_θ(Q, θ)

Calculate the derivative of the matrix `Q` with respect to each of the rotation
parameters in `θ`.
"""
@inline get_Q_θ(θ) = get_Q_θ(get_Q(θ), θ)

@inline function get_Q_θ(Q, θ)

    scaling = rotation_parameter_scaling(θ)
    scaling_θ = rotation_parameter_scaling_θ(scaling, θ)

    c = scaling*θ
    c_θ = scaling_θ*θ' + scaling*I3

    c0 = 2 - c'*c/8
    c0_c = -c/4

    tmp = 1/(4-c0)^2
    tmp_c = 2/(4-c0)^3*c0_c

    Q_c1 = tmp_c[1]*Q/tmp + tmp*(-c[1]/2*I - 2*tilde(e1) + (e1*c' + c*e1')/2)

    Q_c2 = tmp_c[2]*Q/tmp + tmp*(-c[2]/2*I - 2*tilde(e2) + (e2*c' + c*e2')/2)

    Q_c3 = tmp_c[3]*Q/tmp + tmp*(-c[3]/2*I - 2*tilde(e3) + (e3*c' + c*e3')/2)

    Q_θ1 = Q_c1*c_θ[1,1] + Q_c2*c_θ[2,1] + Q_c3*c_θ[3,1]
    Q_θ2 = Q_c1*c_θ[1,2] + Q_c2*c_θ[2,2] + Q_c3*c_θ[3,2]
    Q_θ3 = Q_c1*c_θ[1,3] + Q_c2*c_θ[2,3] + Q_c3*c_θ[3,3]

    return Q_θ1, Q_θ2, Q_θ3
end

"""
    get_Qinv(θ)

Calculate the matrix inverse `Qinv` as defined in the paper "Geometrically
nonlinear analysis of composite beams using Wiener-Milenković parameters" by
Qi Wang and Wenbin Yu given the rotational parameters in `θ`.
"""
@inline function get_Qinv(θ)

    scaling = rotation_parameter_scaling(θ)

    c = scaling*θ

    Qinv = (1 - 1/16*c'*c)*I3 + 1/2*tilde(c) + 1/8*c*c'

    return Qinv
end

"""
    get_Qinv_θ(θ)

Calculate the derivative of the matrix inverse `Qinv` with respect to each of the
rotation parameters in `θ`.
"""
@inline function get_Qinv_θ(θ)

    scaling = rotation_parameter_scaling(θ)
    scaling_θ = rotation_parameter_scaling_θ(scaling, θ)

    c = scaling*θ
    c_θ = scaling_θ*θ' + scaling*I3

    Qinv_c1 = -c[1]/8*I3 + 1/2*tilde(e1) + 1/8*(e1*c' + c*e1')
    Qinv_c2 = -c[2]/8*I3 + 1/2*tilde(e2) + 1/8*(e2*c' + c*e2')
    Qinv_c3 = -c[3]/8*I3 + 1/2*tilde(e3) + 1/8*(e3*c' + c*e3')

    Qinv_θ1 = Qinv_c1*c_θ[1,1] + Qinv_c2*c_θ[2,1] + Qinv_c3*c_θ[3,1]
    Qinv_θ2 = Qinv_c1*c_θ[1,2] + Qinv_c2*c_θ[2,2] + Qinv_c3*c_θ[3,2]
    Qinv_θ3 = Qinv_c1*c_θ[1,3] + Qinv_c2*c_θ[2,3] + Qinv_c3*c_θ[3,3]

    return Qinv_θ1, Qinv_θ2, Qinv_θ3
end

"""
    mul3(A_1, A_2, A_3, b)

Return the product of a 3x3x3 tensor represented by `A_1`, `A_2`, and `A_3` with
the vector `b`.
"""
@inline mul3(A_1, A_2, A_3, b) = hcat(A_1*b, A_2*b, A_3*b)

"""
    gauss_quadrature(f, a, b)

Default gauss-quadrature function used for integrating distributed loads.
"""
@inline function gauss_quadrature(f, a, b)
    h = b - a
    c = (a + b)/2
    x = h/2*GAUSS_NODES .+ c
    return h/2*GAUSS_WEIGHTS'*f.(x)
end

# this function is not used
# @inline function linf_norm_scaling(A)
#
#     droptol!(A, eps(eltype(A)))
#
#     m, n = size(A)
#     rows = rowvals(A)
#     vals = nonzeros(A)
#
#     # initialize outputs
#     rowmax = fill(typemin(eltype(A)), n)
#     rowmin = fill(typemax(eltype(A)), n)
#     @inbounds for j = 1:n
#         @inbounds for i in nzrange(A, j)
#            row = rows[i]
#            val = abs(vals[i])
#            # perform sparse wizardry...
#            if val > rowmax[row]
#                rowmax[row] = val
#            end
#            if val < rowmin[row]
#                rowmin[row] = val
#            end
#         end
#     end
#
#     # use storage already allocated for rowmax for r
#     r = rowmax
#     @inbounds for i = 1:m
#         if i in rows
#             r[i] = nextpow(2, 1/sqrt(rowmax[i]*rowmin[i]))
#         else
#             r[i] = 1
#         end
#     end
#
#     # use storage already allocated for rowmin for s
#     s = rowmin
#     @inbounds for j = 1:n
#         colrange = nzrange(A, j)
#         if isempty(colrange)
#             s[j] = 1
#         else
#             colmax = maximum(i -> r[rows[i]]*abs(vals[i]), colrange)
#             colmin = minimum(i -> r[rows[i]]*abs(vals[i]), colrange)
#             s[j] = nextpow(2, 1/sqrt(colmax*colmin))
#         end
#     end
#
#     return r, s
# end
