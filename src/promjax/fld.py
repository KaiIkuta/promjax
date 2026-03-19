import jax
import jax.numpy as jnp
from jax import jit, lax

@jit
def time_step(w, v_full, dx_full, t_current):
    w_even = w[0::2]
    v_even = v_full[0::2]


    t_eval = jnp.maximum(t_current - dt, 0.0)
    f_even, s_even = get_flux(w_even, v_even, t_eval)
    dx_odd = dx_full[1::2]

    df = f_even[1:] - f_even[:-1]
    s_avg = 0.5 * (s_even[1:] + s_even[:-1])
    w_avg = 0.5 * (w_even[1:] + w_even[:-1])
    w_odd_new = w_avg - (dt / dx_odd[:, None]) * df + dt * s_avg

    v_odd = 0.5 * (v_even[1:] + v_even[:-1])


    f_odd, s_odd = get_flux(w_odd_new, v_odd, t_current)
    dx_main = dx_full[2:-1:2]

    term1 = 0.5 * (f_even[2:] - f_even[:-2])
    term2 = f_odd[1:] - f_odd[:-1]
    flux_term = term1 + term2 #0.5

    s_main = s_even[1:-1]
    dx_minus = dx_full[1:-2:2][:, None]
    dx_plus  = dx_full[3::2][:, None]
    dx_center = dx_main[:, None]

    source_avg = 0.5 * (s_odd[1:] * dx_minus + s_odd[:-1] * dx_plus) / dx_center
    source_term = 0.5 * dt * (s_main + source_avg)

    #Artificial Viscosity


    w_ratio_even = w_even[..., 1] / w_even[..., 0]
    vel_diff = jnp.abs(w_ratio_even[2:] - w_ratio_even[:-2])

  
    qv = av * dx_main * (jnp.maximum(0.5 * vel_diff, dv) - dv)


    qv_padded = jnp.pad(qv, (1, 1), mode='constant')


    w_inner = w_even[1:-1]


    w_inner_new = w_inner - (0.5 * dt / dx_center) * flux_term + source_term

    visc_term1 = (qv_padded[2:, None] + qv_padded[1:-1, None]) * (w_even[2:] - w_even[1:-1]) / dx_plus
    visc_term2 = (qv_padded[:-2, None] + qv_padded[1:-1, None]) * (w_even[1:-1] - w_even[:-2]) / dx_minus



    viscosity = (0.5 * dt / dx_center) * (visc_term1 - visc_term2)

    w_inner_new = w_inner_new + viscosity

    #Boundary conditions
    w_new_odd = w_odd_new

    h0, _, _, _ = get_parameters(v_even[0], t_current)
    h4, _, _, _ = get_parameters(v_even[2], t_current)
    hr = h4 / h0
    w_0 = w_inner_new[1] * hr
    w_0 = w_0.at[1].set(-w_0[1])

    w_2 = w_inner_new[0]
    w_2_val = w_2[2] - 0.5 * w_2[1]**2 / w_2[0]
    w_2 = w_2.at[1].set(0.0)
    w_2 = w_2.at[2].set(w_2_val)

    h_end, _, _, _ = get_parameters(v_even[-1], t_current)
    h_prev, _, _, _ = get_parameters(v_even[-2], t_current)
    hr_end = h_prev / h_end
    w_end = w_inner_new[-1] * hr_end

    w_out = w.at[0::2].set(jnp.concatenate([
        w_0[None, :],
        w_inner_new,
        w_end[None, :]
    ], axis=0))

    w_out = w_out.at[2].set(w_2)
    w_out = w_out.at[1::2].set(w_new_odd)

    return w_out
