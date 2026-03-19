import jax
import jax.numpy as jnp
from jax import jit, lax

def mesh():
    x_idx = jnp.arange(imax + 1, dtype=jnp.float64)
    x_val = -dn + x_idx * dn

    dx_full = jnp.zeros(grid_size, dtype=jnp.float64)

    x_mid = 0.5 * (x_val[1:-1] + x_val[2:])
    dx_odd_vals = jnp.cosh(x_mid) / a


    dx_full = dx_full.at[3:2*imax:2].set(dx_odd_vals)


    val_term = (yint**2 - a**2)
    u_val = val_term / (2.0 * yint * a)


    u_param = pi / 2.0 - jnp.arctan(u_val)


    sum_dx = jnp.sum(dx_full)
    dw = sum_dx - (imax - 1) / a * jnp.cos(u_param)

    dx_full = dx_full - (1.0 / a * jnp.cos(u_param))
    dx_full = dx_full / dw
    dx_full = dx_full * (imax - 1) * dn

    dx_full = dx_full.at[1].set(dx_full[3])
    dx_full = dx_full.at[0].set(dx_full[2])
    dx_full = dx_full.at[2*imax].set(dx_full[2*(imax-1)])


    dx_odd_L = dx_full[1:2*imax-1:2] 
    dx_odd_R = dx_full[3:2*imax+1:2] 
    dx_even_vals = 0.5 * (dx_odd_L + dx_odd_R)


    dx_full = dx_full.at[2:2*imax:2].set(dx_even_vals)

    v_coarse = jnp.zeros(imax + 1, dtype=jnp.float64)


    dx_for_accum = dx_full[3:2*imax+1:2]


    v_cum = jnp.cumsum(dx_for_accum)


    v_start = 0.0


    v_coarse = v_coarse.at[1].set(v_start)
    v_coarse = v_coarse.at[2:2+len(v_cum)].set(v_start + v_cum)


    v_coarse = v_coarse.at[0].set(-v_coarse[2])

    residual = 3.0 - v_coarse[imax]
    dx_full = dx_full.at[2*imax-1].add(residual)

    v_coarse = v_coarse.at[imax].set(3.0)



    v_full = jnp.zeros(grid_size, dtype=jnp.float64)


    v_full = v_full.at[0::2].set(v_coarse)

    return v_full

    # 奇数インデックス: 中点補間
    # w(2*i+1) は 0.5*(v(i)+v(i+1)) の位置にあるとみなす
    v_mid = 0.5 * (v_coarse[:-1] + v_coarse[1:])
    v_full = v_full.at[1::2].set(v_mid)
