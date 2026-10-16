import jax.numpy as jnp
from .param import pi, DEFAULT_JAX_CFG, DEFAULT_META_CFG


def mesh(cfg=DEFAULT_JAX_CFG, meta=DEFAULT_META_CFG):
    imax = meta["imax"]
    grid_size = meta["grid_size"]
    dn, a, yint = cfg["dn"], cfg["a"], cfg["yint"]


    x_mid = (jnp.arange(imax - 1, dtype=jnp.float64) + 0.5) * dn
    u_param = pi / 2.0 - jnp.arctan((yint**2 - a**2) / (2.0 * yint * a))

    dx_inner = (jnp.cosh(x_mid) - jnp.cos(u_param)) / a
    dx_inner = dx_inner / jnp.sum(dx_inner) * (imax - 1) * dn


    v_cum = jnp.cumsum(dx_inner)
    residual = 3.0 - v_cum[-1]
    dx_inner = dx_inner.at[-1].add(residual)
    v_cum = v_cum.at[-1].set(3.0)

    v_coarse = jnp.concatenate([jnp.array([-v_cum[0], 0.0]), v_cum])

    # 3. 全体グリッド (v_full, dx_full) への展開
    dx_odd = jnp.concatenate([dx_inner[:1], dx_inner])
    dx_even_inner = 0.5 * (dx_odd[:-1] + dx_odd[1:])
    dx_even = jnp.concatenate([dx_even_inner[:1], dx_even_inner, dx_even_inner[-1:]])

    dx_full = jnp.zeros(grid_size, dtype=jnp.float64)
    dx_full = dx_full.at[0::2].set(dx_even)
    dx_full = dx_full.at[1::2].set(dx_odd)

    v_full = jnp.zeros(grid_size, dtype=jnp.float64)
    v_full = v_full.at[0::2].set(v_coarse)
    v_full = v_full.at[1::2].set(0.5 * (v_coarse[:-1] + v_coarse[1:]))

    return v_full, dx_full
