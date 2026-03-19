import jax
import jax.numpy as jnp
from jax import jit, lax

@jit
def get_parameters(v, t):
    u0 = jnp.arccos((yint**2 - a**2) / (yint**2 + a**2))
    cons = -vmax / a * (1.0 - jnp.cos(u0)) / (jnp.sin(u0)**2)
    u = pi / 2.0 - jnp.arctan(jnp.cos(u0)/jnp.sin(u0) - cons*t)
    h = (jnp.cosh(v) - jnp.cos(u)) / a
    vn = -cons * jnp.sin(u)**2 / h
    px = jnp.sinh(v) / h
    py = jnp.sin(u) / h
    b = jnp.sqrt(rs**2 - a**2)
    pxd = px - a*(1 - jnp.cos(beta)) - b*jnp.sin(beta)
    pyd = py + a*jnp.sin(beta) + b*jnp.cos(beta)
    r = jnp.sqrt(pxd**2 + pyd**2)
    g = gm * (pxd * (jnp.cosh(v) - px * jnp.sinh(v) / a) + pyd * (-px * jnp.sin(u)) / a)
    g = g / (r**3)
    return h, u, vn, g

@jit
def get_hydro_initial(v):
    h, u, vn, g = get_parameters(v, 0.0)
    au_cond = v <= (fil_t + fil_b) / 2.0
    au = jnp.where(au_cond,
                   0.5 * (jnp.tanh(500 * (v - fil_t)) + 1) + 0.005,
                   0.5 * (jnp.tanh(500 * (-(v - fil_b))) + 1) + 0.005)
    au = jnp.clip(au, 0.01, 1.0)
    rho0 = rhos * au
    rho0 = rho0 / h * jnp.sin(u)**2
    rho1 = jnp.zeros_like(rho0)
    rho2 = rhos / h / (gam - 1) * jnp.sin(u)**2
    return jnp.stack([rho0, rho1, rho2], axis=-1)

@jit
def get_flux(w, v, t):
    w0 = w[..., 0]
    w1 = w[..., 1]
    w2 = w[..., 2]
    vs = w1 / w0
    pa = (gam - 1) * (w2 - 0.5 * w0 * vs**2)
    h, u, vn, grav = get_parameters(v, t)
    rn = jnp.sinh(v) / a
    r_geom = jnp.sin(u) / a
    f0 = w0 * vs * h
    f1 = (w0 * vs**2 + pa) * h
    f2 = (gam / (gam - 1) * pa + 0.5 * w0 * vs**2) * vs * h
    f = jnp.stack([f0, f1, f2], axis=-1)
    s0 = w0 * vs * rn - w0 * vn * r_geom
    s1 = w0 * vs**2 * rn - w0 * (grav + vn * (vn * rn) + 2 * vs * vn * r_geom)
    s2 = (0.5 * w0 * vs**3 * rn +
          pa / (gam - 1) * (gam * vs * rn - (2 * gam - 1) * vn * r_geom) -
          w0 * vs * (grav + vn * (vn * rn) + 1.5 * vs * vn * r_geom) +
          pa * vn * h * 2.0 * jnp.cos(u) / jnp.sin(u))
    s = jnp.stack([s0, s1, s2], axis=-1)
    return f, s

@jit
def get_cfl(w, v, t, dx_local):
    w0 = w[..., 0]
    w1 = w[..., 1]
    w2 = w[..., 2]
    vs = w1 / w0
    pa = (gam - 1) * (w2 - 0.5 * w0 * vs**2)
    pa = jnp.maximum(pa, 1e-10)
    c = jnp.sqrt(gam * pa / w0)
    h, _, _, _ = get_parameters(v, t)
    sp = (c + jnp.abs(vs)) * h * dt / dx_local
    return jnp.max(sp)
