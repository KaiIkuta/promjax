import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


hs = 606452.168665969
cs = 12895.319009743993
rs = 695700000.0 / hs
gm = 1.3271244e20 / (cs * cs * hs)
gam = 5.0 / 3.0
pi = jnp.pi


DEFAULT_BASE_PARAMS = {
    "vtop": 150.0,    # km/s
    "fac": 2,
    "vmin": 0.8,
    "vmax": 1.2,
    "yint": 50000.0,  # km
    "a": 12500.0,     # km
    "beta": 0.0,      # deg
    "rhos": 1.0,
    "av": 1.0,
    "dv": 0.001,
}


def get_config(params=None):

    p = DEFAULT_BASE_PARAMS.copy()
    if params is not None:
        p.update(params)

    fac = int(p["fac"])
    imax = int(p.get("imax", 120000 * fac + 1))
    dn = float(p.get("dn", 0.0001000 / 4.0 / fac))
    dt = float(p.get("dt", 0.0001 / fac / 10.0 / 4.0))
    per = int(p.get("per", 1000 * fac * 10 * 4))
    nmax = int(p.get("nmax", 2000 * 125 * fac * 7 * 3 * 4))
    grid_size = int(p.get("grid_size", 2 * imax + 1))


    vtop_norm = float(p["vtop"] * 1000.0 / cs)
    yint_norm = float(p["yint"] * 1000.0 / hs)
    a_norm = float(p["a"] * 1000.0 / hs)
    beta_rad = float(p["beta"] * float(pi) / 180.0)


    jax_cfg = {
        "vmin": float(p["vmin"]),
        "vmax": float(p["vmax"]),
        "vtop": vtop_norm,
        "yint": yint_norm,
        "a": a_norm,
        "rhos": float(p["rhos"]),
        "beta": beta_rad,
        "av": float(p["av"]),
        "dv": float(p["dv"]),
        "dt": dt,
        "dn": dn,
    }


    meta_cfg = {
        "vtop": float(p["vtop"]),
        "yint": float(p["yint"]),
        "a": float(p["a"]),
        "beta": float(p["beta"]),
        "fac": fac,
        "imax": imax,
        "per": per,
        "nmax": nmax,
        "grid_size": grid_size,
    }

    return jax_cfg, meta_cfg

DEFAULT_JAX_CFG, DEFAULT_META_CFG = get_config()
