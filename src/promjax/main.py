import os
import time
import numpy as np
import jax.numpy as jnp

from .param import hs, cs, gam, get_config
from .mesh import mesh
from .fcn import get_hydro_initial, get_cfl
from .fld import time_step


def run_simulation(params=None, save_dir=None, fn=None):

    cfg, meta = get_config(params)

    grid_size = meta["grid_size"]
    nmax = meta["nmax"]
    per = meta["per"]
    fac = meta["fac"]
    dt = cfg["dt"]

    print("Simulation Start (JAX/GPU)")
    print(f"Grid size: {grid_size}, Max steps: {nmax}")
    print(
        f"Parameters: vtop={meta['vtop']}, "
        f"vmin={cfg['vmin']}, vmax={cfg['vmax']}, "
        f"yint={meta['yint']}, a={meta['a']}"
    )

    v_full, dx_full = mesh(cfg, meta)
    w = get_hydro_initial(v_full, cfg)

    start_time = time.time()
    t_current = 0.0
    history = []

    print("Compiling JIT functions...")
    _ = time_step(w, v_full, dx_full, 0.0, cfg)
    print("Compilation finished.")

    current_w = w
    for n in range(0, int(nmax) + 1):
        if n % per == 0:
            elapsed = time.time() - start_time
            cfl_val = float(get_cfl(current_w[0::2], v_full[0::2], t_current, dx_full[0::2], cfg))
            mv = float(jnp.max(current_w[:, 1] / current_w[:, 0]))
            print(
                f"Step {n:8d} | t={t_current*hs/cs:.4f} | "
                f"CFL={cfl_val:.4f} | MaxV={mv:.4f} | Time={elapsed:.2f}s"
            )

            history.append((
                t_current,
                np.array(current_w[:, 0]),
                np.array(current_w[:, 1] / current_w[:, 0]),
                np.array(
                    (gam - 1.0) * (current_w[:, 2] - 0.5 * current_w[:, 1]**2 / current_w[:, 0])
                ),
            ))

        current_w = time_step(current_w, v_full, dx_full, t_current, cfg)
        t_current += dt

    t_data = np.array([h[0] for h in history])
    rho_data = np.array([h[1] for h in history])
    vs_data = np.array([h[2] for h in history])
    p_data = np.array([h[3] for h in history])
    v_data = np.array(v_full[2::8])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, fn + "_t.csv"), t_data, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_rho.csv"), rho_data[:, 2::8*fac], delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_us.csv"), vs_data[:, 2::8*fac], delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_p.csv"), p_data[:, 2::8*fac], delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_v.csv"), v_data[::fac], delimiter=",")
        print(f"Saved CSV files to: {save_dir}")

    return t_data, rho_data, vs_data, p_data, v_data
