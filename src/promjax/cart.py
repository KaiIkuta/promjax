import os
import numpy as np
from .param import pi, get_config


def calc_post_quantities(t_data, v_data, vs_data, params=None, save_dir=None, fn=None):

    cfg, _ = get_config(params)
    yint = cfg["yint"]
    a = cfg["a"]
    vtop = cfg["vtop"]
    pi_val = float(pi)

    t_arr = np.asarray(t_data)
    v_arr = np.asarray(v_data)
    vs_arr = np.asarray(vs_data)
    if t_arr.ndim == 1 and v_arr.ndim == 1 and vs_arr.ndim == 2:
        t = t_arr[:, None]  # shape: (N_t, 1)
        v = v_arr[None, :]  # shape: (1, N_v)
        is_grid = True
    else:
        t = t_arr
        v = v_arr
        is_grid = False

    u0 = np.arccos((yint**2 - a**2) / (yint**2 + a**2))
    C = -vtop / a * (1.0 - np.cos(u0)) / (np.sin(u0)**2)

    u = pi_val / 2.0 - np.arctan(-C * t + np.cos(u0) / np.sin(u0))
    h = (np.cosh(v) - np.cos(u)) / a
    vn = -C * (np.sin(u)**2) / h

    tan = 1.0 - np.cosh(v) * np.cos(u)
    cos = np.sinh(v) * np.sin(u)
    fac = np.sqrt(1.0 / (cos * cos + tan * tan))

    vx = fac * (tan * vs_arr + cos * Vn)
    vy = fac * (-cos * vs_arr + tan * Vn)

    x = np.sinh(v) / h
    y = np.sin(u) / h


    u_out = u[:, 0] if is_grid else u

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, fn + "_vx.csv"), vx, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_vy.csv"), vy, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_vn.csv"), vn, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_h.csv"), h, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_x.csv"), x, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_y.csv"), y, delimiter=",")
        np.savetxt(os.path.join(save_dir, fn + "_u.csv"), u_out, delimiter=",")
        print(f"Saved post-processed CSV files to: {save_dir}")

    return vx, vy, vn, h, x, y, u_out
