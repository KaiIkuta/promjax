import jax
import jax.numpy as jnp
from jax import jit, lax

jax.config.update("jax_enable_x64", True)

vm = 500.0
fac = 1
imax = 120000 * fac + 1
dn = 0.0001000 / fac / 4.0
dt = 0.0001 / fac / 10.0 / 4.0
per = 1000 * fac * 10 *4
nmax = 2000 * 125 * fac* 7* 3 * 8

hs = 606452.168665969
cs = 12895.319009743993
rs = (695700000.0 / hs)
gm = (1.3271244e20 / (cs * cs * hs))
gam = 5.0 / 3.0
pi = jnp.pi
fil_t = 0.35
fil_b = 0.75
vmax = (vm * 1000.0 / cs)
yint = (60000.0 * 1000.0 / hs)
a = (25000.0 * 1000.0 / hs)
rhos = 1.0
beta = 0.0 * pi / 180.0
av = 1.0
dv = 0.001

i_idx = jnp.arange(imax + 1, dtype=jnp.float64)
grid_size = 2 * imax + 1
dx_grid = jnp.ones(grid_size, dtype=jnp.float64) * dn
