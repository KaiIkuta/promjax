# promjax: Prominence model with 1D hydrodynamics


```bash
pip install git+https://github.com/KaiIkuta/promjax.git
```


```python
import promjax
from promjax import run_simulation, calc_post_quantities

params = { #Case for Filament eruption on EK Dra (Ikuta & Shibata 2024)
    "a": 25000.0, #km
    "yint": 350000.0, #km
    "vtop": 550.0, #km/s
    "vmin": 0.055, 
    "vmax": 0.105,
    "tmax": 7200.0 #sec
}


t_data, rho_data, vs_data, p_data, v_data = run_simulation(
    params=params,
    save_dir=SAVE_DIRECTORY,
    fn=FILE_NAME
)

vx, vy, vn, h, x, y, u = calc_post_quantities(
    t_data, v_data, vs_data,
    params=params,
    save_dir=SAVE_DIRECTORY,
    fn=FILE_NAME
)
```

## Filament Model


Simple model for self-similarly expanding magnetic loop with 1D hydrodynamics ([Ikuta & Shibata 2024, ApJ, 963, 50](https://ui.adsabs.harvard.edu/abs/2024ApJ...963...50I/abstract))

## Prominence Model

Application to a prominence eruption on solar-type star ([Namekata, Ikuta et al. 2024, ApJ, 976, 255](https://ui.adsabs.harvard.edu/abs/2024ApJ...976..255N/abstract))

Application to the solar prominence eruptions ([Ikuta et al., in preparation](https://ui.adsabs.harvard.edu/abs/2024csss.confE.100I/abstract))


### Erratum and Remarks (as of 24/04/04)

Page 2: In Equation 4, adiabatic index $\gamma=5/3$
