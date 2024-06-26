# HIILines
HIILines is an analytical model for lines emitted by the ionized interstellar medium (ISM). It currently covers [OIII], [OII], $H_\alpha$, and $H_\beta$ lines. A detailed review of HIILines can be found in [Yang et al. 2023a](https://arxiv.org/abs/2304.09261).  
## Model overview
Given the incident spectrum shape $L(\nu)$ and amplitude $Q_\mathrm{HI}$ (hydrogen ionizing photon generation rate $Q_\mathrm{HI}=\int_{13.6 \mathrm{eV}}^\infty\dfrac{L}{h\nu}d\nu$), together with characteristic HII region gas density $n_\mathrm{H}$, metallicity $Z$, and temperature $T$, HIILines assumes that the HII region is uniform and spherically symmetric. It first solves the HII, OIII, and OII region volumes assuming ionization-recombination balance among HI, HII, HeI, HeII, OI, OII, OIII. The OIII and OII level populations are then solved assuming all ions have achieved steady state. Finally, HIILines analytically calculates the [OIII], [OII], $H_\alpha$, and $H_\beta$ line luminosities.  
Comparing with other spectral synthesis code, the strength of HIILines is its high computational efficiency. It can be used for: 
* Galaxy spectroscopic survey measurement interpolations assuming a one-zone picture (e.g. [Yang et al. 2020](https://academic.oup.com/mnras/article/499/3/3417/5913327)).
* Galaxy line emission measurement design and forecasts (e.g. [Yang et al. 2021](https://academic.oup.com/mnras/article/504/1/723/6207947)).
* Post-processing of hydrodynamical galaxy formation simulations for ISM emission lines (e.g. [Yang et al. 2023a](https://arxiv.org/abs/2304.09261)).  

HIILines currently lacks models for dust absorption effects, so its application is limited to intrinsic line luminosity calculations.
## Scripts overview
`const.py`: Stores physical constants and constants for unit transfer.  
`const_ion.py`: Stores radiative transfer constants.  
`HIILines.py`: Define function 'assignL' for solving ISM line luminosities, and 'solV.py' for solving OIII and OII region volumes.  
`SolveHII.ipynb`: Example scripts. The first jupyter notebook cell shows an example where the user inputs stellar radiation spectrum shape, amplitude, gas density, metallicity, temperature, and HIILines outputs luminosities of various [OIII], [OII], and Balmer line luminosities. The second cell shows an example where the user uses a pre-tabulated $V_\mathrm{OIII}/V_\mathrm{HII}$ lookup table to skip the OIII region volume correction factor calculation, which will significantly speed up the subsequent steps for computing line luminosities.   
`strongLineDiagnostic.py`: functions for constraining HII region metallicities with {R2,O3O2}, {R3,O3O2}, or {R23,O3O2}. N2O2 is optional for breaking degeneracies among multiple metallcity solutions.   
`StrongLineDiagnostics.ipynb`: Example scripts for strong line diagnostic methods published in [Yang et al. 2023b](https://arxiv.org/abs/2312.09213)  

Please contact Shengqi Yang (shengqiy@lanl.gov) for questions.
