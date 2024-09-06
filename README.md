## ChemPy
Chemical Kinetics code in Python, with option to use ODE solver compatible with PyTorch, namely [torchode](https://github.com/martenlienen/torchode) by Lienen & Gunnemann (2022).

---
### What?

This is a 0D chemical kinetics code, written in Python3. It solves the set of ordinary differential equations (ODEs) that make up the chemical network.

The code is based on the legacy chemistry codes by the [UMIST](http://udfa.ajmarkwick.net/index.php?mode=downloads) Database of Astrochemsitry:
- The dark cloud chemical model source code: rate13_dc_code.tgz
- The circumstellar envelope chemical model source code: rate13_cse_code.tgz, see also [Rate22-CSE code](https://github.com/MarieVdS/rate22_cse_code).

The purpose is to use this code to generate data to build a chemistry emulator.

Written by [S. Maes](https://github.com/silkemaes) & [F. De Ceuster](https://github.com/FredDeCeuster). 2023.

---
### How to run

Run one 0D model:
1. Manually put input values in script ```/scr/input.py```
2. Run script ```/scr/main.py```
    ```
    cd src/
    ```
    ```
    python main.py
    ```

---
### Updates

- 06.09.'24

    The version using torchode does not work. The version using a SciPy ODE solver works.

- 02.08.'23

    Self shielding doesn't work --> causes and infinite loop somehow... In comment in file ```/src/rates.py```.

---
### Possible future work

- Write code using object oriented programming (currently not the case, sorry!).
- Fix self-shielding of CO and N2.
- Fix compatibility with torchode.

---
### Notes

- Self-shielding: using [shielding tables](https://home.strw.leidenuniv.nl/~ewine/photo/CO_photodissociation.html)

From [table](CO shielding --> [tabellen](https://home.strw.leidenuniv.nl/~ewine/photo/CO_photodissociation.html) by Visser et al. (2009).

If we assume that the dust extinction in the AGB wind is similar to the dust extinction in the ISM (so dust has similar properties), we can use this to determine the column densities of H$_2$, CO and N$_2$, without the need of any distances.
[Predehl & Smitt (2995)](https://articles.adsabs.harvard.edu/pdf/1995A%26A...293..889P) found that for the ISM the column density of H$_2$ can be determined by $$N_{\rm H_2} = A_V \, 1.87\times 10^{21}, $$
with $N_{\rm H_2}$ in units of atoms cm$^{-2}$ and $A_V$ in units of mag.

For CO and $N_2$ we will multiply this ($N_{\rm H_2}$) with there parent abundance. We can do this, since the abundances of these two parent species barely vary, relatively speaking.

Rewrite the shielding tables in a more convenient way compared to online:
- Tables are real tables now
- Axis in a separate file: 
	- Rows: $N({\rm H_2})$
	- Columns: $N({\rm CO})$ or $N({\rm N_2})$
- File name:
	- ${\rm CO}$: $\texttt{COshield.H2velocity[km/s].H2temp[K].13C/12C-ratio.dat}$
	- ${\rm N_2}$: $\texttt{N2shield.H2velocity[km/s].1eH2temp[K].N(H)[cm**-2].dat}$

- ODEs in script ```/src/ode/*``` literally copied from the fortran ODEs of [Rate22-CSE code](https://github.com/MarieVdS/rate22_cse_code), using ```ODEs-to-python.ipynb```.

---

### Rate equations 
*More info; see [PhD thesis](https://fys.kuleuven.be/ster/pub/phd-thesis-silke-maes/phd-thesis-silke-maes) Silke Maes, Chap. 3*
- ***Two body:*** $$k=\alpha\left(\frac{T}{300}\right)^\beta\exp\left(-\frac{\gamma}{T}\right)$$ in ${\rm cm^3 \, s^{-1}}$
- ***Cosmic rays***
	- (CP) Direct ionisation: $$k=\alpha$$ in ${\rm s^{-1}}$
	- (CR) Induced photoreaction: $$k = \alpha\left(\frac{T}{300}\right)^\beta\frac{\gamma}{1-w}$$ in ${\rm s^{-1}}$
- ***(PH) Photodissociation:*** $$k = \alpha\, \xi \exp (-\gamma A_V)$$ in ${\rm s^{-1}}$, where $\delta = {\rm RAD}$ in the [Rate22-CSE code](https://github.com/MarieVdS/rate22_cse_code). 

