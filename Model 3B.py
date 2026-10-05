"""
Model 3B: Skin Tissue Diffusion & Proteolytic Decay
====================================================

Governing PDE (wiki Eq. 35).  Tissue domain x in [0, L_tissue], which is the
mirror image of the wiki's -L_tissue <= x <= 0 (x = 0 is the gel/tissue
interface, x = L_tissue is the deep boundary toward the systemic circulation):

    dP/dt = D_eff,tissue * d^2P/dx^2 - (k_deg,tissue + k_clearance) * P

Boundary conditions:
    Eq. 36 (x = 0):  [P]_gel(0,t) = Kp * [P]_tissue(0,t)
                     ->  P(0,t) = [P]_gel(0,t) / Kp
    Eq. 38 (x = L):  P(L,t) = 0    (complete clearance into systemic circulation)

    Eq. 37 (molar flux matching at x = 0) needs the gel-side diffusivity
    D_eff,gel.  Until the gel model is ready, the gel is treated as a
    well-mixed reservoir at [P]_gel(0,t) = P_gel_surface(t), which satisfies
    Eq. 37 trivially (no gel-side gradient).  TODO: couple to the gel model.

Units: x [cm], t [s], P [mol/cm^3], D_eff [cm^2/s], k [1/s].

Numerics: concentrations are solved in dimensionless form theta = P/P0 with
P0 = [P]_gel(0,t)/Kp, so solver tolerances are scale-independent; results are
scaled back for plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# 3.2.1 Geometry
# ---------------------------------------------------------------------------
L_tissue = 0.15  # wound bed + viable skin thickness [cm] (= 1500 um)

# ---------------------------------------------------------------------------
# Gel / tissue interface, wiki Eq. 36:  [P]_gel = Kp * [P]_tissue at x = 0
# ---------------------------------------------------------------------------
Kp = 100  # partition coefficient [P]_gel / [P]_tissue at x = 0 [-] (Eq. 36)

def P_gel_surface(t):
    """[P]_gel(0,t): peptide concentration in the gel at the interface [mol/cm^3].

    TODO: replace the constant below by the output of the gel-release model
    (then [P]_gel(0,t) may become time-dependent and Eq. 37 coupling kicks in).
    """
    c_0 = 1.0e-7  # gel surface concentration [mol/cm^3] = 0.1 mM (placeholder)
    return c_0

# ---------------------------------------------------------------------------
# Tissue structure: inflamed, edematous chronic-wound bed
# ---------------------------------------------------------------------------
epsilon_tissue = 0.8  # extracellular volume fraction (porosity) [-]
tau_tissue = epsilon_tissue ** (-0.5)  # tortuosity [-]
# -> D_eff = D0 * eps/tau = D0 * eps^1.5  (Bruggeman-type hindered diffusion)

# ---------------------------------------------------------------------------
# Species parameters
# ---------------------------------------------------------------------------
# Variant 14 (engineered peptide)
M_v14 = 1355.59             # molecular weight [Da]
t_half_v14 = 1.9 * 3600     # PROTEOLYTIC half-life in tissue (e.g. MMPs) [s]
                            # (not the plasma half-life -> no double counting
                            #  with k_clearance)
k_clearance_v14 = 3.5e-3   # systemic microvascular clearance [1/s]

# FGF2-G3
M_fgf = 16487.87            # molecular weight [Da]
t_half_fgf = 168 * 3600     # PROTEOLYTIC half-life in tissue [s]
k_clearance_fgf = 1.15e-4   # systemic microvascular clearance [1/s]

# ---------------------------------------------------------------------------
# Stokes-Einstein in bulk water at 37 C, then hindered to the tissue (D_eff)
# ---------------------------------------------------------------------------
k_B = 1.38e-23  # Boltzmann constant [J/K]
T = 310.15      # temperature [K] (37 C)
eta = 6.9e-4    # viscosity of WATER at 37 C [Pa*s = kg/(m*s)]

def stokes_einstein_cm2s(M):
    """D0 in bulk water at 37 C [cm^2/s] for a molecule of weight M [Da].

    Hydrodynamic radius from Erickson's rule: r_h = 0.066 * M^(1/3) [nm].
    r_h must be converted nm -> m for the SI formula; the result m^2/s is
    converted to cm^2/s (1 m^2/s = 1e4 cm^2/s) because x is in cm.
    """
    r_h_m = 0.066 * M ** (1 / 3) * 1e-9        # [nm] -> [m]
    D0_m2s = k_B * T / (6 * np.pi * eta * r_h_m)  # [m^2/s]
    return D0_m2s * 1e4                        # [m^2/s] -> [cm^2/s]

D0_v14 = stokes_einstein_cm2s(M_v14)
D0_fgf = stokes_einstein_cm2s(M_fgf)

def calc_derived_params(D0, t_half, k_clearance):
    """Tissue-hindered diffusivity [cm^2/s] and total decay rate [1/s]."""
    D_eff = D0 * (epsilon_tissue / tau_tissue)   # [cm^2/s]
    k_deg = np.log(2) / t_half                   # proteolysis only [1/s]
    k_total = k_deg + k_clearance                # [1/s]
    return D_eff, k_total

D_eff_v14, k_total_v14 = calc_derived_params(D0_v14, t_half_v14, k_clearance_v14)
D_eff_fgf, k_total_fgf = calc_derived_params(D0_fgf, t_half_fgf, k_clearance_fgf)

# ---------------------------------------------------------------------------
# Spatial grid (x = 0 at the gel interface, x = L_tissue at the deep boundary)
# ---------------------------------------------------------------------------
N = 100
x = np.linspace(0.0, L_tissue, N)   # [cm]
dx = x[1] - x[0]                    # [cm]

def pde_system(t, u, D_eff, k_total):
    """RHS of Eq. 35 for the N-2 INTERIOR nodes, in dimensionless form.

    u[i] = theta = P/P0 at x[i+1].  Both boundary values are Dirichlet and
    enter the Laplacian directly (no mutation of the solver state).
    """
    P = np.empty(N)
    P[0] = 1.0        # Eq. 36: theta(0,t) = 1 (scaled Dirichlet)
    P[1:-1] = u
    P[-1] = 0.0       # Eq. 38: complete clearance at the deep boundary

    d2P = np.zeros(N)
    d2P[1:-1] = (P[2:] - 2.0 * P[1:-1] + P[:-2]) / dx**2

    dudt = D_eff * d2P[1:-1] - k_total * u   # [1/s]
    return dudt

# ---------------------------------------------------------------------------
# Simulation (12 h)
# ---------------------------------------------------------------------------
t_max = 12 * 3600                         # [s]
t_eval = np.linspace(0.0, t_max, 200)     # [s]
u0 = np.zeros(N - 2)                      # no peptide in the tissue at t = 0

sol_v14 = solve_ivp(lambda t, u: pde_system(t, u, D_eff_v14, k_total_v14),
                    (0.0, t_max), u0, t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-10)
sol_fgf = solve_ivp(lambda t, u: pde_system(t, u, D_eff_fgf, k_total_fgf),
                    (0.0, t_max), u0, t_eval=t_eval,
                    method="RK45", rtol=1e-6, atol=1e-10)

def full_profile(sol):
    """Assemble the full grid profile [mol/cm^3], incl. both boundary nodes."""
    P0_t = np.array([P_gel_surface(t) / Kp for t in sol.t])  # Eq. 36 [mol/cm^3]
    P = np.empty((N, sol.y.shape[1]))
    P[0] = P0_t
    P[1:-1] = sol.y * P0_t[None, :]
    P[-1] = 0.0  # Eq. 38
    return P

P_v14 = full_profile(sol_v14)
P_fgf = full_profile(sol_fgf)

# ---------------------------------------------------------------------------
# Sanity check vs analytic steady state (dP/dt = 0 in Eqs. 35-38):
#   P_ss(x) = P0 * sinh((L - x)/delta) / sinh(L/delta),  delta = sqrt(D_eff/k_total)
# At t = 12 h the profiles must have converged to this within ~1e-3.
# ---------------------------------------------------------------------------
for name, D_eff, k_total, P in [("Variant 14", D_eff_v14, k_total_v14, P_v14),
                                ("FGF2-G3",   D_eff_fgf, k_total_fgf, P_fgf)]:
    delta = np.sqrt(D_eff / k_total)                          # [cm]
    P0 = P_gel_surface(0.0) / Kp                              # [mol/cm^3]
    P_ss = P0 * np.sinh((L_tissue - x) / delta) / np.sinh(L_tissue / delta)
    rel_err = np.max(np.abs(P[:, -1] - P_ss) / P0)
    print(f"{name:10s}: delta = {delta*1e4:6.0f} um | "
          f"max rel. dev. at t=12h vs steady state: {rel_err:.2e}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x_um = x * 1e4   # [cm] -> [um]
times_h = [1, 4, 12]
colors = {"v14": ["#08519c", "#4292c6", "#9ecae1"],
          "fgf": ["#a63603", "#e6550d", "#fdae6b"]}

plt.figure(figsize=(10, 6), dpi=120)
for hr in times_h:
    idx = np.argmin(np.abs(t_eval - hr * 3600))
    i = times_h.index(hr)
    plt.plot(x_um, P_v14[:, idx], color=colors["v14"][i],
             label=f"Variant 14 ({hr} h)")
    plt.plot(x_um, P_fgf[:, idx], color=colors["fgf"][i], ls="--",
             label=f"FGF2-G3 ({hr} h)")

plt.title("Model 3B: Skin Tissue Diffusion & Proteolytic Decay (Eqs. 35-38)")
plt.xlabel("Depth from gel/tissue interface (μm)")
plt.ylabel("[P]_tissue (mol/cm³)")
plt.grid(True, ls=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "model_3B_profiles.png")
plt.savefig(out_png, dpi=300)
plt.show()
