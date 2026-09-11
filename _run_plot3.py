import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # save to file instead of opening a window

# ---------- PARAMETERS (current placeholders) ----------
D_water = 3.0e-5                 # from literature

eps_c, tau_c = 0.94, 1.03         # TODO: cellulose
eps_h, tau_h = 0.618, 1.27         # TODO: hydrogel

t_half = 2.0                    # TODO: seconds
k_decay = np.log(2) / t_half     # 1/s

L_c = 50e-4                       # cm (50 um)
L_h = 150e-4                      # cm (150 um)
L_total = L_c + L_h

C_w = 3.6e-8                     # mol/L (M), wound NO = 36 nM

# ---------- EFFECTIVE DIFFUSIVITIES & DERIVED QUANTITIES ----------
D1 = D_water * eps_c / tau_c
D2 = D_water * eps_h / tau_h
lam1 = np.sqrt(k_decay / D1)
lam2 = np.sqrt(k_decay / D2)
phi1 = D1 * lam1
phi2 = D2 * lam2

# ---------- CLOSED-FORM SOLUTION ----------
C_int = phi1 * C_w / (phi1*np.cosh(lam1*L_c) + phi2 *
                      np.sinh(lam1*L_c)*np.tanh(lam2*L_h))

x1 = np.linspace(0.0, L_c,     300)
x2 = np.linspace(L_c,  L_total, 300)

C1 = (C_w*np.sinh(lam1*(L_c - x1)) + C_int *
      np.sinh(lam1*x1)) / np.sinh(lam1*L_c)
C2 = C_int*np.cosh(lam2*(L_total - x2)) / np.cosh(lam2*L_h)

# ---------- VERIFY FLUX CONTINUITY ----------
slope1 = lam1*(C_int*np.cosh(lam1*L_c) - C_w) / np.sinh(lam1*L_c)
slope2 = -lam2*C_int*np.tanh(lam2*L_h)
print("D2/D1        = %.4f" % (D2/D1))
print("slope1/slope2= %.4f" % (slope1/slope2))
print("C_interface  = %.2f nM" % (C_int*1e9))
print("C at top     = %.2f nM" %
      (C_int*np.cosh(lam2*(L_total-L_total))/np.cosh(lam2*L_h)*1e9))

# ---------- PLOT (concentration in nM) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(x1*1e4, C1*1e9, color="#0B5394", lw=2, label="Cellulose (layer 1)")
ax.plot(x2*1e4, C2*1e9, color="#C0392B", lw=2, label="Hydrogel (layer 2)")
ax.axvline(L_c*1e4, ls="--", color="0.5", lw=1.2)
ax.text(L_c*1e4 + 3, 0.92*C_w*1e9,
        r"$x = L_{cell} = %.0f$ $\mu$m" % (L_c*1e4), va="top", fontsize=9)

ax.set_xlabel(r"Distance through patch, $x$  ($\mu$m)")
ax.set_ylabel("NO concentration (nM)")
ax.set_title("Steady-state NO gradient across the two-layer patch")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "plot3_no_gradient.png")
plt.savefig(out, dpi=150)
print("saved ->", out)
