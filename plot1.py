import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

q_chronic = 50e-9 / 86400 / 1e6        
k_scavenge = 0.01   # 1/s
L_tissue_cm = 0.1                              # cm (1 mm, assumed wound-bed active depth)
N_mac_baseline = 262 * 100 / L_tissue_cm       # 2.62e5 cells/cm3  (baseline M1, CD38+, CG T0)
N_mac_chronic  = 241 * 100 / L_tissue_cm       # 2.41e5 cells/cm3  (untreated chronic M1, CD38+, day 30)
K_p65_nM = 10.0
n_Hill = 2

def q_iNOS(p65_nM, q_max):
    """Single-cell NO catalytic rate (mol/cell/s) vs nuclear p65 (nM)."""
    return q_max * p65_nM**n_Hill / (K_p65_nM**n_Hill + p65_nM**n_Hill)

def no_ss(p65_nM, N_mac, q_max, k_scav):
    """Analytic steady-state [NO]_wound (uM), floored at 0."""
    R = N_mac * q_iNOS(p65_nM, q_max) * 1e12   # uM/s
    return np.maximum(0.0, R / k_scav)

p65_seq = np.linspace(0.1, 60, 400)   # nM

N_levels = [
    (N_mac_baseline, "Baseline M1 (CD38+, 262 cells/mm\u00b2)",                   "#0072B2"),
    (N_mac_chronic,  "Untreated chronic M1, day 30 (CD38+, 241 cells/mm\u00b2)",  "#E69F00"),
]

fig, ax = plt.subplots(figsize=(7.5, 5))
for N_mac, label, colour in N_levels:
    ax.plot(p65_seq, no_ss(p65_seq, N_mac, q_chronic, k_scavenge),
            color=colour, linewidth=1.8, label=label)

ax.set_xlabel(r"Nuclear NF-$\kappa$B p65 concentration, $[\mathrm{p65}]_{\mathrm{nuc}}$ (nM)")
ax.set_ylabel(r"Steady-state wound nitric oxide concentration, $[\mathrm{NO}]_{\mathrm{wound}}$ ($\mu$M)")
fig.suptitle("Plot 1: Inflammation-driven NO generation (Model 1A)", y=0.98)
ax.set_title("Chronic sustained iNOS rate (50 nmol/10\u2076 cells/day), "
             "k_scavenge = 0.01 s\u207b\u00b9, no patch sink", fontsize=9, color="grey", pad=10)
ax.legend(title="M1 macrophage density (CD38+)", frameon=True)
ax.spines[["top", "right"]].set_visible(False)

plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "plot1_no_generation_curve_python.png", dpi=300)
print(f"Saved {OUT_DIR / 'plot1_no_generation_curve_python.png'}")