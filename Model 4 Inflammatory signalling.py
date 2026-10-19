import matplotlib.pyplot as plt
import numpy as np

# --- Model parameters ---
K_D_P = 1.0        # Dissociation constant of the V14 peptide (uM)
K_D_L_nM = 65.0    # Dissociation constant of LPS for TLR4/MD2 (nM)
L_uM = 1.0         # Free LPS (endotoxin) concentration (uM)

# Endotoxin load: a = [LPS] / K_D,L
a = (L_uM * 1000.0) / K_D_L_nM

conc_min_uM, conc_max_uM = 0.0, 160.0  # V14 concentration range (uM)

# Target: at most 10% of receptors bound by LPS
target_theta = 0.1


def calculate_dosing_curve(psi, a):
    """Competitive binding model: fraction of TLR4/MD2 bound by LPS.

    NOTE: this is receptor OCCUPANCY, not activation. Real TLR4 activation
    requires (TLR4/MD2/LPS)_2 dimerization, which makes the dose-response
    cooperative (Hill ~ 2). We assume Hill = 1 here; justify this
    simplification on the wiki.
    """
    return a / (1 + a + psi)


plt.figure(figsize=(9, 6), dpi=150)

C_V = np.linspace(conc_min_uM, conc_max_uM, 500)
psi = C_V / K_D_P

theta_LPS = calculate_dosing_curve(psi, a)
plt.plot(C_V, theta_LPS, linewidth=2, label=f"Endotoxin load $a = {a:.1f}$")

# Required V14 concentration at which the curve reaches the target.
psi_req = (a / target_theta) - (1 + a)
c_req = psi_req * K_D_P

plt.plot(c_req, target_theta, "ro", markersize=6)
plt.annotate(
    rf"$C_{{\mathrm{{req}}}}={c_req:.1f}\,\mu$M",
    xy=(c_req, target_theta),
    xytext=(c_req - 10, target_theta + 0.05),
    horizontalalignment="right",
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
)

plt.axhline(
    y=target_theta,
    color="r",
    linestyle="--",
    alpha=0.7,
    label=f"Target $\\theta_{{\\mathrm{{LPS}}}} = {target_theta}$",
)
plt.xlabel(r"V14 Tissue Concentration $C_V^t$ ($\mu$M)", fontsize=12)
plt.ylabel(r"Fraction of TLR4/MD2 Bound by LPS ($\theta_{\mathrm{LPS}}$)", fontsize=12)
plt.title(
    rf"V14 Dosing Concentration Curve ($K_{{\mathrm{{D,P}}}} = {K_D_P}\ \mu$M)",
    fontsize=13,
)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="upper right")
plt.tight_layout()

plt.savefig("v14_dosing_curve.png", dpi=300, bbox_inches="tight")
plt.show()
