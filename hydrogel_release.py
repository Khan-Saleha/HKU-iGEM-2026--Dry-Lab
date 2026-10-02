from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= EDIT PARAMETERS HERE =================
PARAMETER_STATUS = "ILLUSTRATIVE - replace before quantitative interpretation"
L_MM = 1.0                       # Hydrogel thickness, mm
END_HOURS = 24.0
PEPTIDES = {
    "V14": {"D_CM2_S": 1e-7, "HALF_LIFE_H": 24.0, "C0_UM": 10.0},
    "FGF2-G3": {"D_CM2_S": 3e-8, "HALF_LIFE_H": 12.0, "C0_UM": 10.0},
}
PROFILE_TIMES_H = [0.0, 1.0, 4.0, 12.0, 24.0]
OUTPUT_DIR = Path("hydrogel_results")
N_MODES = 4096
# ========================================================


def calculate(D, k, C0, L, t, x, n_modes=N_MODES):
    """Lengths cm, times s, C0 mol/cm^3; outward sink flux is positive.

    Cumulative release integrates the flux analytically, including its integrable
    t=0 singularity. It is NOT 'initial mass minus remaining mass' when k>0.
    """
    mu = (np.arange(n_modes) + 0.5) * np.pi / L
    rate = D * mu**2 + k
    decay = np.exp(-np.outer(t, rate))
    weight = 2.0 / (L * mu)**2
    remaining = decay @ weight
    integrated = -np.expm1(-np.outer(t, rate))
    released = integrated @ (weight * D * mu**2 / rate)
    degraded = integrated @ (weight * k / rate)
    profiles = ((decay * (2.0 * C0 / (L * mu))) @
                np.sin(np.outer(mu, x)))
    flux = (2.0 * C0 * D / L) * decay.sum(axis=1)
    zero = t == 0
    # Exact initial profile: boundary switched to sink at t=0; the corner at
    # x=0 is incompatible with uniform loading, but valid for t>0.
    profiles[zero] = C0
    profiles[:, 0] = 0.0
    remaining[zero] = 1.0
    flux[zero] = np.nan  # Ideal sink causes infinite instantaneous initial flux.
    return profiles, flux, remaining, released, degraded


def main():
    if L_MM <= 0 or END_HOURS <= 0:
        raise ValueError("L_MM and END_HOURS must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    L = L_MM / 10.0
    # Positive times are logarithmically spaced to resolve the initial burst.
    t = np.r_[0.0, np.geomspace(min(1.0, END_HOURS * 3600 / 1000),
                               END_HOURS * 3600, 600)]
    x = np.linspace(0.0, L, 201)
    profile_h = np.array([h for h in PROFILE_TIMES_H if 0 <= h <= END_HOURS])
    fig_profiles, axes = plt.subplots(1, len(PEPTIDES), figsize=(11, 4.5),
                                      squeeze=False, constrained_layout=True)
    fig_release, ax_release = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    fig_flux, ax_flux = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    results = {}
    print(PARAMETER_STATUS)
    for ax, (name, p) in zip(axes.flat, PEPTIDES.items()):
        D, half_h, C0_uM = p["D_CM2_S"], p["HALF_LIFE_H"], p["C0_UM"]
        if D <= 0 or half_h <= 0 or C0_uM <= 0:
            raise ValueError(f"Invalid parameters for {name}")
        k = np.log(2) / (half_h * 3600)
        C0 = C0_uM * 1e-9  # 1 uM = 1e-9 mol/cm^3
        _, flux, remaining, released, degraded = calculate(D, k, C0, L, t, x)
        residual = np.max(np.abs(remaining + released + degraded - 1))
        if residual > 1e-4:
            raise RuntimeError(f"Mass conservation failed for {name}: {residual}")
        if min(remaining.min(), released.min(), degraded.min()) < -1e-10:
            raise RuntimeError("Negative mass fraction")
        profiles, *_ = calculate(D, k, C0, L, profile_h * 3600, x)
        for h, c in zip(profile_h, profiles):
            ax.plot(x * 10, c * 1e9, label=f"{h:g} h")
        ax.set(title=name, xlabel="Distance from sink (mm)", ylabel="Gel concentration (uM)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        ax_release.plot(t / 3600, released * 100, label=name)
        # mol/cm^2/s -> nmol/cm^2/h
        ax_flux.loglog(t[1:] / 3600, flux[1:] * 1e9 * 3600, label=name)
        with (OUTPUT_DIR / f"{name}_release.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_h", "outward_flux_nmol_cm2_h", "remaining_fraction",
                        "intact_released_fraction", "degraded_fraction",
                        "intact_released_nmol_cm2"])
            w.writerows(zip(t / 3600, flux * 1e9 * 3600, remaining,
                            released, degraded, released * C0 * L * 1e9))
        with (OUTPUT_DIR / f"{name}_profiles.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["x_mm"] + [f"C_uM_at_{h:g}h" for h in profile_h])
            w.writerows(zip(x * 10, *(profiles * 1e9)))
        results[name] = {
            "D_cm2_s": D, "half_life_h": None if np.isinf(half_h) else half_h,
            "C0_uM": C0_uM, "final_intact_released_fraction": float(released[-1]),
            "final_degraded_fraction": float(degraded[-1]),
            "final_remaining_fraction": float(remaining[-1]),
            "max_mass_balance_error": float(residual),
        }
        print(f"{name}: released={released[-1]:.2%}, degraded={degraded[-1]:.2%}, "
              f"remaining={remaining[-1]:.2%}; mass error={residual:.2g}")
    ax_release.set(xlabel="Time (h)", ylabel="Intact released / initial load (%)", ylim=(0, 100))
    ax_flux.set(xlabel="Time (h)", ylabel="Outward flux (nmol / cm^2 / h)")
    for ax in [ax_release, ax_flux]:
        ax.legend()
        ax.grid(alpha=.2)
    for fig, name in [(fig_profiles, "concentration_profiles"),
                      (fig_release, "cumulative_intact_release"), (fig_flux, "release_flux")]:
        fig.suptitle("ILLUSTRATIVE PARAMETERS - " + name.replace("_", " "), fontsize=10)
        fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=200)
        plt.close(fig)
    with (OUTPUT_DIR / "parameters_and_checks.json").open("w", encoding="utf-8") as f:
        json.dump({"status": PARAMETER_STATUS, "L_mm": L_MM,
                   "end_hours": END_HOURS, "boundary": "ideal sink + impermeable backing",
                   "source": 0, "n_modes": N_MODES, "peptides": results}, f, indent=2)
    print(f"Saved to: {OUTPUT_DIR.resolve()}")
    print("Initial flux is NaN in CSV: the ideal-sink t=0 singularity is omitted from the plot.")


if __name__ == "__main__":
    main()
