from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PatchParameters:
    wound_no_nm: float = 36.0
    d_water_cm2_s: float = 3.0e-5
    cellulose_porosity: float = 0.94
    cellulose_tortuosity: float = 1.03
    hydrogel_porosity: float = 0.618
    hydrogel_tortuosity: float = 1.27
    cellulose_thickness_um: float = 50.0
    current_hydrogel_thickness_um: float = 150.0

    @property
    def d_cellulose_cm2_s(self) -> float:
        return self.d_water_cm2_s * self.cellulose_porosity / self.cellulose_tortuosity

    @property
    def d_hydrogel_cm2_s(self) -> float:
        return self.d_water_cm2_s * self.hydrogel_porosity / self.hydrogel_tortuosity


def penetration_depth_um(d_eff_cm2_s: float, half_life_s: float) -> float:
    """Return sqrt(D_eff/k) in micrometres, where k = ln(2)/half-life."""
    if d_eff_cm2_s <= 0 or half_life_s <= 0:
        raise ValueError("Diffusivity and half-life must both be positive.")
    k_decay_s = np.log(2.0) / half_life_s
    return float(np.sqrt(d_eff_cm2_s / k_decay_s) * 1.0e4)


def chassis_no_nm(
    hydrogel_thickness_um: np.ndarray | float,
    params: PatchParameters,
    cellulose_half_life_s: float,
    hydrogel_half_life_s: float,
) -> np.ndarray:
    """NO concentration after serial attenuation through both layers."""
    hydrogel_um = np.asarray(hydrogel_thickness_um, dtype=float)
    if np.any(hydrogel_um < 0):
        raise ValueError("Hydrogel thickness cannot be negative.")

    ld_cell_um = penetration_depth_um(
        params.d_cellulose_cm2_s, cellulose_half_life_s
    )
    ld_hydrogel_um = penetration_depth_um(
        params.d_hydrogel_cm2_s, hydrogel_half_life_s
    )
    attenuation = (
        params.cellulose_thickness_um / ld_cell_um
        + hydrogel_um / ld_hydrogel_um
    )
    return params.wound_no_nm * np.exp(-attenuation)


def maximum_total_thickness_um(
    threshold_nm: float,
    params: PatchParameters,
    cellulose_half_life_s: float,
    hydrogel_half_life_s: float,
) -> float | None:
    """Analytic L_total where C_chassis reaches the supplied threshold."""
    if threshold_nm <= 0:
        raise ValueError("Activation threshold must be positive.")

    ld_cell_um = penetration_depth_um(
        params.d_cellulose_cm2_s, cellulose_half_life_s
    )
    ld_hydrogel_um = penetration_depth_um(
        params.d_hydrogel_cm2_s, hydrogel_half_life_s
    )
    max_hydrogel_um = ld_hydrogel_um * (
        np.log(params.wound_no_nm / threshold_nm)
        - params.cellulose_thickness_um / ld_cell_um
    )
    if max_hydrogel_um < 0:
        return None
    return float(params.cellulose_thickness_um + max_hydrogel_um)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcell-um", type=float, default=50.0)
    parser.add_argument("--current-lhydrogel-um", type=float, default=150.0)
    parser.add_argument("--max-lhydrogel-um", type=float, default=350.0)
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--base-half-life-s", type=float, default=2.0)
    parser.add_argument("--min-half-life-s", type=float, default=0.09)
    parser.add_argument("--threshold-nm", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("plot4_output"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.lcell_um <= 0 or args.current_lhydrogel_um < 0:
        raise ValueError("Layer thicknesses must be non-negative and L_cell > 0.")
    if args.max_lhydrogel_um <= 0 or args.points < 2:
        raise ValueError("Sweep maximum must be positive and points must be >= 2.")
    if args.min_half_life_s > args.base_half_life_s:
        raise ValueError("Minimum half-life cannot exceed the base half-life.")

    params = PatchParameters(
        cellulose_thickness_um=args.lcell_um,
        current_hydrogel_thickness_um=args.current_lhydrogel_um,
    )
    hydrogel_um = np.linspace(0.0, args.max_lhydrogel_um, args.points)
    total_um = params.cellulose_thickness_um + hydrogel_um

    base_no = chassis_no_nm(
        hydrogel_um, params, args.base_half_life_s, args.base_half_life_s
    )
    short_half_life_no = chassis_no_nm(
        hydrogel_um, params, args.min_half_life_s, args.min_half_life_s
    )

    current_total_um = (
        params.cellulose_thickness_um + params.current_hydrogel_thickness_um
    )
    current_base_no = float(
        chassis_no_nm(
            params.current_hydrogel_thickness_um,
            params,
            args.base_half_life_s,
            args.base_half_life_s,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "plot4_fixed_lcell_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cellulose_thickness_um",
                "hydrogel_thickness_um",
                "total_thickness_um",
                f"chassis_no_nm_t_half_{args.min_half_life_s:g}s",
                f"chassis_no_nm_t_half_{args.base_half_life_s:g}s",
            ]
        )
        for lh, lt, c_short, c_base in zip(
            hydrogel_um, total_um, short_half_life_no, base_no
        ):
            writer.writerow(
                [
                    f"{params.cellulose_thickness_um:.6g}",
                    f"{lh:.8g}",
                    f"{lt:.8g}",
                    f"{c_short:.10g}",
                    f"{c_base:.10g}",
                ]
            )

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.fill_between(
        total_um,
        short_half_life_no,
        base_no,
        color="#4C78A8",
        alpha=0.18,
        label=(
            "Half-life sensitivity "
            f"({args.min_half_life_s:g}–{args.base_half_life_s:g} s)"
        ),
    )
    ax.plot(
        total_um,
        base_no,
        color="#1F5A94",
        linewidth=2.6,
        label=f"Base case: $t_{{1/2}}={args.base_half_life_s:g}$ s",
    )
    ax.plot(
        total_um,
        short_half_life_no,
        color="#1F5A94",
        linewidth=1.6,
        linestyle="--",
        label=f"Short half-life: $t_{{1/2}}={args.min_half_life_s:g}$ s",
    )
    ax.scatter(
        [current_total_um],
        [current_base_no],
        color="#E45756",
        edgecolor="white",
        linewidth=0.9,
        s=85,
        zorder=4,
        label=(
            "Current design: "
            f"{params.cellulose_thickness_um:g} + "
            f"{params.current_hydrogel_thickness_um:g} μm"
        ),
    )

    threshold_result = "not supplied"
    if args.threshold_nm is not None:
        ax.axhline(
            args.threshold_nm,
            color="#F58518",
            linestyle=":",
            linewidth=2.0,
            label=f"Activation threshold = {args.threshold_nm:g} nM",
        )
        max_total_um = maximum_total_thickness_um(
            args.threshold_nm,
            params,
            args.base_half_life_s,
            args.base_half_life_s,
        )
        if max_total_um is not None and max_total_um <= total_um[-1]:
            ax.axvline(max_total_um, color="#F58518", linestyle=":", linewidth=1.5)
            ax.annotate(
                f"$L_{{max}}$ = {max_total_um:.1f} μm",
                xy=(max_total_um, args.threshold_nm),
                xytext=(12, 16),
                textcoords="offset points",
                color="#9B4D00",
                arrowprops={"arrowstyle": "->", "color": "#9B4D00"},
            )
            threshold_result = f"{max_total_um:.6g} μm"
        elif max_total_um is None:
            threshold_result = "no feasible hydrogel thickness"
        else:
            threshold_result = f"> {total_um[-1]:.6g} μm (outside sweep)"

    ax.set_xlabel("Total patch thickness, $L_{total}$ (μm)")
    ax.set_ylabel("NO concentration at chassis (nM)")
    ax.set_title("Plot 4: Patch Thickness Design Curve (fixed cellulose layer)")
    ax.set_xlim(total_um[0], total_um[-1])
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    figure_path = args.output_dir / "plot4_fixed_lcell.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ld_cell_base = penetration_depth_um(
        params.d_cellulose_cm2_s, args.base_half_life_s
    )
    ld_hydrogel_base = penetration_depth_um(
        params.d_hydrogel_cm2_s, args.base_half_life_s
    )
    print(f"D_eff,cellulose = {params.d_cellulose_cm2_s:.6g} cm^2/s")
    print(f"D_eff,hydrogel   = {params.d_hydrogel_cm2_s:.6g} cm^2/s")
    print(f"L_d,cellulose    = {ld_cell_base:.3f} μm (base case)")
    print(f"L_d,hydrogel     = {ld_hydrogel_base:.3f} μm (base case)")
    print(
        f"Current design   = {current_total_um:g} μm total; "
        f"C_chassis = {current_base_no:.6g} nM (base case)"
    )
    print(f"Activation threshold / L_max = {threshold_result}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
