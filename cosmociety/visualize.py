import matplotlib.pyplot as plt


def plot_equilibrium(result: dict, save_path: str = "outputs/radiative_equilibrium.png") -> None:
    r = result["r"]
    T = result["temperature"]
    source = result["source"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r, T, label="Equilibrium temperature T(r)")
    ax.plot(r, source / source.max() * T.max() * 0.3, "--", label="Core source, scaled")

    ax.set_xlabel("Normalized radius r/R")
    ax.set_ylabel("Magnitude")
    ax.set_title("Minimal 1D Radiative Equilibrium Model")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_transport_diagnostics(
    result: dict,
    save_path: str = "outputs/transport_diagnostics.png",
) -> None:
    r = result["r"]
    r_interface = result["r_interface"]
    T = result["temperature"]
    density = result["density"]
    kappa = result["kappa"]
    diffusivity = result["diffusivity"]
    radiative_diffusivity = result["radiative_diffusivity"]
    convective_diffusivity = result["convective_diffusivity"]
    flux = result["flux"]
    luminosity = result["luminosity"]
    gradient = result["temperature_gradient"]
    threshold = result["convective_threshold"]
    criterion = result.get("convective_criterion", "gradient")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")

    axes[0, 0].plot(r, T)
    axes[0, 0].set_ylabel("T")
    axes[0, 0].set_title("Temperature")

    axes[0, 1].plot(r, kappa, label="kappa")
    axes[0, 1].plot(r, density, label="rho")
    axes[0, 1].plot(r, radiative_diffusivity, label="D_rad")
    axes[0, 1].plot(r, convective_diffusivity, label="D_conv")
    axes[0, 1].plot(r, diffusivity, label="D_total")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("log scale")
    axes[0, 1].set_title("Opacity and transport")
    axes[0, 1].legend()

    axes[1, 0].plot(r_interface, gradient)
    if criterion == "gradient":
        axes[1, 0].axhline(-threshold, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Normalized radius r/R")
    axes[1, 0].set_ylabel("dT/dr")
    axes[1, 0].set_title(f"Temperature gradient ({criterion} convection)")

    axes[1, 1].plot(r_interface, flux, label="flux")
    axes[1, 1].plot(r_interface, luminosity, label="r^2 flux")
    axes[1, 1].set_xlabel("Normalized radius r/R")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].set_title("Flux and luminosity")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Transport diagnostics")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
