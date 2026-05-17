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
    