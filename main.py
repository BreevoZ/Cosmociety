from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.visualize import plot_equilibrium


def main():
    result = relax_to_equilibrium()
    plot_equilibrium(result)

    print("Minimal radiative model complete.")
    print(f"Converged step: {result['converged_step']}")
    print("Saved plot to outputs/radiative_equilibrium.png")


if __name__ == "__main__":
    main()
