from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.visualize import plot_equilibrium
from cosmociety.animation import animate_relaxation


def main():
    result = relax_to_equilibrium()

    plot_equilibrium(result)

    animate_relaxation(result)

    print("Simulation complete.")
    print(f"Converged step: {result['converged_step']}")


if __name__ == "__main__":
    main()
    