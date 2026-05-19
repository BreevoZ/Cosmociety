from cosmociety.equilibrium import relax_to_equilibrium
from cosmociety.diagnostics import format_summary, summarize_result
from cosmociety.visualize import plot_equilibrium, plot_transport_diagnostics
from cosmociety.animation import animate_relaxation


def main():
    result = relax_to_equilibrium()

    if result["converged_step"] is None:
        raise RuntimeError("Simulation did not reach equilibrium; skipping diagnostics.")

    plot_equilibrium(result)
    plot_transport_diagnostics(result)

    animate_relaxation(result)

    print("Simulation complete.")
    print(f"Converged step: {result['converged_step']}")
    print(format_summary(summarize_result(result)))


if __name__ == "__main__":
    main()
    
