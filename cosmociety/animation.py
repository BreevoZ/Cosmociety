import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_relaxation(
    result: dict,
    save_path: str = "outputs/radiative_relaxation.gif",
    fps: int = 30,
):
    """
    Animate temperature relaxation toward equilibrium.
    """

    r = result["r"]
    history = result["history"]
    source = result["source"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ymax = max(history.max(), source.max()) * 1.1

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Normalized radius r/R")
    ax.set_ylabel("Temperature")

    ax.set_title("Radiative Relaxation Toward Equilibrium")

    ax.grid(True, alpha=0.3)

    # Static source profile (scaled for visibility)
    source_scaled = source / source.max() * ymax * 0.3
    ax.plot(r, source_scaled, "--", label="Core source (scaled)")

    # Dynamic temperature curve
    line, = ax.plot([], [], linewidth=2, label="Temperature T(r)")

    text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.legend()

    def init():
        line.set_data([], [])
        text.set_text("")
        return line, text

    def update(frame):
        T = history[frame]

        line.set_data(r, T)

        text.set_text(f"Frame: {frame}/{len(history)-1}")

        return line, text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(history),
        interval=1000 / fps,
        blit=True,
    )

    anim.save(save_path, writer=PillowWriter(fps=fps))

    plt.close(fig)

    print(f"Saved animation to: {save_path}")