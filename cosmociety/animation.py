import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch


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
    threshold = result.get("convective_gradient_threshold")
    show_convection = bool(result.get("enable_convection", False)) and threshold is not None
    dr = r[1] - r[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ymax = max(history.max(), source.max()) * 1.1

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Normalized radius r/R")
    ax.set_ylabel("Temperature")

    ax.set_title("Radiative Relaxation Toward Equilibrium")

    ax.grid(True, alpha=0.3)

    convective_spans = []

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

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if show_convection:
        legend_handles.append(
            Patch(
                facecolor=(1.0, 0.55, 0.12, 0.22),
                edgecolor="none",
                label="Convective region",
            )
        )
    ax.legend(legend_handles, legend_labels + (["Convective region"] if show_convection else []))

    def active_regions(active):
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        return zip(starts, ends)

    def init():
        line.set_data([], [])
        text.set_text("")
        return line, text

    def update(frame):
        nonlocal convective_spans
        T = history[frame]

        line.set_data(r, T)

        for span in convective_spans:
            span.remove()
        convective_spans = []

        if show_convection:
            active = (-np.diff(T) / dr) > threshold
            for start, end in active_regions(active):
                convective_spans.append(
                    ax.axvspan(
                        r[start],
                        r[end + 1],
                        color="darkorange",
                        alpha=0.28,
                        linewidth=0,
                        zorder=1,
                    )
                )

            active_fraction = active.mean()
            text.set_text(
                f"Frame: {frame}/{len(history)-1}\n"
                f"Convective fraction: {active_fraction:.2%}"
            )
        else:
            text.set_text(f"Frame: {frame}/{len(history)-1}")

        return line, text, *convective_spans

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(history),
        interval=1000 / fps,
        blit=False,
    )

    anim.save(save_path, writer=PillowWriter(fps=fps))

    plt.close(fig)

    print(f"Saved animation to: {save_path}")
