import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


NAMES = [
    "N_W", "N_S", "N_E",
    "S_W", "S_N", "S_E",
    "E_N", "E_S", "E_W",
    "W_N", "W_S", "W_E",
    "N_N", "S_S", "E_E", "W_W"
]

POSITIONS = {
    "N_W": (-2.0, 2.8), "N_S": (0.0, 2.8), "N_E": (2.0, 2.8),
    "S_W": (-2.0, -2.8), "S_N": (0.0, -2.8), "S_E": (2.0, -2.8),
    "E_N": (3.6, 1.6), "E_S": (3.6, 0.0), "E_W": (3.6, -1.6),
    "W_N": (-3.6, 1.6), "W_S": (-3.6, 0.0), "W_E": (-3.6, -1.6)
}


def plot_intersection_state(state, selected_movement=None, title="Traffic intersection state"):
    """
    Returns a matplotlib Figure showing the 16-value state
    (instead of plt.show(), so Gradio can render it directly).
    """
    state = np.asarray(state)
    if state.shape[0] != 16:
        raise ValueError("State must contain 16 values.")

    values = dict(zip(NAMES, state))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.axis("off")

    center = Rectangle((-1.4, -1.2), 2.8, 2.4, fill=False, linewidth=2)
    ax.add_patch(center)
    ax.text(0, 0, "INTERSECTION", ha="center", va="center", fontsize=12)

    ax.text(0, 4.3, f"N_N\n{values['N_N']} cars", ha="center", va="center", fontsize=11)
    ax.text(0, -4.3, f"S_S\n{values['S_S']} cars", ha="center", va="center", fontsize=11)
    ax.text(5, 0, f"E_E\n{values['E_E']} cars", ha="center", va="center", fontsize=11)
    ax.text(-5, 0, f"W_W\n{values['W_W']} cars", ha="center", va="center", fontsize=11)

    for movement, (x, y) in POSITIONS.items():
        marker = "  <-- GREEN" if movement == selected_movement else ""
        ax.text(x, y, f"{movement}: {values[movement]}{marker}", ha="center", va="center", fontsize=9)

    ax.annotate("", xy=(0, 3.8), xytext=(0, 1.3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0, -3.8), xytext=(0, -1.3), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(4.4, 0), xytext=(1.5, 0), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(-4.4, 0), xytext=(-1.5, 0), arrowprops=dict(arrowstyle="->"))

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig
