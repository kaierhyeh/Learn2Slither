"""
Generate Q-learning convergence curve (Sessions vs Avg Length & Max Length).

Trains QTableAgent progressively from 1 to 10,000 sessions, saves checkpoint
models to models/, evaluates each checkpoint over 200 episodes, and plots a
dual-Y-axis convergence chart (Left Y: Average Length, Right Y: Max Length)
to assets/img/convergence_curve.png using matplotlib.
"""

import os
import random
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

from ai import QTableAgent  # noqa: E402
from core import Environment  # noqa: E402

CHECKPOINTS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
SAVE_SESSIONS = {1, 10, 100, 1000, 10000}
EVAL_EPISODES = 200


def evaluate_agent(
    agent: QTableAgent,
    env: Environment,
    episodes: int = EVAL_EPISODES,
) -> Tuple[float, int, float]:
    """Evaluate frozen agent and return (avg_len, max_len, goal_rate_pct)."""
    rng_state = random.getstate()
    prev_dontlearn = agent.dontlearn
    prev_eps = agent.epsilon
    agent.dontlearn = True
    agent.epsilon = 0.0

    lengths: List[int] = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 2000:
            action = agent.choose_action(state)
            state, _, done, _ = env.step(action)
            steps += 1
        lengths.append(env.max_length)

    agent.dontlearn = prev_dontlearn
    agent.epsilon = prev_eps
    random.setstate(rng_state)

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    goal_rate = 100.0 * sum(1 for x in lengths if x >= 10) / len(lengths)
    return avg_len, max_len, goal_rate


def run_convergence_experiment(
    models_dir: str,
) -> List[Dict[str, float]]:
    """Train progressively across CHECKPOINTS and collect metrics."""
    random.seed(42)
    env = Environment()
    agent = QTableAgent()

    results: List[Dict[str, float]] = []
    current_sess = 0

    for target_sess in CHECKPOINTS:
        while current_sess < target_sess:
            state = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
            current_sess += 1

        if target_sess in SAVE_SESSIONS:
            model_path = os.path.join(models_dir, f"{target_sess}sess.txt")
            agent.save(model_path)

        avg_len, max_len, goal_rate = evaluate_agent(agent, env)
        states_count = len(agent.q_table)
        results.append({
            "sessions": float(target_sess),
            "avg_len": avg_len,
            "max_len": float(max_len),
            "goal_rate": goal_rate,
            "states": float(states_count),
        })
        print(
            f"Sessions={target_sess:5d} | States={states_count:3d} | "
            f"Avg Len={avg_len:5.2f} | Max Len={max_len:2d} | "
            f">=10: {goal_rate:5.1f}%"
        )

    return results


def plot_dual_axis_convergence(
    results: List[Dict[str, float]],
    output_path: str,
) -> None:
    """Plot dual-Y-axis convergence curve with matplotlib."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sessions = [r["sessions"] for r in results]
    avg_lens = [r["avg_len"] for r in results]
    max_lens = [r["max_len"] for r in results]

    bg_color = "#0f172a"
    panel_color = "#1e293b"
    grid_color = "#334155"
    text_color = "#f8fafc"
    subtext_color = "#94a3b8"
    avg_color = "#38bdf8"
    max_color = "#fb923c"
    goal_color = "#4ade80"

    fig, ax1 = plt.subplots(figsize=(10.5, 6.0), dpi=240)
    fig.patch.set_facecolor(bg_color)
    ax1.set_facecolor(panel_color)

    ax2 = ax1.twinx()

    # Plot curves on Left Y (Avg Length) and Right Y (Max Length)
    line1 = ax1.plot(
        sessions,
        avg_lens,
        color=avg_color,
        linewidth=2.8,
        marker="o",
        markersize=7,
        markerfacecolor=avg_color,
        markeredgecolor=bg_color,
        markeredgewidth=1.5,
        label="Average Length (Left Axis)",
        zorder=4,
    )
    line2 = ax2.plot(
        sessions,
        max_lens,
        color=max_color,
        linewidth=2.4,
        linestyle="--",
        marker="s",
        markersize=6.5,
        markerfacecolor=max_color,
        markeredgecolor=bg_color,
        markeredgewidth=1.5,
        label="Max Length (Right Axis)",
        zorder=3,
    )

    # 42 Subject Goal Line (Length >= 10 on Left Axis)
    goal_line = ax1.axhline(
        y=10.0,
        color=goal_color,
        linestyle=":",
        linewidth=1.8,
        label="42 Subject Goal (Avg Length >= 10)",
        zorder=2,
    )
    ax1.text(
        1.25,
        10.5,
        "42 Subject Target (Length >= 10)",
        color=goal_color,
        fontsize=9.5,
        fontweight="bold",
    )

    # X-axis configuration (Log scale)
    ax1.set_xscale("log")
    ax1.set_xlim(0.8, 14000)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""
    ))
    ax1.set_xlabel(
        "Training Sessions (Log Scale)",
        color=text_color,
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    # Left Y-axis configuration (Average Length)
    ax1.set_ylim(0, 28)
    ax1.set_ylabel(
        "Average Length (Cells)",
        color=avg_color,
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )
    ax1.tick_params(axis="y", colors=avg_color, labelsize=10.5)
    ax1.tick_params(axis="x", colors=text_color, labelsize=10.5)

    # Right Y-axis configuration (Max Length)
    ax2.set_ylim(0, 56)
    ax2.set_ylabel(
        "Max Length (Cells)",
        color=max_color,
        fontsize=12,
        fontweight="bold",
        labelpad=12,
    )
    ax2.tick_params(axis="y", colors=max_color, labelsize=10.5)

    # Grid & Spines
    ax1.grid(True, which="major", color=grid_color, linestyle="-", alpha=0.55)
    ax1.grid(True, which="minor", color=grid_color, linestyle=":", alpha=0.25)
    for spine in ax1.spines.values():
        spine.set_color("#475569")
    for spine in ax2.spines.values():
        spine.set_color("#475569")

    # Annotate key checkpoints
    highlight = {1, 10, 100, 500, 1000, 2500, 5000, 10000}
    for r in results:
        sess = int(r["sessions"])
        if sess not in highlight:
            continue
        avg_v = r["avg_len"]
        max_v = int(r["max_len"])
        avg_norm = avg_v / 28.0
        max_norm = max_v / 56.0
        if avg_norm >= max_norm:
            avg_offset = (0, 9)
            max_offset = (0, -15)
        else:
            avg_offset = (0, -15)
            max_offset = (0, 9)

        ax1.annotate(
            f"{avg_v:.2f}",
            xy=(sess, avg_v),
            xytext=avg_offset,
            textcoords="offset points",
            ha="center",
            color="#7dd3fc",
            fontsize=8.5,
            fontweight="bold",
        )
        ax2.annotate(
            f"{max_v}",
            xy=(sess, max_v),
            xytext=max_offset,
            textcoords="offset points",
            ha="center",
            color="#fdba74",
            fontsize=8.5,
            fontweight="bold",
        )

    # Title & Subtitle
    fig.suptitle(
        "Q-Learning Asymptotic Convergence: "
        "Training Sessions vs. Snake Length",
        color=text_color,
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    ax1.set_title(
        "200-Episode Evaluation per Checkpoint "
        "(alpha = 0.1, gamma = 0.5, 3-Level Distance Binning, 338 States)",
        color=subtext_color,
        fontsize=10,
        pad=12,
    )

    # Combined legend
    lines_all = line1 + line2 + [goal_line]
    labels_all = [str(ln.get_label()) for ln in lines_all]
    legend = ax1.legend(
        lines_all,
        labels_all,
        loc="upper left",
        facecolor=bg_color,
        edgecolor="#475569",
        fontsize=9.5,
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(output_path, facecolor=bg_color, dpi=240)
    plt.close(fig)
    print(f"Saved convergence plot to {output_path}")


def main() -> None:
    """Entry point to train checkpoints and render convergence plot."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    png_path = os.path.join(base_dir, "assets", "img", "convergence_curve.png")

    results = run_convergence_experiment(models_dir=models_dir)
    plot_dual_axis_convergence(results=results, output_path=png_path)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
