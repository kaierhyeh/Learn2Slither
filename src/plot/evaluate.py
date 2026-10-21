"""Batch evaluation summary and dashboard chart generation."""

import os
import statistics
from typing import List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def get_eval_image_path(
    model_path: Optional[str] = None,
    sessions: int = 1,
    output_dir: str = "assets/img",
) -> str:
    """Generate non-overwriting image path: eval_M<model>S<sessions>[_<n>]."""
    if model_path:
        base = os.path.basename(model_path)
        stem = os.path.splitext(base)[0]
        if stem.endswith("sess"):
            stem = stem[:-4]
        model_tag = f"M{stem}"
    else:
        model_tag = "Mnone"

    sess_tag = f"S{sessions}"
    base_name = f"eval_{model_tag}{sess_tag}"

    os.makedirs(output_dir, exist_ok=True)
    first_path = os.path.join(output_dir, f"{base_name}.png")
    if not os.path.exists(first_path):
        return first_path

    seq = 1
    while True:
        candidate = os.path.join(output_dir, f"{base_name}_{seq}.png")
        if not os.path.exists(candidate):
            return candidate
        seq += 1


def generate_eval(
    lengths: List[int],
    durations: List[int],
    reasons: List[str],
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    output_dir: str = "assets/img",
) -> str:
    """Print terminal summary statistics and save a 2x2 dashboard plot."""
    n = len(lengths)
    if n == 0:
        return ""

    if not output_path:
        output_path = get_eval_image_path(
            model_path=model_path,
            sessions=n,
            output_dir=output_dir,
        )

    avg_len = sum(lengths) / n
    med_len = statistics.median(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    std_len = statistics.stdev(lengths) if n > 1 else 0.0

    goal_pass = sum(1 for l_val in lengths if l_val >= 10)
    goal_rate = (goal_pass / n) * 100

    tier_15 = (sum(1 for l_val in lengths if l_val >= 15) / n) * 100
    tier_20 = (sum(1 for l_val in lengths if l_val >= 20) / n) * 100
    tier_25 = (sum(1 for l_val in lengths if l_val >= 25) / n) * 100
    tier_30 = (sum(1 for l_val in lengths if l_val >= 30) / n) * 100
    tier_35 = (sum(1 for l_val in lengths if l_val >= 35) / n) * 100

    apples_eaten = sum(max(0, l_val - 3) for l_val in lengths)
    total_steps = sum(durations)
    steps_per_apple = (total_steps / apples_eaten) if apples_eaten > 0 else 0.0

    self_cnt = reasons.count("self")
    wall_cnt = reasons.count("wall")
    zero_cnt = reasons.count("zero_length")
    self_pct = (self_cnt / n) * 100
    wall_pct = (wall_cnt / n) * 100
    zero_pct = (zero_cnt / n) * 100

    # 1. Print formatted terminal summary
    print("\n" + "=" * 58)
    print(f"            Batch Evaluation Summary (N={n})")
    if model_path:
        print(f"Model               : {model_path}")
    print("=" * 58)
    print(f"Average Length      : {avg_len:.2f}")
    print(f"Median Length       : {med_len:.1f}")
    print(f"Min / Max Length    : {min_len} / {max_len}")
    print(f"Std Deviation       : {std_len:.2f}")
    print(f"Goal Rate (>= 10)   : {goal_rate:.1f}% ({goal_pass}/{n})")
    print("Bonus Tiers:")
    print(
        f"  >=15: {tier_15:.1f}% | >=20: {tier_20:.1f}% | "
        f">=25: {tier_25:.1f}% | >=30: {tier_30:.1f}% | >=35: {tier_35:.1f}%"
    )
    print(f"Avg Foraging Rate   : {steps_per_apple:.2f} steps / green apple")
    print(
        f"Death Reasons       : Self: {self_pct:.1f}% | "
        f"Wall: {wall_pct:.1f}% | Zero Length: {zero_pct:.1f}%"
    )
    print("-" * 58)

    # 2. Render 2x2 Matplotlib Dashboard
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    bg_color = "#18181B"
    card_color = "#27272A"
    text_color = "#F4F4F5"
    grid_color = "#3F3F46"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), facecolor=bg_color)
    plt.subplots_adjust(top=0.92, hspace=0.32, wspace=0.28)

    model_label = (
        os.path.basename(model_path) if model_path else "Untrained Agent"
    )
    fig.suptitle(
        f"Evaluation Dashboard: {model_label} (N={n} Episodes)",
        color=text_color, fontsize=13, fontweight="bold", y=0.97
    )

    for row in axes:
        for ax in row:
            ax.set_facecolor(card_color)
            ax.tick_params(colors=text_color, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(grid_color)
            ax.grid(True, linestyle="--", alpha=0.35, color=grid_color)

    # --- Subplot 1: Max Length Per Session ---
    ax1 = axes[0, 0]
    sessions_x = list(range(1, n + 1))
    bar_colors = ["#28C850" if val >= 10 else "#EF4444" for val in lengths]
    ax1.bar(sessions_x, lengths, color=bar_colors, alpha=0.85, width=0.7)
    ax1.axhline(
        10, color="#28C850", linestyle="--", linewidth=1.8,
        label="Goal (>=10)"
    )
    ax1.axhline(
        avg_len, color="#F59E0B", linestyle="-.", linewidth=1.8,
        label=f"Mean ({avg_len:.1f})"
    )
    ax1.set_title(
        f"Session Max Length (N={n}, Goal Rate={goal_rate:.1f}%)",
        color=text_color, fontsize=12, fontweight="bold", pad=10
    )
    ax1.set_xlabel("Session Number", color=text_color, fontsize=10)
    ax1.set_ylabel("Max Length", color=text_color, fontsize=10)
    ax1.legend(
        facecolor=card_color, edgecolor=grid_color, labelcolor=text_color
    )

    # --- Subplot 2: Length Distribution & Bonus Tiers (Inverted Axes) ---
    ax2 = axes[0, 1]
    tier_labels = [
        "< 10", "10-14", "15-19", "20-24", "25-29", "30-34", ">= 35"
    ]
    tier_counts = [
        sum(1 for val in lengths if val < 10),
        sum(1 for val in lengths if 10 <= val < 15),
        sum(1 for val in lengths if 15 <= val < 20),
        sum(1 for val in lengths if 20 <= val < 25),
        sum(1 for val in lengths if 25 <= val < 30),
        sum(1 for val in lengths if 30 <= val < 35),
        sum(1 for val in lengths if val >= 35),
    ]
    tier_palette = [
        "#EF4444", "#3B82F6", "#06B6D4", "#10B981", "#22C55E",
        "#84CC16", "#EAB308"
    ]
    x_pos = list(range(len(tier_labels)))
    ax2.bar(x_pos, tier_counts, color=tier_palette, alpha=0.88, width=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tier_labels, color=text_color, fontsize=9)
    ax2.set_title(
        "Length Distribution Across Bonus Tiers",
        color=text_color, fontsize=12, fontweight="bold", pad=10
    )
    ax2.set_xlabel("Length Tiers", color=text_color, fontsize=10)
    ax2.set_ylabel("Session Count", color=text_color, fontsize=10)

    max_cnt = max(tier_counts) if tier_counts else 0
    ax2.set_ylim(0, max(max_cnt * 1.18 + 1, 5))

    for i, count in enumerate(tier_counts):
        if count > 0:
            pct = (count / n) * 100
            ax2.text(
                i, count + max(0.2, max_cnt * 0.02),
                f"{count}\n({pct:.0f}%)",
                ha="center", va="bottom", color=text_color, fontsize=8.5
            )

    # --- Subplot 3: Length vs. Duration (Foraging Efficiency) ---
    ax3 = axes[1, 0]
    ax3.scatter(
        durations, lengths, c=lengths, cmap="viridis",
        s=55, alpha=0.9, edgecolors="none"
    )
    if n > 1 and max(durations) > min(durations):
        try:
            slope, intercept = statistics.linear_regression(durations, lengths)
            xs = [min(durations), max(durations)]
            ys = [slope * x + intercept for x in xs]
            ax3.plot(
                xs, ys, color="#F59E0B", linestyle=":", linewidth=1.8,
                label=f"Trend (~{steps_per_apple:.1f} steps/apple)"
            )
            ax3.legend(
                facecolor=card_color,
                edgecolor=grid_color,
                labelcolor=text_color,
            )
        except Exception:
            pass

    ax3.set_title(
        "Foraging Efficiency (Length vs Steps Survived)",
        color=text_color, fontsize=12, fontweight="bold", pad=10
    )
    ax3.set_xlabel("Duration (Steps)", color=text_color, fontsize=10)
    ax3.set_ylabel("Max Length", color=text_color, fontsize=10)

    # --- Subplot 4: Death Reasons Donut & KPI Summary Card ---
    ax4 = axes[1, 1]
    reason_labels = []
    reason_sizes = []
    reason_colors = []
    if self_cnt > 0:
        reason_labels.append("Self")
        reason_sizes.append(self_cnt)
        reason_colors.append("#3B82F6")
    if wall_cnt > 0:
        reason_labels.append("Wall")
        reason_sizes.append(wall_cnt)
        reason_colors.append("#EF4444")
    if zero_cnt > 0:
        reason_labels.append("Zero Length")
        reason_sizes.append(zero_cnt)
        reason_colors.append("#F59E0B")

    if reason_sizes:
        wedges, texts, autotexts = ax4.pie(
            reason_sizes, labels=reason_labels, colors=reason_colors,
            autopct="%1.0f%%", startangle=140, pctdistance=0.75,
            textprops={"color": text_color, "fontsize": 9}
        )
        for autotext in autotexts:
            autotext.set_color("#FFFFFF")
            autotext.set_fontweight("bold")
        centre_circle = plt.Circle(
            (0, 0), 0.50, fc=card_color, edgecolor=grid_color
        )
        ax4.add_artist(centre_circle)
        ax4.set_title(
            f"Cause of Death Breakdown (N={n})",
            color=text_color, fontsize=12, fontweight="bold", pad=10
        )
    else:
        ax4.text(
            0.5, 0.5, "No Termination Data",
            ha="center", va="center", color=text_color
        )

    plt.savefig(output_path, dpi=180, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation chart to: {output_path}")
    print("=" * 58 + "\n")
    return output_path


generate_eval_report = generate_eval
