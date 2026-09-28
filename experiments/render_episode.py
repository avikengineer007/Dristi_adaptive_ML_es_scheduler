import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from drishti.utils.config import load_scenario_config, create_env_from_config


def render_episode(config_name: str = "hard", seed: int = 1, output_dir: str = "results") -> str:
    """
    Renders and saves a ground-truth spectrum-time waterfall heatmap for an episode.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_data = load_scenario_config(config_name)
    env = create_env_from_config(config_data)

    obs, info = env.reset(seed=seed)
    max_steps = env.max_steps
    num_bands = env.num_bands

    # Step through episode (using default sequential sweep for trajectory)
    for t in range(max_steps):
        action = t % num_bands
        env.step(action)

    gt_matrix = env.ground_truth_matrix[:max_steps, :]
    assert gt_matrix is not None, "Ground truth matrix was not recorded."

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")

    # Plot Ground Truth Waterfall (Time on X, Band on Y)
    # Binary/Power heatmap: Gold/Cyan for active, Dark Navy for silent
    im = ax.imshow(
        gt_matrix.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, max_steps, -0.5, num_bands - 0.5],
        interpolation="nearest",
    )

    ax.set_title(
        f"DRISHTI: Ground Truth Spectrum-Time Heatmap | Scenario: '{config_name.upper()}' (Seed {seed})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Discrete Time Slot (t, ms)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Frequency Channel Index (b)", color="#94a3b8", fontsize=11)
    ax.set_yticks(range(num_bands))
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.25, color="#64748b")

    # Annotate emitter channels on right Y axis
    emitter_summary = {}
    for e in env.emitters:
        emitter_summary[e.emitter_id] = e.__class__.__name__

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    cbar.set_label("Emission State (0 = Silent, 1 = Active)", color="#94a3b8", fontsize=10)
    cbar.ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    output_filename = f"spectrum_heatmap_{config_name}_seed{seed}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"\n[DRISHTI] Spectrum heatmap successfully generated:")
    print(f"  - Scenario: {config_name}")
    print(f"  - Seed: {seed}")
    print(f"  - Bands: {num_bands} | Time Slots: {max_steps}")
    print(f"  - Saved to: {output_path}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render DRISHTI spectrum-time ground truth heatmap")
    parser.add_argument("--config", type=str, default="hard", help="Scenario config name (easy, medium, hard, nonstationary)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output image")
    args = parser.parse_args()

    render_episode(config_name=args.config, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
