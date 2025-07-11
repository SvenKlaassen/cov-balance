from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def plot_propensity_distributions(data, title_prefix="", figsize=(12, 5)):
    """
    Plot propensity score distributions for a given dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset containing 'true_propensity_score' and 'treatment' columns
    title_prefix : str
        Prefix for plot titles (e.g., 'Training', 'Test', 'Validation')
    figsize : tuple
        Figure size for the plots
    """
    colors = sns.color_palette()

    plt.figure(figsize=figsize)

    # Subplot 1: Histogram of propensity scores
    plt.subplot(1, 2, 1)
    plt.hist(
        data["true_propensity_score"],
        bins=30,
        alpha=0.7,
        color=colors[0],
        edgecolor="black",
    )
    plt.title(
        f"Distribution of True Propensity Scores\n({title_prefix} Set)", fontsize=14
    )
    plt.xlabel("True Propensity Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Add summary statistics as text
    ps_mean = data["true_propensity_score"].mean()
    ps_std = data["true_propensity_score"].std()
    ps_min = data["true_propensity_score"].min()
    ps_max = data["true_propensity_score"].max()

    plt.text(
        0.02,
        0.95,
        f"Mean: {ps_mean:.3f}\nStd: {ps_std:.3f}\nMin: {ps_min:.3f}\nMax: {ps_max:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Subplot 2: Mirrored propensity scores by treatment group
    plt.subplot(1, 2, 2)
    treated = data[data["treatment"] == 1]["true_propensity_score"]
    control = data[data["treatment"] == 0]["true_propensity_score"]

    # Create histogram data
    bins = np.linspace(0, 1, 21)
    control_counts, _ = np.histogram(control, bins=bins)
    treated_counts, _ = np.histogram(treated, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot mirrored histograms with colorblind-friendly colors
    plt.bar(
        bin_centers,
        control_counts,
        width=bins[1] - bins[0],
        alpha=0.7,
        color=colors[1],
        edgecolor="black",
        label="Control (D=0)",
    )
    plt.bar(
        bin_centers,
        -treated_counts,
        width=bins[1] - bins[0],
        alpha=0.7,
        color=colors[0],
        edgecolor="black",
        label="Treatment (D=1)",
    )

    plt.title(
        f"Mirrored Propensity Scores by Treatment Group\n({title_prefix} Set)",
        fontsize=14,
    )
    plt.xlabel("True Propensity Score")
    plt.ylabel("Control (↑) vs Treatment (↓)")
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    n_total = len(data)
    n_treated = len(treated)
    n_control = len(control)
    treatment_rate = n_treated / n_total

    print(f"{title_prefix} Set Summary:")
    print(f"  Total observations: {n_total}")
    print(f"  Treated: {n_treated} ({treatment_rate:.1%})")
    print(f"  Control: {n_control} ({1-treatment_rate:.1%})")
    print(f"  Propensity Score - Mean: {ps_mean:.3f}, Std: {ps_std:.3f}")
    print(f"  Propensity Score - Range: [{ps_min:.3f}, {ps_max:.3f}]")
    print()


def plot_calibration_curves(y_true, prob_dict, n_bins=10, figsize=(15, 10)):
    """
    Plot calibration curves for multiple models.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    prob_dict : dict
        Dictionary with model names as keys and predicted probabilities as values
    n_bins : int
        Number of bins for calibration curve
    figsize : tuple
        Figure size
    """
    # Get colorblind-friendly colors from current palette
    colors = sns.color_palette()

    n_models = len(prob_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)

    # Create individual calibration plots
    for idx, (model_name, y_prob) in enumerate(prob_dict.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )

        # Calculate Brier score
        brier_score = brier_score_loss(y_true, y_prob)

        # Plot calibration curve
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=8,
            label=f"{model_name} (Brier: {brier_score:.3f})",
        )

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Plot: {model_name.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide extra subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Create combined calibration plot
    plt.figure(figsize=(10, 8))

    for idx, (model_name, y_prob) in enumerate(prob_dict.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        brier_score = brier_score_loss(y_true, y_prob)

        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=8,
            label=f"{model_name} (Brier: {brier_score:.3f})",
        )

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=2, label="Perfect calibration")

    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Fraction of Positives", fontsize=14)
    plt.title("Calibration Curves Comparison (Training Data)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_vs_true_propensity_comparison(true_ps, prob_dict, figsize=(12, 24)):
    """
    Plot predicted vs true propensity scores for base models and their calibrated versions.
    Left column: uncalibrated models, Right column: calibrated models

    Parameters:
    -----------
    true_ps : array-like
        True propensity scores
    prob_dict : dict
        Dictionary with model names as keys and predicted probabilities as values
    figsize : tuple
        Figure size
    """

    # Get colorblind-friendly colors
    colors = sns.color_palette()

    # Find base models and their calibrated versions
    base_models = {k: v for k, v in prob_dict.items() if not k.endswith("_calibrated")}

    # Filter to only include base models that have calibrated versions
    valid_base_models = {}
    for base_name, base_probs in base_models.items():
        calibrated_name = f"{base_name}_calibrated"
        if calibrated_name in prob_dict:
            valid_base_models[base_name] = base_probs

    if not valid_base_models:
        print("No valid base-calibrated model pairs found!")
        return

    # Create figure with subplots - 2 columns per model (uncalibrated, calibrated)
    n_models = len(valid_base_models)
    rows = n_models
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)

    base_color = colors[0]
    calibrated_color = colors[1]

    for idx, (base_name, base_probs) in enumerate(valid_base_models.items()):

        # Get calibrated version
        calibrated_name = f"{base_name}_calibrated"
        calibrated_probs = prob_dict[calibrated_name]

        # Calculate correlations and MSE
        base_corr, _ = pearsonr(true_ps, base_probs)
        base_mse = mean_squared_error(true_ps, base_probs)

        cal_corr, _ = pearsonr(true_ps, calibrated_probs)
        cal_mse = mean_squared_error(true_ps, calibrated_probs)

        # Left column: Uncalibrated model
        ax_left = axes[idx, 0]
        ax_left.scatter(
            true_ps,
            base_probs,
            alpha=0.5,
            color=base_color,
            s=20,
            label=f"{base_name.upper()} (r={base_corr:.3f}, MSE={base_mse:.3f})",
        )
        ax_left.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect prediction")
        ax_left.set_xlabel("True Propensity Score")
        ax_left.set_ylabel("Predicted Propensity Score")
        ax_left.set_title(f"{base_name.upper()}: Uncalibrated vs True PS")
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)
        ax_left.set_xlim(0, 1)
        ax_left.set_ylim(0, 1)

        # Right column: Calibrated model
        ax_right = axes[idx, 1]
        ax_right.scatter(
            true_ps,
            calibrated_probs,
            alpha=0.5,
            color=calibrated_color,
            s=20,
            label=f"{base_name.upper()} Cal. (r={cal_corr:.3f}, MSE={cal_mse:.3f})",
        )
        ax_right.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect prediction")
        ax_right.set_xlabel("True Propensity Score")
        ax_right.set_ylabel("Predicted Propensity Score")
        ax_right.set_title(f"{base_name.upper()}: Calibrated vs True PS")
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)
        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
