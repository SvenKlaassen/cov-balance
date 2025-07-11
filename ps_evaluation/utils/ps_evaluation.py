"""
Propensity Score Evaluation Utilities

This module provides comprehensive evaluation functions for propensity score models,
including all standard classification metrics, calibration measures, and correlation
with true propensity scores.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# Configure logger for this module
logger = logging.getLogger(__name__)


def evaluate_propensity_scores(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    true_ps: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Comprehensive evaluation of propensity score models.

    Computes all standard classification metrics, calibration measures,
    and correlation with true propensity scores if available.

    Parameters
    ----------
    y_true : array-like
        True binary treatment labels (0 or 1)
    prob_dict : dict
        Dictionary with model names as keys and predicted probabilities as values
    true_ps : array-like, optional
        True propensity scores for oracle evaluation
    threshold : float, default=0.5
        Classification threshold for binary metrics
    n_bins : int, default=10
        Number of bins for calibration curve calculation

    Returns
    -------
    pd.DataFrame
        Comprehensive evaluation results with all metrics
    """
    results = []

    for model_name, y_prob in prob_dict.items():
        # Ensure probabilities are numpy arrays
        y_prob = np.array(y_prob)
        y_true_arr = np.array(y_true)

        # Binary predictions for classification metrics
        y_pred = (y_prob >= threshold).astype(int)

        # === BASIC CLASSIFICATION METRICS ===
        accuracy = accuracy_score(y_true_arr, y_pred)
        precision = precision_score(y_true_arr, y_pred, zero_division=0)
        recall = recall_score(y_true_arr, y_pred, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)

        # === PROBABILISTIC METRICS ===
        try:
            roc_auc = roc_auc_score(y_true_arr, y_prob)
        except ValueError:
            roc_auc = np.nan

        brier_score = brier_score_loss(y_true_arr, y_prob)

        try:
            log_loss_val = log_loss(y_true_arr, y_prob)
        except ValueError:
            log_loss_val = np.nan

        # === CALIBRATION METRICS ===
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_arr, y_prob, n_bins=n_bins, strategy="uniform"
            )
            # Expected Calibration Error
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            # Maximum Calibration Error
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        except (ValueError, IndexError):
            ece = np.nan
            mce = np.nan

        # === CORRELATION WITH TRUE PROPENSITY SCORES ===
        if true_ps is not None:
            true_ps_arr = np.array(true_ps)
            try:
                pearson_corr, _ = pearsonr(true_ps_arr, y_prob)
            except ValueError:
                pearson_corr = np.nan

            try:
                spearman_corr, _ = spearmanr(true_ps_arr, y_prob)
            except ValueError:
                spearman_corr = np.nan

            # Error metrics vs true propensity scores
            mse_true_ps = mean_squared_error(true_ps_arr, y_prob)
            mae_true_ps = mean_absolute_error(true_ps_arr, y_prob)
            rmse_true_ps = np.sqrt(mse_true_ps)
        else:
            pearson_corr = np.nan
            spearman_corr = np.nan
            mse_true_ps = np.nan
            mae_true_ps = np.nan
            rmse_true_ps = np.nan

        # === ADDITIONAL METRICS ===
        # Predicted probability statistics
        prob_mean = np.mean(y_prob)
        prob_std = np.std(y_prob)
        prob_min = np.min(y_prob)
        prob_max = np.max(y_prob)

        # Treatment prediction rate
        treatment_pred_rate = np.mean(y_pred)

        # Collect all metrics
        results.append(
            {
                "Model": model_name,
                # Classification Metrics
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                # Probabilistic Metrics
                "ROC_AUC": roc_auc,
                "Brier_Score": brier_score,
                "Log_Loss": log_loss_val,
                # Calibration Metrics
                "ECE": ece,
                "MCE": mce,
                # Correlation with True PS
                "Pearson_r": pearson_corr,
                "Spearman_r": spearman_corr,
                "MSE_vs_True_PS": mse_true_ps,
                "MAE_vs_True_PS": mae_true_ps,
                "RMSE_vs_True_PS": rmse_true_ps,
                # Probability Statistics
                "Prob_Mean": prob_mean,
                "Prob_Std": prob_std,
                "Prob_Min": prob_min,
                "Prob_Max": prob_max,
                # Prediction Statistics
                "Treatment_Pred_Rate": treatment_pred_rate,
                "Threshold": threshold,
            }
        )

    # Create DataFrame with results
    results_df = pd.DataFrame(results)

    # Round numeric columns for better readability
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(4)

    return results_df


def rank_models(
    results_df: pd.DataFrame, key_metrics: Optional[list] = None
) -> pd.DataFrame:
    """
    Rank models based on their performance across key metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_propensity_scores
    key_metrics : list, optional
        List of metrics to use for ranking. If None, uses default key metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with model rankings
    """
    if key_metrics is None:
        key_metrics = ["Log_Loss", "ROC_AUC", "Pearson_r", "MSE_vs_True_PS", "ECE"]

    # Create a copy for ranking
    ranking_df = results_df.copy()

    # Define metrics where lower is better vs higher is better
    lower_better = [
        "Log_Loss",
        "Brier_Score",
        "MSE_vs_True_PS",
        "MAE_vs_True_PS",
        "RMSE_vs_True_PS",
        "ECE",
        "MCE",
    ]
    higher_better = [
        "ROC_AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "Pearson_r",
        "Spearman_r",
    ]

    # Calculate ranks for each metric
    for metric in key_metrics:
        if metric in lower_better:
            ranking_df[f"{metric}_rank"] = ranking_df[metric].rank(
                method="min", ascending=True
            )
        elif metric in higher_better:
            ranking_df[f"{metric}_rank"] = ranking_df[metric].rank(
                method="min", ascending=False
            )

    # Calculate overall rank as average of key metric ranks
    rank_columns = [
        f"{metric}_rank"
        for metric in key_metrics
        if f"{metric}_rank" in ranking_df.columns
    ]
    if rank_columns:
        ranking_df["Overall_Rank"] = ranking_df[rank_columns].mean(axis=1)
        ranking_df = ranking_df.sort_values("Overall_Rank")

    return ranking_df


def get_summary_stats(
    y_true: np.ndarray, true_ps: Optional[np.ndarray] = None
) -> Dict[str, Union[int, float]]:
    """
    Get summary statistics for the dataset.

    Parameters
    ----------
    y_true : array-like
        True binary treatment labels
    true_ps : array-like, optional
        True propensity scores

    Returns
    -------
    dict
        Dictionary with summary statistics
    """
    y_true_arr = np.array(y_true)

    summary = {
        "Dataset_Size": len(y_true_arr),
        "Treatment_Rate": np.mean(y_true_arr),
        "Control_Rate": 1 - np.mean(y_true_arr),
        "N_Treated": np.sum(y_true_arr),
        "N_Control": len(y_true_arr) - np.sum(y_true_arr),
    }

    if true_ps is not None:
        true_ps_arr = np.array(true_ps)
        summary.update(
            {
                "True_PS_Mean": np.mean(true_ps_arr),
                "True_PS_Std": np.std(true_ps_arr),
                "True_PS_Min": np.min(true_ps_arr),
                "True_PS_Max": np.max(true_ps_arr),
                "True_PS_Median": np.median(true_ps_arr),
            }
        )

    return summary


def evaluate_and_rank_models(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    true_ps: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    key_metrics: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete evaluation and ranking of propensity score models.

    Parameters
    ----------
    y_true : array-like
        True binary treatment labels (0 or 1)
    prob_dict : dict
        Dictionary with model names as keys and predicted probabilities as values
    true_ps : array-like, optional
        True propensity scores for oracle evaluation
    threshold : float, default=0.5
        Classification threshold for binary metrics
    n_bins : int, default=10
        Number of bins for calibration curve calculation
    key_metrics : list, optional
        List of metrics to use for ranking

    Returns
    -------
    tuple
        (results_df, ranking_df, summary_stats)
    """
    # Evaluate all models
    results_df = evaluate_propensity_scores(
        y_true=y_true,
        prob_dict=prob_dict,
        true_ps=true_ps,
        threshold=threshold,
        n_bins=n_bins,
    )

    # Rank models
    ranking_df = rank_models(results_df, key_metrics)

    # Get summary statistics
    summary_stats = get_summary_stats(y_true, true_ps)

    return results_df, ranking_df, summary_stats


def log_evaluation_report(
    results_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    summary_stats: Dict,
    show_all_metrics: bool = False,
    log_level: int = logging.INFO,
) -> None:
    """
    Log a comprehensive evaluation report using the logging module.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_propensity_scores
    ranking_df : pd.DataFrame
        Ranking results from rank_models
    summary_stats : dict
        Summary statistics from get_summary_stats
    show_all_metrics : bool, default=False
        Whether to show all metrics or just key ones
    log_level : int, default=logging.INFO
        Logging level to use for the report
    """
    logger.log(log_level, "=" * 80)
    logger.log(log_level, "PROPENSITY SCORE MODEL EVALUATION REPORT")
    logger.log(log_level, "=" * 80)

    # Dataset summary
    logger.log(log_level, "Dataset Summary:")
    for key, value in summary_stats.items():
        if isinstance(value, float):
            logger.log(log_level, f"  {key}: {value:.4f}")
        else:
            logger.log(log_level, f"  {key}: {value}")

    # Key performance metrics
    logger.log(log_level, "Key Performance Metrics:")
    key_metrics = [
        "Model",
        "Log_Loss",
        "ROC_AUC",
        "Brier_Score",
        "Pearson_r",
        "MSE_vs_True_PS",
        "ECE",
    ]
    if all(col in results_df.columns for col in key_metrics):
        logger.log(log_level, f"\n{results_df[key_metrics].to_string(index=False)}")

    # Classification metrics
    logger.log(log_level, "Classification Metrics (threshold=0.5):")
    class_metrics = ["Model", "Accuracy", "Precision", "Recall", "F1_Score"]
    if all(col in results_df.columns for col in class_metrics):
        logger.log(log_level, f"\n{results_df[class_metrics].to_string(index=False)}")

    # Calibration metrics
    logger.log(log_level, "Calibration Metrics:")
    cal_metrics = ["Model", "ECE", "MCE"]
    if all(col in results_df.columns for col in cal_metrics):
        logger.log(log_level, f"\n{results_df[cal_metrics].to_string(index=False)}")

    # Oracle evaluation (if available)
    if "Pearson_r" in results_df.columns and not results_df["Pearson_r"].isna().all():
        logger.log(log_level, "Oracle Evaluation (vs True Propensity Scores):")
        oracle_metrics = [
            "Model",
            "Pearson_r",
            "Spearman_r",
            "MSE_vs_True_PS",
            "MAE_vs_True_PS",
            "RMSE_vs_True_PS",
        ]
        if all(col in results_df.columns for col in oracle_metrics):
            logger.log(
                log_level, f"\n{results_df[oracle_metrics].to_string(index=False)}"
            )

    # Model rankings
    logger.log(log_level, "Model Rankings:")
    ranking_metrics = ["Model", "Overall_Rank"]
    if "Overall_Rank" in ranking_df.columns:
        ranking_display = ranking_df[ranking_metrics].copy()
        ranking_display["Overall_Rank"] = ranking_display["Overall_Rank"].round(2)
        logger.log(log_level, f"\n{ranking_display.to_string(index=False)}")

    # Best performers
    if "Overall_Rank" in ranking_df.columns:
        best_overall = ranking_df.iloc[0]["Model"]
        logger.log(log_level, f"Best Overall Model: {best_overall}")

    # Show all metrics if requested
    if show_all_metrics:
        logger.log(log_level, "Complete Results:")
        logger.log(log_level, f"\n{results_df.to_string(index=False)}")

    logger.log(log_level, "=" * 80)
    logger.log(log_level, "METRIC DEFINITIONS:")
    logger.log(log_level, "- Log_Loss: Lower is better (perfect = 0)")
    logger.log(log_level, "- Brier_Score: Lower is better (perfect = 0)")
    logger.log(log_level, "- ROC_AUC: Higher is better (perfect = 1)")
    logger.log(log_level, "- Pearson_r: Correlation with true PS (perfect = 1)")
    logger.log(
        log_level, "- MSE_vs_True_PS: Mean squared error vs true PS (perfect = 0)"
    )
    logger.log(log_level, "- ECE/MCE: Expected/Maximum Calibration Error (perfect = 0)")
    logger.log(log_level, "=" * 80)
