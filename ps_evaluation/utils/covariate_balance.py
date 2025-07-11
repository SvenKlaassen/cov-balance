import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import plotnine as p9
import seaborn as sns


# Get current palette colors for plotnine
def get_current_palette_colors():
    """Get current seaborn palette colors as hex for plotnine"""
    from matplotlib.colors import to_hex

    return [to_hex(c) for c in sns.color_palette()]


class BalanceTable:
    def __init__(self, df, cov_cols, treatment, weights):
        """
        Initialize the BalanceTable class and compute the standardized mean difference (SMD).

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            cov_cols (list): A list of columns as covariates.
            treatment (str): The column name of the treatment variable.
            weights (pd.Series or np.ndarray): A vector of weights
        """
        self.df = df
        self.cov_cols = cov_cols
        self.treatment = treatment
        self.weights = weights

        self.df_effective_sample_size, self.df_smd = self.compute_smd()

    @property
    def effective_sample_size(self):
        return self.df_effective_sample_size

    @property
    def smd_table(self):
        return self.df_smd

    def __str__(self):
        summary = f"Effective Sample Size:\n{self.df_effective_sample_size}\n\nSMD Table:\n{self.df_smd}"
        return summary

    def compute_smd(self):
        """
        Compute the standardized mean difference (SMD) for covariates in a dataframe.

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            cov_cols (list): A list of columns as covariates.
            treatment (str): The column name of the treatment variable.
            weights (pd.Series or np.ndarray): A vector of weights

        Returns: Tuple[pd.DataFrame, pd.DataFrame]
            pd.DataFrame: A pandas.DataFrame containing the effective sample size with and without weighting.
            pd.DataFrame: A pandas.DataFrame containing the SMD for each covariate with weighting and without.
        """
        df_smd = pd.DataFrame()

        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0

        df_treated = self.df[treated]
        df_control = self.df[control]

        df_effective_sample_size = pd.DataFrame(
            {
                "Control": [
                    control.sum(),
                    np.sum(self.weights[control]) ** 2
                    / np.sum(self.weights[control] ** 2),
                ],
                "Treated": [
                    treated.sum(),
                    np.sum(self.weights[treated]) ** 2
                    / np.sum(self.weights[treated] ** 2),
                ],
            },
            index=["Unadjusted", "Adjusted"],
        ).round(2)

        sum_weights = self.weights.sum()
        weights_factor = sum_weights / ((sum_weights) ** 2 - (self.weights**2).sum())

        for column in self.cov_cols:

            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                weighted_mean = np.average(self.df[column], weights=self.weights)
                sd2 = weights_factor * np.sum(
                    self.weights * (self.df[column] - weighted_mean) ** 2
                )
                sd = np.sqrt(sd2)

            treated_mean_balanced = np.average(
                df_treated[column], weights=self.weights[df_treated.index]
            )
            control_mean_balanced = np.average(
                df_control[column], weights=self.weights[df_control.index]
            )

            treated_mean_unbalanced = np.mean(df_treated[column])
            control_mean_unbalanced = np.mean(df_control[column])

            smd_balanced = (treated_mean_balanced - control_mean_balanced) / sd
            smd_unbalanced = (treated_mean_unbalanced - control_mean_unbalanced) / sd

            new_row = pd.DataFrame(
                {
                    "covariate": [column],
                    "type": ["Binary" if col_is_binary else "Contin."],
                    "diff_unadj": [smd_unbalanced],
                    "diff_adj": [smd_balanced],
                }
            )
            df_smd = pd.concat([df_smd, new_row], ignore_index=True)

        return df_effective_sample_size, df_smd.round(4)

    def love_plot(self, thresholds=[0.1]):
        """
        Generate a love plot for the corresponding standardized mean difference (SMD) table.
        """

        df_renamed = self.df_smd.rename(
            columns={"diff_unadj": "Unadjusted", "diff_adj": "Adjusted"}
        )

        cov_order = df_renamed.sort_values(by="Unadjusted", ascending=True)[
            "covariate"
        ].tolist()

        df_plot = df_renamed.melt(
            id_vars=["covariate", "type"],
            value_vars=["Unadjusted", "Adjusted"],
            var_name="Sample",
            value_name="SMD",
        )

        p = (
            p9.ggplot(
                df_plot, p9.aes(x="covariate", y="SMD", fill="Sample", group="Sample")
            )
            + p9.geom_point(size=2)
            + p9.geom_line(p9.aes(color="Sample"), size=0.5, show_legend=False)
            + p9.theme_minimal()
            + p9.theme_bw()
            + p9.scale_x_discrete(limits=cov_order)
            + p9.coord_flip()
            + p9.labs(
                title="Covariate Balance",
                x="Covariate",
                y="Standardized Mean Difference (SMD)",
            )
        )

        for threshold in thresholds:
            p += p9.geom_hline(
                yintercept=abs(threshold), linetype="dashed", color="red"
            )
            p += p9.geom_hline(
                yintercept=-abs(threshold), linetype="dashed", color="red"
            )

        return p

    def balance_plot(self, covariate, n_bins=20):
        """
        Generate a balance plot for the specified covariate.

        Parameters:
            covariate (str): The covariate to plot.
            n_bins (int): The number of bins to use for the histogram.
        """

        cov_is_binary = set(self.df[covariate].unique()).issubset({0, 1})

        treated = self.df[self.treatment] == 1

        df_plot = pd.DataFrame(
            {
                "covariate": self.df[covariate],
                "unadjusted": 1.0,
                "adjusted": self.weights,
                "treatment": np.where(treated, "Treated", "Control"),
            }
        ).melt(
            value_vars=["unadjusted", "adjusted"],
            id_vars=["treatment", "covariate"],
            var_name="Sample",
            value_name="weight",
        )

        # Create subplots (2 subplots for 'unadjusted' and 'adjusted')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        palette = sns.color_palette()
        colors = {"Treated": palette[0], "Control": palette[1]}

        for i, sample in enumerate(["unadjusted", "adjusted"]):
            ax = axes[i]
            ax.set_title(f"{sample.capitalize()} Distribution")
            ax.set_xlabel(covariate)
            ax.set_ylabel("Proportion")

            sample_data = df_plot[df_plot["Sample"] == sample]
            if cov_is_binary:
                unique_values = sorted(sample_data["covariate"].unique())
                ax.set_xticks(unique_values)
                ax.set_xticklabels(unique_values)
            else:
                bin_range = (
                    sample_data["covariate"].min(),
                    sample_data["covariate"].max(),
                )

            for treatment_group in ["Treated", "Control"]:
                treatment_data = sample_data[
                    sample_data["treatment"] == treatment_group
                ].copy()

                if cov_is_binary:
                    # adjust x to avoid overlapping bars
                    x_adjustment = 0.2 if treatment_group == "Control" else -0.2
                    treatment_data["covariate"] += x_adjustment
                    sns.histplot(
                        treatment_data,
                        x="covariate",
                        weights="weight",
                        alpha=0.6,
                        stat="proportion",
                        color=colors[treatment_group],
                        label=treatment_group,
                        discrete=True,
                        shrink=0.4,
                        ax=ax,
                    )
                else:
                    sns.histplot(
                        treatment_data,
                        x="covariate",
                        weights="weight",
                        alpha=0.6,
                        stat="proportion",
                        bins=n_bins,
                        color=colors[treatment_group],
                        label=treatment_group,
                        binrange=bin_range,
                        ax=ax,
                    )

            ax.legend(title="Treatment")

        plt.subplots_adjust(top=0.85)
        fig.suptitle(f"Distributional Balance for {covariate}", fontsize=16)

        return fig

    def plot_prop_balance(
        self, propensity_score, covs=None, n_bins=10, confint=False, level=0.95
    ):

        if covs is None:
            covs = self.cov_cols

        treated = self.df[self.treatment] == 1
        control = self.df[self.treatment] == 0

        bin_edges = np.linspace(
            propensity_score.min(), propensity_score.max(), n_bins + 1
        )
        bin_midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
        bin_size = [(bin_edges[i + 1] - bin_edges[i]) for i in range(n_bins)]

        bin_obs_list = [None] * n_bins
        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if i == 0:
                bin_obs_list[i] = (propensity_score >= bin_start) & (
                    propensity_score <= bin_end
                )
            else:
                bin_obs_list[i] = (propensity_score > bin_start) & (
                    propensity_score <= bin_end
                )

        df_plot = pd.DataFrame()
        smd = np.full((n_bins, len(covs)), np.nan)
        lower_bounds = np.full((n_bins, len(covs)), np.nan)
        upper_bounds = np.full((n_bins, len(covs)), np.nan)
        for j, column in enumerate(covs):
            # scaling on the whole sample
            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                s0_sample = self.df[column][control].std()
                s1_sample = self.df[column][treated].std()
                # using pooled variance from cobalt
                sd2 = (s0_sample**2 + s1_sample**2) / 2
                sd = np.sqrt(sd2)

            for i, bin_obs in enumerate(bin_obs_list):

                df_bin_treated = self.df[treated & bin_obs]
                df_bin_control = self.df[control & bin_obs]
                n0 = len(df_bin_control)
                n1 = len(df_bin_treated)

                if n0 == 0 and n1 == 0:
                    bin_smd = 0
                    bin_lower_bound = 0
                    bin_upper_bound = 0
                elif n1 == 0:
                    warnings.warn(
                        f"No treated observations in bin {i}. Setting SMD to 5."
                    )
                    bin_smd = 5
                    bin_lower_bound = 5
                    bin_upper_bound = 5
                elif n0 == 0:
                    warnings.warn(
                        f"No control observations in bin {i}. Setting SMD to -5."
                    )
                    bin_smd = -5
                    bin_lower_bound = -5
                    bin_upper_bound = -5
                else:
                    treated_mean = np.average(df_bin_treated[column])
                    control_mean = np.average(df_bin_control[column])
                    mean_diff = treated_mean - control_mean
                    bin_smd = mean_diff / sd

                    if n1 < 2 or n0 < 2:
                        warnings.warn(
                            f"Bin {i} has less than 2 observations in one group. No CI computed."
                        )
                        bin_lower_bound = np.nan
                        bin_upper_bound = np.nan
                    else:
                        s0 = df_bin_control[column].std()
                        s1 = df_bin_treated[column].std()

                        sd_diff = np.sqrt((s0**2 / n0) + (s1**2 / n1))

                        if n0 == 1 or n1 == 1:
                            df = 1
                        else:
                            df = ((s0**2 / n0) + (s1**2 / n1)) ** 2 / (
                                ((s0**2 / n0) ** 2 / (n0 - 1))
                                + ((s1**2 / n1) ** 2 / (n1 - 1))
                            )

                        t_critical = stats.t.ppf((1 + level) / 2, df)

                        bin_lower_bound = (mean_diff - t_critical * sd_diff) / sd
                        bin_upper_bound = (mean_diff + t_critical * sd_diff) / sd

                smd[i, j] = bin_smd
                lower_bounds[i, j] = bin_lower_bound
                upper_bounds[i, j] = bin_upper_bound

            df_plot_cov = pd.DataFrame(
                {
                    "bin_midpoints": bin_midpoints,
                    "bin_size": bin_size,
                    "smd": smd[:, j],
                    "covariate": column,
                    "lower_bound": lower_bounds[:, j],
                    "upper_bound": upper_bounds[:, j],
                }
            )

            df_plot = pd.concat([df_plot, df_plot_cov], ignore_index=True)

        p = (
            p9.ggplot(df_plot, p9.aes(x="bin_midpoints", y="smd", fill="covariate"))
            + p9.geom_point(size=2)
            + p9.geom_line(mapping=p9.aes(color="covariate"), size=0.5)
            + p9.theme_minimal()
            + p9.theme_bw()
            + p9.coord_flip()
            + p9.labs(
                title="Covariate Balance",
                x="Propensity Score",
                y="Standardized Mean Difference (SMD)",
            )
        )

        if confint:
            p += p9.geom_ribbon(
                p9.aes(ymin="lower_bound", ymax="upper_bound", fill="covariate"),
                alpha=0.2,
            )

        return p


class MultiMethodBalanceTable:
    def __init__(
        self, df, cov_cols, treatment, propensity_scores_dict, matching_methods=None
    ):
        """
        Parameters:
            propensity_scores_dict (dict): Dictionary with method names as keys
                                         and propensity scores as values
            matching_methods (list): List of matching methods to include.
                                   Options: ['ipw', 'nn_matching', 'caliper_matching']
                                   Default: ['ipw', 'nn_matching']
        """
        self.df = df
        self.cov_cols = cov_cols
        self.treatment = treatment
        self.propensity_scores_dict = propensity_scores_dict

        if matching_methods is None:
            matching_methods = ["ipw", "nn_matching"]
        self.matching_methods = matching_methods

        self.balance_tables = {}

        for method_name, ps_scores in propensity_scores_dict.items():
            for weighting_method in matching_methods:
                key = f"{method_name}_{weighting_method}"
                weights = self._ps_to_weights(ps_scores, weighting_method)
                self.balance_tables[key] = BalanceTable(
                    df, cov_cols, treatment, weights
                )

    def _ps_to_weights(self, propensity_scores, method):
        """Convert propensity scores to weights using different methods"""
        treated = self.df[self.treatment] == 1

        if method == "ipw":
            weights = np.where(
                treated,
                1 / propensity_scores,  # 1/e(x) for treated
                1 / (1 - propensity_scores),
            )  # 1/(1-e(x)) for control
            return weights

        elif method == "nn_matching":
            return self._nn_matching_weights(propensity_scores, treated)

        elif method == "caliper_matching":
            return self._caliper_matching_weights(propensity_scores, treated)

        else:
            raise ValueError(f"Unknown weighting method: {method}")

    def _nn_matching_weights(self, propensity_scores, treated, ratio=1):
        """
        Nearest neighbor matching weights (1:1 matching by default)

        Parameters:
            propensity_scores: Array of propensity scores
            treated: Boolean array indicating treatment status
            ratio: Number of control units to match to each treated unit
        """
        weights = np.zeros(len(propensity_scores))

        # Get indices
        treated_idx = np.where(treated)[0]
        control_idx = np.where(~treated)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            return weights

        # Fit nearest neighbors model on control units
        control_ps = propensity_scores[control_idx].reshape(-1, 1)
        treated_ps = propensity_scores[treated_idx].reshape(-1, 1)

        nn = NearestNeighbors(
            n_neighbors=min(ratio, len(control_idx)), metric="euclidean"
        )
        nn.fit(control_ps)

        # Find matches for each treated unit
        distances, indices = nn.kneighbors(treated_ps)

        # Set weights: all treated units get weight 1
        weights[treated_idx] = 1.0

        # Count how many times each control unit is matched
        control_match_counts = np.zeros(len(control_idx))
        for match_indices in indices:
            for idx in match_indices:
                control_match_counts[idx] += 1

        # Set weights for control units
        for i, count in enumerate(control_match_counts):
            if count > 0:
                weights[control_idx[i]] = count

        return weights

    def _caliper_matching_weights(self, propensity_scores, treated, caliper=0.1):
        """
        Caliper matching weights

        Parameters:
            propensity_scores: Array of propensity scores
            treated: Boolean array indicating treatment status
            caliper: Maximum allowed difference in propensity scores
        """
        weights = np.zeros(len(propensity_scores))

        treated_idx = np.where(treated)[0]
        control_idx = np.where(~treated)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            return weights

        # All treated units get weight 1
        weights[treated_idx] = 1.0

        # For each treated unit, find control units within caliper
        control_match_counts = np.zeros(len(control_idx))

        for t_idx in treated_idx:
            t_ps = propensity_scores[t_idx]

            # Find control units within caliper distance
            distances = np.abs(propensity_scores[control_idx] - t_ps)
            within_caliper = distances <= caliper

            if np.any(within_caliper):
                # Find the closest match within caliper
                closest_idx = np.argmin(distances[within_caliper])
                matched_control_idx = np.where(within_caliper)[0][closest_idx]
                control_match_counts[matched_control_idx] += 1

        # Set weights for matched control units
        for i, count in enumerate(control_match_counts):
            if count > 0:
                weights[control_idx[i]] = count

        return weights

    def get_smd_comparison(self):
        """Get SMD comparison across all methods and weighting schemes"""
        comparison_df = pd.DataFrame()

        for method_key, bt in self.balance_tables.items():
            method_smd = bt.smd_table.copy()

            # Parse method name and weighting scheme
            parts = method_key.split("_")
            ps_method = parts[0]
            weighting_method = "_".join(parts[1:])

            method_smd["ps_method"] = ps_method
            method_smd["weighting_method"] = weighting_method
            method_smd["combined_method"] = method_key

            comparison_df = pd.concat([comparison_df, method_smd], ignore_index=True)

        return comparison_df

    def love_plot_comparison(self, facet_by="weighting_method"):
        """
        Create love plot comparing all methods

        Parameters:
            facet_by: Either 'weighting_method' or 'ps_method' to facet the plot
        """
        comparison_df = self.get_smd_comparison()

        # Get current palette colors
        palette_colors = get_current_palette_colors()

        # Reshape for plotting
        plot_df = comparison_df.melt(
            id_vars=[
                "covariate",
                "type",
                "ps_method",
                "weighting_method",
                "combined_method",
            ],
            value_vars=["diff_unadj", "diff_adj"],
            var_name="adjustment",
            value_name="SMD",
        )

        # Create base plot
        if facet_by == "weighting_method":
            p = (
                p9.ggplot(
                    plot_df,
                    p9.aes(
                        x="covariate", y="SMD", color="ps_method", shape="adjustment"
                    ),
                )
                + p9.facet_wrap("~weighting_method")
                + p9.geom_point(size=3, position=p9.position_dodge(width=0.3))
                + p9.coord_flip()
                + p9.geom_hline(yintercept=0.1, linetype="dashed", color="red")
                + p9.geom_hline(yintercept=-0.1, linetype="dashed", color="red")
                + p9.scale_color_manual(values=palette_colors)
                + p9.labs(
                    title="SMD Comparison Across Methods and Weighting Schemes",
                    x="Covariate",
                    y="Standardized Mean Difference",
                )
            )
        else:  # facet_by == 'ps_method'
            p = (
                p9.ggplot(
                    plot_df,
                    p9.aes(
                        x="covariate",
                        y="SMD",
                        color="weighting_method",
                        shape="adjustment",
                    ),
                )
                + p9.facet_wrap("~ps_method")
                + p9.geom_point(size=3, position=p9.position_dodge(width=0.3))
                + p9.coord_flip()
                + p9.geom_hline(yintercept=0.1, linetype="dashed", color="red")
                + p9.geom_hline(yintercept=-0.1, linetype="dashed", color="red")
                + p9.scale_color_manual(values=palette_colors)
                + p9.labs(
                    title="SMD Comparison Across Methods and Weighting Schemes",
                    x="Covariate",
                    y="Standardized Mean Difference",
                )
            )

        return p

    def get_balance_summary(self):
        """Get summary statistics for balance across all methods"""
        comparison_df = self.get_smd_comparison()

        summary_stats = []
        for method in comparison_df["combined_method"].unique():
            method_data = comparison_df[comparison_df["combined_method"] == method]

            # Calculate summary statistics
            adj_smd_abs = np.abs(method_data["diff_adj"])
            unadj_smd_abs = np.abs(method_data["diff_unadj"])

            summary_stats.append(
                {
                    "method": method,
                    "mean_abs_smd_adj": adj_smd_abs.mean(),
                    "max_abs_smd_adj": adj_smd_abs.max(),
                    "prop_balanced_adj": (
                        adj_smd_abs <= 0.1
                    ).mean(),  # Proportion with |SMD| <= 0.1
                    "mean_abs_smd_unadj": unadj_smd_abs.mean(),
                    "max_abs_smd_unadj": unadj_smd_abs.max(),
                }
            )

        return pd.DataFrame(summary_stats)
