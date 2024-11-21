import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotnine as p9
import seaborn as sns


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
            {"Control": [control.sum(), np.sum(self.weights[control])**2 / np.sum(self.weights[control]**2)],
             "Treated": [treated.sum(), np.sum(self.weights[treated])**2 / np.sum(self.weights[treated]**2)],
             },
            index=['Unadjusted', 'Adjusted']).round(2)

        sum_weights = self.weights.sum()
        weights_factor = sum_weights / ((sum_weights)**2 - (self.weights**2).sum())

        for column in self.cov_cols:

            col_is_binary = set(self.df[column].unique()).issubset({0, 1})
            if col_is_binary:
                sd = 1
            else:
                sd2 = weights_factor * np.sum(self.weights * (self.df[column] - np.mean(self.df[column]))**2)
                sd = np.sqrt(sd2)

            treated_mean_balanced = np.average(df_treated[column], weights=self.weights[df_treated.index])
            control_mean_balanced = np.average(df_control[column], weights=self.weights[df_control.index])

            treated_mean_unbalanced = np.mean(df_treated[column])
            control_mean_unbalanced = np.mean(df_control[column])

            smd_balanced = (treated_mean_balanced - control_mean_balanced) / sd
            smd_unbalanced = (treated_mean_unbalanced - control_mean_unbalanced) / sd

            new_row = pd.DataFrame({
                'covariate': [column],
                'type': ['Binary' if col_is_binary else 'Contin.'],
                'diff_unadj': [smd_unbalanced],
                'diff_adj': [smd_balanced],
            })
            df_smd = pd.concat([df_smd, new_row], ignore_index=True)

        return df_effective_sample_size, df_smd

    def love_plot(self, thresholds=[0.1]):
        """
        Generate a love plot for the corresponding standardized mean difference (SMD) table.
        """

        df_renamed = self.df_smd.rename(columns={
            'diff_unadj': 'Unadjusted',
            'diff_adj': 'Adjusted'
        })

        cov_order = df_renamed.sort_values(
            by='Unadjusted', ascending=True
        )["covariate"].tolist()

        df_plot = df_renamed.melt(
            id_vars=['covariate', 'type'],
            value_vars=['Unadjusted', 'Adjusted'],
            var_name='Sample',
            value_name='SMD')

        p = (p9.ggplot(df_plot, p9.aes(x='covariate', y='SMD', fill='Sample', group='Sample')) +
             p9.geom_point(size=2) +
             p9.geom_line(p9.aes(color="Sample"), size=.5, show_legend=False) +
             p9.theme_minimal() +
             p9.theme_bw() +
             p9.scale_x_discrete(limits=cov_order) +
             p9.coord_flip() +
             p9.labs(title='Covariate Balance', x='Covariate', y='Standardized Mean Difference (SMD)'))

        for threshold in thresholds:
            p += p9.geom_hline(yintercept=abs(threshold), linetype='dashed', color='red')
            p += p9.geom_hline(yintercept=-abs(threshold), linetype='dashed', color='red')

        return p

    def balance_plot(self, covariate, n_bins=20):
        """
        Generate a balance plot for the specified covariate.

        Parameters:
            covariate (str): The covariate to plot.
            n_bins (int): The number of bins to use for the histogram.
        """

        df_plot = pd.DataFrame({
            "covariate": self.df[covariate],
            "unadjusted": 1.0,
            "adjusted": self.weights,
            "treatment": np.where(self.df[self.treatment] == 1, "Treated", "Control")
        }).melt(
            value_vars=["unadjusted", "adjusted"],
            id_vars=["treatment", "covariate"],
            var_name="Sample",
            value_name="weight")

        # Create subplots (2 subplots for 'unadjusted' and 'adjusted')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Define color mapping for treatment groups
        palette = sns.color_palette("colorblind")
        colors = {'Treated': palette[0], 'Control': palette[1]}

        # Plot each subplot
        for i, sample in enumerate(["unadjusted", "adjusted"]):
            ax = axes[i]
            # Filter data for the current sample
            sample_data = df_plot[df_plot['Sample'] == sample]

            # Plot the histogram for each treatment group with separate histograms
            for treatment_group in ['Treated', 'Control']:
                treatment_data = sample_data[sample_data['treatment'] == treatment_group]
                sns.histplot(
                    treatment_data,
                    x="covariate",
                    weights="weight",
                    kde=True,
                    alpha=0.6,
                    stat="density",
                    bins=n_bins,
                    color=colors[treatment_group],
                    label=treatment_group,
                    ax=ax
                )
            ax.set_title(f"{sample.capitalize()} Distribution")
            ax.set_xlabel(covariate)
            ax.set_ylabel('Density')
            ax.legend(title='Treatment')

        # Adjust the layout and title
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'Distributional Balance for {covariate}', fontsize=16)

        return fig