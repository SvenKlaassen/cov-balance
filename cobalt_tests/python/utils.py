import numpy as np
import pandas as pd
import plotnine as p9


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
