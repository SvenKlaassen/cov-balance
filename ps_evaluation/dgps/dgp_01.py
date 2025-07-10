import numpy as np
import pandas as pd
from typing import Optional, Literal
from scipy.special import expit


def _generate_linear_features(X: np.ndarray) -> np.ndarray:
    """Generate linear features from base covariates."""
    return X


def _generate_nonlinear_features(X: np.ndarray) -> np.ndarray:
    """Generate nonlinear features from base covariates."""
    features = [X]  # Start with original features

    # Polynomial terms
    features.append(X[:, [0]] ** 2)
    features.append(X[:, [1]] ** 3)
    features.append(X[:, [2]] ** 2)

    # Exponential and log terms
    features.append(np.exp(X[:, [0]] / 3))
    features.append(np.log(np.abs(X[:, [1]]) + 1))

    # Interactions
    features.append((X[:, [0]] * X[:, [1]]).reshape(-1, 1))
    features.append((X[:, [0]] * X[:, [2]]).reshape(-1, 1))
    features.append((X[:, [1]] * X[:, [2]]).reshape(-1, 1))

    # Trigonometric terms
    features.append(np.sin(X[:, [3]] * np.pi))
    features.append(np.cos(X[:, [4]] * np.pi / 2))

    # Step functions
    features.append((X > 0).astype(float))

    return np.column_stack(features)


def _generate_treatment_assignment(
    X: np.ndarray,
    assignment_type: Literal["linear", "nonlinear"],
    coef_std: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate treatment assignment and propensity scores.

    Parameters:
    -----------
    X : np.ndarray
        Base covariates
    assignment_type : str
        Type of assignment mechanism
    coef_std : float, optional
        Standard deviation for coefficient generation. If None, uses defaults.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Treatment assignments and true propensity scores
    """
    if assignment_type == "linear":
        X_features = _generate_linear_features(X)
        default_coef_std = 0.5
    elif assignment_type == "nonlinear":
        X_features = _generate_nonlinear_features(X)
        default_coef_std = 0.3
    else:
        raise ValueError(f"Unknown treatment assignment type: {assignment_type}")

    # Use provided coef_std or default
    if coef_std is None:
        coef_std = default_coef_std

    # Generate coefficients and linear combination
    treatment_coef = np.random.normal(0, coef_std, size=X_features.shape[1])
    linear_treatment = X_features @ treatment_coef

    # Generate treatment and propensity scores
    true_ps = expit(linear_treatment)
    treatment = np.random.binomial(1, true_ps)

    return treatment, true_ps


def _generate_outcome(
    X: np.ndarray,
    treatment: np.ndarray,
    assignment_type: Literal["linear", "nonlinear"],
    treatment_effect: float,
    noise_std: float,
    coef_std: float = None,
) -> np.ndarray:
    """
    Generate outcomes based on covariates and treatment.

    Parameters:
    -----------
    X : np.ndarray
        Base covariates
    treatment : np.ndarray
        Treatment assignments
    assignment_type : str
        Type of outcome mechanism
    treatment_effect : float
        True treatment effect
    noise_std : float
        Standard deviation of noise
    coef_std : float, optional
        Standard deviation for coefficient generation. If None, uses defaults.

    Returns:
    --------
    np.ndarray
        Continuous outcomes
    """
    if assignment_type == "linear":
        X_features = _generate_linear_features(X)
        default_coef_std = 0.5
    elif assignment_type == "nonlinear":
        X_features = _generate_nonlinear_features(X)
        default_coef_std = 0.3
    else:
        raise ValueError(f"Unknown outcome assignment type: {assignment_type}")

    # Use provided coef_std or default
    if coef_std is None:
        coef_std = default_coef_std

    # Generate coefficients and linear combination
    outcome_coef = np.random.normal(0, coef_std, size=X_features.shape[1])
    linear_outcome = X_features @ outcome_coef + treatment_effect * treatment

    # Add noise
    noise = np.random.normal(0, noise_std, size=len(treatment))
    outcome = linear_outcome + noise

    return outcome


def dgp_01(
    n: int = 1000,
    n_covariates: int = 5,
    treatment_assignment: Literal["linear", "nonlinear"] = "linear",
    outcome_assignment: Literal["linear", "nonlinear"] = "linear",
    treatment_effect: float = 1.0,
    noise_std: float = 1.0,
    treatment_coef_std: float = None,
    outcome_coef_std: float = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a DGP with binary treatment and continuous outcome.

    Parameters:
    -----------
    n : int
        Number of observations to generate
    n_covariates : int
        Number of covariates to generate (assumes at least 5)
    treatment_assignment : str
        Treatment assignment mechanism: 'linear' or 'nonlinear'
    outcome_assignment : str
        Outcome assignment mechanism: 'linear' or 'nonlinear'
    treatment_effect : float
        True treatment effect on outcome
    noise_std : float
        Standard deviation of outcome noise
    treatment_coef_std : float, optional
        Standard deviation for treatment assignment coefficients.
        If None, uses defaults (0.5 for linear, 0.3 for nonlinear)
    outcome_coef_std : float, optional
        Standard deviation for outcome generation coefficients.
        If None, uses defaults (0.5 for linear, 0.3 for nonlinear)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing covariates, treatment, outcome, and true propensity score
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate base covariates from standard normal distribution
    X = np.random.normal(0, 1, size=(n, n_covariates))

    # Generate treatment assignment
    treatment, true_ps = _generate_treatment_assignment(
        X, treatment_assignment, treatment_coef_std
    )

    # Generate outcome
    outcome = _generate_outcome(
        X, treatment, outcome_assignment, treatment_effect, noise_std, outcome_coef_std
    )

    # Create DataFrame
    covariate_names = [f"X{i+1}" for i in range(n_covariates)]

    df = pd.DataFrame(X, columns=covariate_names)
    df["treatment"] = treatment
    df["outcome"] = outcome
    df["true_propensity_score"] = true_ps

    return df
