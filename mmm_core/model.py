"""
PyMC model construction, fitting, and inference for MMM.
"""

import numpy as np
import pymc as pm
import arviz as az
from typing import Tuple, Literal


def build_mmm_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "mmm"
) -> pm.Model:
    """
    Build a Bayesian Marketing Mix Model using PyMC.
    
    Model structure:
        - alpha: Intercept (baseline)
        - betas: Coefficients for each media channel (truncated normal, >= 0)
        - sigma: Error standard deviation
        - y_obs: Observed sales with StudentT likelihood (robust to outliers)
    
    Args:
        X: Feature matrix, standardized (n_samples, n_features)
        y: Target vector, standardized (n_samples,)
        model_name: Name for the PyMC model
        
    Returns:
        PyMC Model object (not yet fitted)
    """
    n_features = X.shape[1]
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        
        betas = pm.TruncatedNormal(
            "betas",
            mu=0.0,
            sigma=5.0,
            lower=0.0,
            shape=n_features
        )
        
        sigma = pm.HalfNormal("sigma", sigma=2.0)
        
        # Linear model
        mu = alpha + pm.math.dot(X, betas)
        
        # Likelihood (StudentT is robust to outliers)
        y_obs = pm.StudentT("y_obs", nu=5, mu=mu, sigma=sigma, observed=y)
    
    return model


def fit_mmm_model(
    model: pm.Model,
    method: Literal['advi', 'nuts'] = 'advi',
    draws: int = 2000,
    tune: int = 1000,
    n_advi: int = 8000,
    random_seed: int = 42
) -> az.InferenceData:
    """
    Fit the MMM model using either ADVI (fast) or NUTS (accurate).
    
    Args:
        model: PyMC model to fit
        method: 'advi' for variational inference or 'nuts' for MCMC
        draws: Number of posterior samples (for NUTS or ADVI.sample)
        tune: Number of tuning steps (for NUTS only)
        n_advi: Number of ADVI iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        ArviZ InferenceData object with posterior samples
    """
    with model:
        if method == 'advi':
            # Variational inference (faster, approximate)
            approx = pm.fit(
                n=n_advi,
                method="advi",
                random_seed=random_seed,
                progressbar=False
            )
            idata = approx.sample(draws)
        elif method == 'nuts':
            # MCMC sampling (slower, more accurate)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                random_seed=random_seed,
                progressbar=False
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'advi' or 'nuts'")
    
    return idata


def predict_posterior(
    model: pm.Model,
    idata: az.InferenceData,
    X: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate posterior predictive samples.
    
    Args:
        model: Fitted PyMC model
        idata: InferenceData from fitting
        X: Feature matrix for prediction (standardized). If None, uses the data from fitting.
        
    Returns:
        Tuple of (predictions_mean, predictions_std) in standardized scale
    """
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=42,
            progressbar=False
        )
    
    # Extract predictions: shape is (chains, draws, n_samples)
    ppc_arr = ppc.posterior_predictive["y_obs"].values
    
    # Reshape to (total_samples, n_observations)
    ppc_mat = ppc_arr.reshape(-1, ppc_arr.shape[-1])
    
    # Calculate mean and std across samples
    pred_mean = ppc_mat.mean(axis=0)
    pred_std = ppc_mat.std(axis=0)
    
    return pred_mean, pred_std


def get_posterior_summary(
    idata: az.InferenceData,
    var_names: list = None
) -> az.InferenceData:
    """
    Get summary statistics of posterior distributions.
    
    Args:
        idata: InferenceData from fitting
        var_names: List of variable names to summarize (None = all)
        
    Returns:
        Summary DataFrame with mean, std, hdi_3%, hdi_97%, etc.
    """
    if var_names is None:
        var_names = ["alpha", "betas", "sigma"]
    
    return az.summary(idata, var_names=var_names)


def extract_beta_coefficients(idata: az.InferenceData) -> np.ndarray:
    """
    Extract mean beta coefficients from posterior.
    
    Args:
        idata: InferenceData from fitting
        
    Returns:
        Array of mean beta values (n_features,)
    """
    summary = az.summary(idata, var_names=["betas"])
    return summary["mean"].values
