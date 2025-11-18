"""
PyMC model construction, fitting, and inference for MMM.
"""

import logging
import numpy as np
import pymc as pm
import arviz as az
from typing import Tuple, Literal, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    shuffle: bool = False,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    For time series data, typically use shuffle=False to respect temporal order.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        shuffle: Whether to shuffle data before splitting
        random_seed: Random seed for shuffling
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    n_samples = len(y)
    n_train = int(n_samples * (1 - test_size))
    
    if shuffle:
        logger.info(f"Shuffling data with random_seed={random_seed}")
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(n_samples)
    else:
        logger.info("Using temporal order (no shuffle)")
        indices = np.arange(n_samples)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"Split data: {len(y_train)} train samples, {len(y_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def fit_mmm_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    method: Literal['advi', 'nuts'] = 'advi',
    draws: int = 2000,
    tune: int = 1000,
    n_advi: int = 8000,
    random_seed: int = 42
) -> Tuple[pm.Model, az.InferenceData, Dict[str, float]]:
    """
    Fit MMM model with optional train/test validation.
    
    Args:
        X_train: Training features (standardized)
        y_train: Training target (standardized)
        X_test: Test features (standardized), optional
        y_test: Test target (standardized), optional
        method: 'advi' or 'nuts'
        draws: Number of posterior samples
        tune: Number of tuning steps (NUTS only)
        n_advi: Number of ADVI iterations
        random_seed: Random seed
        
    Returns:
        Tuple of (model, idata, metrics_dict)
        metrics_dict contains:
            - train_r2, train_rmse, train_mape
            - test_r2, test_rmse, test_mape (if test data provided)
    """
    from .metrics import compute_fit_metrics
    
    # Build and fit model on training data
    logger.info(f"Building MMM model with {X_train.shape[1]} features")
    model = build_mmm_model(X_train, y_train, model_name="mmm_train")
    
    logger.info(f"Fitting model with method={method}")
    idata = fit_mmm_model(
        model=model,
        method=method,
        draws=draws,
        tune=tune,
        n_advi=n_advi,
        random_seed=random_seed
    )
    
    # Compute training metrics
    y_pred_train, _ = predict_posterior(model, idata, X_train)
    train_metrics = compute_fit_metrics(y_train, y_pred_train)
    
    metrics_dict = {
        "train_r2": train_metrics["R2"],
        "train_rmse": train_metrics["RMSE"],
        "train_mape": train_metrics["MAPE"]
    }
    
    logger.info(f"Training metrics - R²: {train_metrics['R2']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
    
    # Compute test metrics if test data provided
    if X_test is not None and y_test is not None:
        # For test set, we need to build a new model with test data
        # but use the posterior from training
        model_test = build_mmm_model(X_test, y_test, model_name="mmm_test")
        y_pred_test, _ = predict_posterior(model_test, idata, X_test)
        test_metrics = compute_fit_metrics(y_test, y_pred_test)
        
        metrics_dict.update({
            "test_r2": test_metrics["R2"],
            "test_rmse": test_metrics["RMSE"],
            "test_mape": test_metrics["MAPE"]
        })
        
        logger.info(f"Test metrics - R²: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.4f}")
    
    return model, idata, metrics_dict
