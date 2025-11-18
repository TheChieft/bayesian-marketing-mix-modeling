"""
Marketing Mix Modeling Core Module
===================================
A reusable library for Bayesian Marketing Mix Modeling.

Modules:
    - data: Data loading and validation
    - transforms: Adstock, Hill, and scaling transformations
    - model: PyMC model construction and fitting
    - metrics: Performance metrics, contributions, ROI/ROAS
    - viz: Visualization functions
"""

__version__ = "1.0.0"
__author__ = "TheChieft"

from . import data
from . import transforms
from . import model
from . import metrics
from . import viz

__all__ = ["data", "transforms", "model", "metrics", "viz"]
