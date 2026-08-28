"""Verejne API balicku ``dataio`` pro cviceni 05.

Nacitani a standardizace dat (Breast Cancer Wisconsin), sprava konfigurace
(typovane dataclassy nad ``config.yaml``) a vykreslovani. Vsechny tyto moduly
jsou PREDVYPLNENE — kresleni a I/O nejsou predmetem cviceni.
"""

from dataio.config_manager import (
    ClusteringConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSelectionConfig,
    PCAConfig,
    load_config,
    validate_config,
)
from dataio.loader import load_breast_cancer_data
from dataio.plotting import (
    plot_cumulative_variance,
    plot_feature_vs_pca_space,
    plot_pca_tradeoff,
    plot_reconstruction,
)

__all__ = [
    "load_breast_cancer_data",
    "load_config",
    "validate_config",
    "DataConfig",
    "FeatureSelectionConfig",
    "PCAConfig",
    "ClusteringConfig",
    "ExperimentConfig",
    "plot_cumulative_variance",
    "plot_reconstruction",
    "plot_pca_tradeoff",
    "plot_feature_vs_pca_space",
]
