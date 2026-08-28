"""Typovana sprava konfigurace nad ``config.yaml``.

Modul definuje vnorene dataclassy odpovidajici sekcim ``config.yaml`` a dve
funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy, zvaliduje) a
``validate_config`` (rozsahove kontroly s ceskymi chybovymi hlaskami).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Nastaveni nacitani dat (sekce ``data`` v ``config.yaml``)."""

    standardize: bool
    random_state: int


@dataclass
class FeatureSelectionConfig:
    """Nastaveni vyberu priznaku (sekce ``feature_selection``)."""

    alpha: float
    min_effect_size: float
    silhouette_target: float
    knn_accuracy_target: float
    knn_neighbors: int
    test_size: float


@dataclass
class PCAConfig:
    """Nastaveni PCA (sekce ``pca``)."""

    variance_threshold: float
    n_components: int | None


@dataclass
class ClusteringConfig:
    """Nastaveni navazujiciho shlukovani (sekce ``clustering``)."""

    n_clusters: int


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci."""

    data: DataConfig
    feature_selection: FeatureSelectionConfig
    pca: PCAConfig
    clustering: ClusteringConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami. K jednotlivym hodnotam
    se pristupuje pres atributy (napr. ``cfg.pca.variance_threshold``),
    nikdy ne pres klice slovniku.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud nektera hodnota nesplnuje rozsahove kontroly ve
        ``validate_config``.
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    cfg = ExperimentConfig(
        data=DataConfig(
            standardize=bool(raw["data"]["standardize"]),
            random_state=int(raw["data"]["random_state"])
        ),
        feature_selection=FeatureSelectionConfig(
            alpha=float(raw["feature_selection"]["alpha"]),
            min_effect_size=float(raw["feature_selection"]["min_effect_size"]),
            silhouette_target=float(raw["feature_selection"]["silhouette_target"]),
            knn_accuracy_target=float(raw["feature_selection"]["knn_accuracy_target"]),
            knn_neighbors=int(raw["feature_selection"]["knn_neighbors"]),
            test_size=float(raw["feature_selection"]["test_size"]),
        ),
        pca=PCAConfig(
            variance_threshold=float(raw["pca"]["variance_threshold"]),
            n_components=(
                None
                if raw["pca"]["n_components"] is None
                else int(raw["pca"]["n_components"])
            ),
        ),
        clustering=ClusteringConfig(
            n_clusters=int(raw["clustering"]["n_clusters"]),
        ),
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou. Kontroluji se:

    - ``0 < alpha < 1``
    - ``min_effect_size >= 0``
    - ``0 < silhouette_target <= 1``
    - ``0 < knn_accuracy_target <= 1``
    - ``knn_neighbors >= 1``
    - ``0 < test_size < 1``
    - ``0 < variance_threshold <= 100``
    - ``n_components`` je ``None`` nebo ``>= 1``
    - ``n_clusters >= 2``

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    fs = cfg.feature_selection
    pca = cfg.pca
    clustering = cfg.clustering

    if not 0.0 < fs.alpha < 1.0:
        raise ValueError(
            f"alpha musi byt v intervalu (0, 1), zadano: {fs.alpha}"
        )
    if fs.min_effect_size < 0.0:
        raise ValueError(
            f"min_effect_size musi byt >= 0, zadano: {fs.min_effect_size}"
        )
    if not 0.0 < fs.silhouette_target <= 1.0:
        raise ValueError(
            f"silhouette_target musi byt v intervalu (0, 1], zadano: {fs.silhouette_target}"
        )
    if not 0.0 < fs.knn_accuracy_target <= 1.0:
        raise ValueError(
            f"knn_accuracy_target musi byt v intervalu (0, 1], zadano: {fs.knn_accuracy_target}"
        )
    if fs.knn_neighbors < 1:
        raise ValueError(
            f"knn_neighbors musi byt >= 1, zadano: {fs.knn_neighbors}"
        )
    if not 0.0 < fs.test_size < 1.0:
        raise ValueError(
            f"test_size musi byt v intervalu (0, 1), zadano: {fs.test_size}"
        )
    if not 0.0 < pca.variance_threshold <= 100.0:
        raise ValueError(
            f"variance_threshold musi byt v intervalu (0, 100], zadano: {pca.variance_threshold}"
        )
    if pca.n_components is not None and pca.n_components < 1:
        raise ValueError(
            f"n_components musi byt None nebo >= 1, zadano: {pca.n_components}"
        )
    if clustering.n_clusters < 2:
        raise ValueError(
            f"n_clusters musi byt >= 2, zadano: {clustering.n_clusters}"
        )
