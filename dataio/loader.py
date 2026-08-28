"""Nacitani a standardizace dat Breast Cancer Wisconsin.

Modul poskytuje jedinou funkci ``load_breast_cancer_data``, ktera vrati
priznakovou matici, binarni cilovou promennou a nazvy priznaku, pripadne
z-skorovane sloupce pripravene pro PCA.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_data(
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Nacte dataset Breast Cancer Wisconsin a vrati ``(X, y, feature_names)``.

    Parametry
    ---------
    standardize:
        Pokud ``True``, kazdy sloupec ``X`` se z-skoruje (odecte se prumer
        a deli se smerodatnou odchylkou pocitanou s ``ddof=0``, tj.
        populacni odchylkou). Vysledny sloupec ma prumer 0 a smerodatnou
        odchylku 1. Konzistentne se pouziva ``ddof=0`` pro standardizaci
        i pozdejsi kovariancni vypocty.

    Navratova hodnota
    -----------------
    X:
        ``np.ndarray`` tvaru ``(569, 30)`` typu ``float64``.
    y:
        ``np.ndarray`` tvaru ``(569,)`` s hodnotami ``{0, 1}``, kde
        **1 = maligni (zhoubny)** a **0 = benigni (nezhoubny)**.
    feature_names:
        Seznam 30 nazvu priznaku (``list[str]``).

    Poznamka ke kodovani cilove promenne
    ------------------------------------
    ``sklearn.datasets.load_breast_cancer`` koduje ``target`` opacne:
    0 = malignant, 1 = benign. Kurz vsak pracuje s konvenci
    "vystup 1 -> maligni", proto se stitky prohazuji vztahem
    ``y = 1 - target``. Po prohozeni plati ``y.sum() == 212`` (pocet
    malignich vzorku v datasetu).

    Poznamka k PCA a standardizaci
    ------------------------------
    PCA je citliva na meritko: priznak s vetsim rozsahem hodnot (napr.
    "mean area" v tisicich) by bez standardizace dominoval kovariancni
    matici a prvni komponenty by kopirovaly jen tento priznak, nezavisle
    na jeho informacni hodnote. Po z-skorovani maji vsechny priznaky
    rozptyl 1 a kovariancni matice standardizovanych dat se rovna
    **korelacni matici** puvodnich dat. PCA nad standardizovanymi daty je
    tedy totez jako PCA nad korelacni matici -- to je v tomto cviceni
    zvolena, meritkove neutralni varianta.
    """
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float64)
    # Prohozeni stitku: sklearn ma 0 = malignant, 1 = benign; kurz chce 1 = maligni.
    y = 1 - np.asarray(dataset.target, dtype=np.int64)
    feature_names = [str(name) for name in dataset.feature_names]

    if standardize:
        # Z-skorovani po sloupcich; ddof=0 (populacni smerodatna odchylka).
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=0)
        # Ochrana proti deleni nulou u konstantniho priznaku (v tomto datasetu
        # nenastava, ale drzime funkci robustni).
        std_safe = np.where(std == 0.0, 1.0, std)
        x = (x - mean) / std_safe

    return x, y, feature_names
