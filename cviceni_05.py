# -*- coding: utf-8 -*-
"""
================================================================================
 Predmet:  BPC-UIM / KPC-UIM — Umela inteligence v medicine
 Cviceni:  05 — Vyber priznaku a PCA (analyza hlavnich komponent)
 Soubor:   cviceni_05.py  (SPOUSTECI SOUBOR — needitujte, PREDVYPLNENO)
================================================================================
 Popis:
   Vstupni bod ulohy. Nacte konfiguraci a data, spusti vyber priznaku
   (filtr + dva wrappery), natrenuje PCA od zakladu, ulozi a znovu nacte
   model, vykresli srovnani puvodniho a PCA prostoru a nakonec spusti
   experiment s kompromisem mezi zachovanou varianci a kvalitou shlukovani.

   Repozitar bezi v kazde fazi. Dokud nejsou ukoly hotove, kazda faze se
   zastavi jen hlaskou "Úkol: ..." — nikdy nezpracovanym tracebackem.

 Autor (student): __________________________   Login: __________
 Python: 3.12
================================================================================
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# --- Import guard: srozumitelna hlaska misto holeho ImportError -----------------
try:
    from dataio.config_manager import load_config
    from dataio.loader import load_breast_cancer_data
    from dataio.plotting import (
        plot_cumulative_variance,
        plot_feature_vs_pca_space,
        plot_pca_tradeoff,
        plot_reconstruction,
    )
    from src.feature_selection import (
        FilterSelector,
        KNNWrapperSelector,
        SilhouetteWrapperSelector,
    )
    from src.pca import PCA
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)

GRAPHS_DIR = "graphs"       # vystupni grafy (.png)
MODELS_DIR = "models"       # natrenovany a ulozeny PCA model (.npz)
MODEL_PATH = f"{MODELS_DIR}/pca_model.npz"


def _banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol."""
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")



def faze_vyber_priznaku(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Cil 0 — filtrovaci a obalovy vyber priznaku."""
    _banner("Cil 0: Vyber priznaku (filtr + wrappery)")
    fs = cfg.feature_selection

    try:
        vybrane = FilterSelector(fs.alpha, fs.min_effect_size).select(x, y)
        print(f"  Filtr (alpha={fs.alpha}, |d|>={fs.min_effect_size}): "
              f"ponechano {len(vybrane)} z {x.shape[1]} priznaku.")
        print("  Nazvy: " + ", ".join(names[i] for i in vybrane))
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    try:
        sw = SilhouetteWrapperSelector(fs.silhouette_target, cfg.clustering.n_clusters,
                                       cfg.data.random_state)
        idx = sw.select(x, y)
        print(f"  Wrapper (silueta, cil {fs.silhouette_target}): "
              f"staci {len(idx)} priznaku.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    try:
        kw = KNNWrapperSelector(fs.knn_accuracy_target, fs.knn_neighbors,
                                fs.test_size, cfg.data.random_state)
        idx = kw.select(x, y)
        print(f"  Wrapper (presnost kNN, cil {fs.knn_accuracy_target}): "
              f"staci {len(idx)} priznaku.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_pca(cfg, x: np.ndarray) -> tuple[PCA | None, np.ndarray | None]:
    """Cil 1 — natrenovani PCA, projekce a rekonstrukce.

    Vraci ``(pca, x_pca)`` nebo ``(None, None)``, pokud faze neni hotova.
    """
    _banner("Cil 1: PCA od zakladu (kovariance -> vlastni cisla -> projekce)")
    try:
        pca = PCA(variance_threshold=cfg.pca.variance_threshold,
                  n_components=cfg.pca.n_components)
        pca.fit(x)
        print(f"  Ponechano {pca.n_components_} komponent pro "
              f"{cfg.pca.variance_threshold} % kumulativni variance.")
        plot_cumulative_variance(pca.eigenvalues_, cfg.pca.variance_threshold,
                                 save_path=f"{GRAPHS_DIR}/kumulativni_variance.png")
        x_pca = pca.transform(x)
        x_rec = pca.inverse_transform(x_pca)
        rms = float(np.sqrt(np.mean((x - x_rec) ** 2)))
        print(f"  Tvar po projekci: {x_pca.shape};  RMS chyba rekonstrukce: {rms:.4f}")
        plot_reconstruction(x, x_rec, save_path=f"{GRAPHS_DIR}/rekonstrukce.png")
        return pca, x_pca
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
        return None, None



def faze_persistence(pca: PCA | None, x: np.ndarray, x_pca: np.ndarray | None) -> None:
    """Model = ulozene naucene parametry: save -> load -> overeni."""
    _banner("Model jako ulozene parametry: save / load (.npz)")
    if pca is None or x_pca is None:
        print("  Preskoceno — PCA neni natrenovana.")
        return
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        pca.save(MODEL_PATH)
        obnovena = PCA.load(MODEL_PATH)
        shoda = bool(np.allclose(obnovena.transform(x), x_pca))
        print(f"  Model ulozen do {MODEL_PATH}, znovu nacten.")
        print(f"  restored.transform(x) == puvodni x_pca:  {shoda}")
        print("  -> Rozdeleni 'nauc ted / pouzij pozdeji' na induktivnim modelu.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)



def faze_vizualizace(x: np.ndarray, x_pca: np.ndarray | None, y: np.ndarray,
                     names: list[str]) -> None:
    """Srovnani puvodniho prostoru priznaku a prostoru hlavnich komponent."""
    _banner("Vizualizace: puvodni priznaky vs. prostor hlavnich komponent")
    if x_pca is None:
        print("  Preskoceno — PCA neni natrenovana.")
        return
    plot_feature_vs_pca_space(x, x_pca, y, feature_names=names,
                              save_path=f"{GRAPHS_DIR}/priznaky_vs_pca.png")
    print("  Graf ulozen. POZOR: PCA je bez ucitele — lepsi oddeleni trid")
    print("  v PC prostoru je casty vedlejsi efekt, ne cil ani zaruka (od toho je LDA).")



def faze_experiment(cfg, x: np.ndarray, y: np.ndarray) -> None:
    """Cil 2 — kompromis mezi zachovanou varianci, kvalitou shluku a casem."""
    _banner("Cil 2: Experiment — variance vs. kvalita shlukovani vs. cas")
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    prahy = [50.0, 75.0, 95.0, 100.0]
    sil, casy, shody = [], [], []
    try:
        for prah in prahy:
            t0 = time.perf_counter()
            x_red = PCA(variance_threshold=prah).fit_transform(x)
            km = KMeans(n_clusters=cfg.clustering.n_clusters,
                        random_state=cfg.data.random_state, n_init=10)
            labels = km.fit_predict(x_red)
            dt = time.perf_counter() - t0
            s = float(silhouette_score(x_red, labels))
            ari = float(adjusted_rand_score(y, labels))
            sil.append(s)
            casy.append(dt)
            shody.append(ari)
            print(f"  prah {prah:5.1f} %  ->  dim {x_red.shape[1]:2d}, "
                  f"silueta {s:.3f}, shoda s y (ARI) {ari:.3f}, cas {dt*1e3:.1f} ms")
        plot_pca_tradeoff(np.array(prahy), np.array(sil), np.array(casy),
                          save_path=f"{GRAPHS_DIR}/pca_kompromis.png")
        print("  Graf ulozen. Zaver: PCA obvykle udrzi kvalitu shluku pri nizsi dimenzi.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def main() -> None:
    """Spusti celou pipeline cviceni 05 s ochrannymi bloky u kazde faze."""
    _banner("CVICENI 05 — Vyber priznaku & PCA — start")

    # --- Config guard -------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    x, y, names = load_breast_cancer_data(cfg.data.standardize)
    print(f"  Data: x {x.shape}, malignich vzorku {int(y.sum())} / {len(y)}, "
          f"standardizace={cfg.data.standardize}")

    faze_vyber_priznaku(cfg, x, y, names)
    pca, x_pca = faze_pca(cfg, x)
    faze_persistence(pca, x, x_pca)
    faze_vizualizace(x, x_pca, y, names)
    faze_experiment(cfg, x, y)

    _banner("CVICENI 05 — konec")


if __name__ == "__main__":
    main()
