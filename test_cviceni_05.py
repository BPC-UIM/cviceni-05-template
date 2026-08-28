# -*- coding: utf-8 -*-
"""
================================================================================
 Predmet:  BPC-UIM / KPC-UIM — Umela inteligence v medicine
 Cviceni:  05 — Vyber priznaku a PCA
 Soubor:   test_cviceni_05.py  (PREDVYPLNENO — testy k self-checku reseni)
================================================================================
 Spousteni:  pytest -v

 Ve stavu stubu se sada NACTE a jednotlive testy, ktere volaji nedokoncene
 ukoly, se oznaci jako xfail (ocekavane selhani s NotImplementedError) — nikdy
 neskonci holym tracebackem. Po dokonceni ukolu se z nich stanou xpass a
 nasledne plne prochazejici testy.

 Znamenko vlastniho vektoru je LIBOVOLNE (analogie s libovolnymi ID slucovani
 v cv2 a permutaci popisku v cv4). Testy proto NIKDY neporovnavaji surove
 vlastni vektory — jen podil vysvetleneho rozptylu, chybu rekonstrukce, nebo
 hodnoty v absolutni hodnote. Zadny DummyDistance zde neni: cviceni 05
 nepocita zadne parove vzdalenosti, neni tedy co oddelovat.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SKPCA

from src.feature_selection import (
    FilterSelector,
    KNNWrapperSelector,
    SilhouetteWrapperSelector,
)
from src.pca import PCA

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="student ukol jeste neni dokoncen")


@pytest.fixture
def data_pca() -> np.ndarray:
    """Nahodna, ale korelovana data (200 x 6) pro testy PCA."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 6)) @ rng.normal(size=(6, 6))


@pytest.fixture
def data_fs() -> tuple[np.ndarray, np.ndarray]:
    """Synteticka data: 3 oddelujici priznaky, 3 sumove, 1 vyznamny-ale-trivialni."""
    n = 240
    rng = np.random.default_rng(1)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    oddelujici = np.column_stack([
        rng.normal(y * 3.0, 1.0),
        rng.normal(y * 2.5, 1.0),
        rng.normal(y * 4.0, 1.0),
    ])
    sum_ = rng.normal(size=(n, 3))
    # Posun jen 0.05 sd: pri n=240 casto vyjde p < 0.05, ale |d| << 0.5.
    trivialni = rng.normal(y * 0.05, 1.0).reshape(-1, 1)
    X = np.hstack([oddelujici, sum_, trivialni])
    return X, y


class TestPCA:
    """Jadro PCA: kovariance, podil rozptylu, rekonstrukce."""

    @STUB
    def test_kovariancni_matice_odpovida_np_cov(self, data_pca: np.ndarray) -> None:
        pca = PCA()
        cov = pca._covariance_matrix(data_pca)
        assert cov.shape == (data_pca.shape[1], data_pca.shape[1])
        assert np.allclose(cov, np.cov(data_pca, rowvar=False))

    @STUB
    def test_explained_variance_ratio_odpovida_sklearn(self, data_pca: np.ndarray) -> None:
        pca = PCA(variance_threshold=100.0).fit(data_pca)
        sk = SKPCA().fit(data_pca)
        # Porovnavame PODIL vysvetleneho rozptylu, ne surove vektory.
        assert np.allclose(pca.explained_variance_ratio_, sk.explained_variance_ratio_)
        assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)

    @STUB
    def test_select_components_monotonni_v_prahu(self, data_pca: np.ndarray) -> None:
        n_low = PCA(variance_threshold=50.0).fit(data_pca).n_components_
        n_high = PCA(variance_threshold=95.0).fit(data_pca).n_components_
        assert 1 <= n_low <= n_high <= data_pca.shape[1]

    @STUB
    def test_rekonstrukce_pri_vsech_komponentach(self, data_pca: np.ndarray) -> None:
        pca = PCA(variance_threshold=100.0).fit(data_pca)
        X_rec = pca.inverse_transform(pca.transform(data_pca))
        # Znamenko komponent je jedno — chyba rekonstrukce na nem nezavisi.
        assert np.allclose(X_rec, data_pca, atol=1e-8)

    @STUB
    def test_transform_ma_spravny_tvar(self, data_pca: np.ndarray) -> None:
        pca = PCA(variance_threshold=90.0).fit(data_pca)
        out = pca.transform(data_pca)
        assert out.shape == (data_pca.shape[0], pca.n_components_)


class TestPCAPersistence:
    """Model = ulozena naucena pole: save -> load round-trip."""

    @STUB
    def test_save_load_zachova_transform(self, data_pca: np.ndarray, tmp_path) -> None:
        pca = PCA(variance_threshold=95.0).fit(data_pca)
        X_pca = pca.transform(data_pca)
        cesta = tmp_path / "model.npz"
        pca.save(str(cesta))
        assert cesta.exists()
        obnovena = PCA.load(str(cesta))
        assert np.allclose(obnovena.transform(data_pca), X_pca)

    @STUB
    def test_save_load_zachova_pole(self, data_pca: np.ndarray, tmp_path) -> None:
        pca = PCA(variance_threshold=95.0).fit(data_pca)
        cesta = tmp_path / "model.npz"
        pca.save(str(cesta))
        obnovena = PCA.load(str(cesta))
        assert np.allclose(obnovena.components_, pca.components_)
        assert np.allclose(obnovena.mean_, pca.mean_)
        assert obnovena.n_components_ == pca.n_components_


class TestFeatureSelection:
    """Filtr (dvojkriterium) a oba wrappery."""

    @STUB
    def test_filtr_vybere_oddelujici_priznaky(self, data_fs) -> None:
        X, y = data_fs
        vybrane = FilterSelector(alpha=0.05, min_effect_size=0.5).select(X, y)
        assert set(vybrane.tolist()) >= {0, 1, 2}

    @STUB
    def test_filtr_zahodi_sumove_priznaky(self, data_fs) -> None:
        X, y = data_fs
        vybrane = set(FilterSelector(alpha=0.05, min_effect_size=0.5).select(X, y).tolist())
        assert {3, 4, 5}.isdisjoint(vybrane)

    @STUB
    def test_filtr_zahodi_vyznamny_ale_trivialni_priznak(self, data_fs) -> None:
        """Duale kriterium: p < alpha NESTACI, kdyz |d| je male."""
        X, y = data_fs
        selektor = FilterSelector(alpha=0.05, min_effect_size=0.5)
        # Priznak c. 6 je posunuty jen o 0.05 sd — mala velikost ucinku.
        a, b = X[y == 0, 6], X[y == 1, 6]
        assert abs(selektor._cohens_d(a, b)) < 0.5
        assert 6 not in selektor.select(X, y).tolist()

    def test_cohens_d_je_predvyplneno(self, data_fs) -> None:
        """Pomocna _cohens_d neni ukol — musi fungovat i ve stavu stubu."""
        X, y = data_fs
        selektor = FilterSelector()
        d_big = selektor._cohens_d(X[y == 0, 2], X[y == 1, 2])
        d_zero = selektor._cohens_d(np.arange(50.0), np.arange(50.0))
        assert abs(d_big) > 1.0
        assert abs(d_zero) < 1e-9

    @STUB
    def test_silhouette_wrapper_dosahne_cile(self, data_fs) -> None:
        X, y = data_fs
        idx = SilhouetteWrapperSelector(target=0.5, n_clusters=2).select(X[:, :3], y)
        assert len(idx) >= 1
        assert list(idx) == list(range(len(idx)))  # souvisly prefix

    @STUB
    def test_knn_wrapper_dosahne_cile(self, data_fs) -> None:
        X, y = data_fs
        idx = KNNWrapperSelector(target=0.9, n_neighbors=3, test_size=0.3).select(X[:, :3], y)
        assert len(idx) >= 1
        assert list(idx) == list(range(len(idx)))
