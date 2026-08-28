"""Vyber priznaku pro cviceni 05.

Obsahuje filtrovaci metodu zalozenou na dvouvyberovych statistickych testech
a obalove (wrapper) metody s hladovou doprednou selekci pres sdilenou smycku
(navrhovy vzor Template Method). Toto je misto, kde do kurzu poprve vstupuje
cilova promenna ``y``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Knihovni nastroje jsou zde pouzity zamerne a v omezene mire: k-means, kNN
# a silueta jsou periferie (siluetu jsme implementovali v cv3), tezistem cv5
# je logika vyberu priznaku, ne re-implementace techto algoritmu.
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class FeatureSelector(ABC):
    """Spolecne rozhrani vsech selektoru priznaku.

    Az dosud byl cely kurz bez ucitele (unsupervised) — pracoval pouze s matici
    priznaku ``X``. Zde poprve vstupuje do hry cilova promenna ``y``
    (diagnoza: 0 = benigni, 1 = maligni). Filtrovaci selektor ji pouziva
    ke srovnani obou diagnostickych skupin statistickym testem, wrapper ji
    pouziva jako cil optimalizace. Je to mekky prechod k uceni s ucitelem
    (supervised) — jeste bez klasifikace samotne.
    """

    @abstractmethod
    def select(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vybere priznaky a vrati pole jejich indexu.

        Parameters
        ----------
        x:
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y:
            Binarni cilovy vektor delky ``n_samples`` (hodnoty 0 a 1).

        Returns
        -------
        np.ndarray
            1D pole celociselnych indexu vybranych sloupcu ``x``
            (poradi odpovida sloupcum ``x``).
        """


class FilterSelector(FeatureSelector):
    """Filtrovaci vyber priznaku pomoci dvouvyberovych statistickych testu.

    Selektor hodnoti kazdy priznak nezavisle na ostatnich a ponechava jen ty,
    ktere se mezi skupinami ``y == 0`` a ``y == 1`` lisi ZAROVEN
    *vyznamne* (mala p-hodnota) a *podstatne* (dostatecna velikost ucinku).
    """

    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.5) -> None:
        """Inicializuje selektor.

        Parameters
        ----------
        alpha:
            Hladina vyznamnosti — priznak je "vyznamny", pokud p-hodnota
            testu je mensi nez ``alpha``.
        min_effect_size:
            Minimalni pozadovana absolutni hodnota velikosti ucinku
            (Cohenovo ``d``), aby byl priznak povazovan za "podstatny".
        """
        self.alpha = alpha
        self.min_effect_size = min_effect_size

    def select(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vybere priznaky, ktere obe skupiny oddeluji vyznamne i podstatne.

        Algoritmus (pro kazdy priznak ``j`` samostatne):

        1. Rozdel hodnoty sloupce ``x[:, j]`` na dve skupiny podle ``y``:
           ``a = x[y == 0, j]`` a ``b = x[y == 1, j]``.
        2. Otestuj normalitu obou skupin pomoci ``scipy.stats.shapiro``.
        3. Vyber test:
             - pokud OBE skupiny vypadaji normalne
               (``p_shapiro > self.alpha`` u obou), pouzij parametricky
               ``scipy.stats.ttest_ind``;
             - jinak pouzij neparametricky ``scipy.stats.mannwhitneyu``.
           Z testu si vezmi p-hodnotu ``p``.
        4. Spocitej velikost ucinku (Cohenovo ``d``) pomoci pripraveneho
           pomocneho ``self._cohens_d(a, b)``.
        5. Priznak PONECHEJ prave tehdy, kdyz
           ``p < self.alpha`` A ZAROVEN ``abs(d) >= self.min_effect_size``.

        Vrat indexy ponechanych priznaku jako ``np.ndarray`` typu int
        (napr. ``np.array([0, 3, 4])``); pokud neprojde zadny priznak,
        vrat prazdne pole ``np.array([], dtype=int)``.

        Proc DVE kriteria, ne jen p-hodnota
        -----------------------------------
        Mala p-hodnota rika, ze pozorovany rozdil mezi skupinami je
        *nepravdepodobne dilem nahody* — ale nerika NIC o tom, jak je ten
        rozdil VELKY. Pri dostatecne velkem poctu vzorku vyjde jako
        "statisticky vyznamny" i zcela trivialni, prakticky bezvyznamny
        rozdil. Velikost ucinku (Cohenovo ``d``) meri prave tuto magnitudu
        a chrani nas pred ponechanim priznaku, ktery skupiny oddeluje jen
        zanedbatelne. Toto dvojkriterium (vyznamnost A velikost ucinku) je
        pojmovym jadrem cele filtrovaci casti.

        Parameters
        ----------
        x:
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y:
            Binarni cilovy vektor delky ``n_samples`` (hodnoty 0 a 1).

        Returns
        -------
        np.ndarray
            Celociselne indexy ponechanych priznaku.
        """
        # assert  Ověřte, že X je 2D pole (X.ndim == 2)
        # assert  Ověřte, že len(y) == X.shape[0]
        # assert  Ověřte, že y je binarni (mnozina hodnot je podmnozinou {0, 1})
        raise NotImplementedError(
            "Úkol: implementujte filtrovaci vyber priznaku dvojkriteriem "
            "vyznamnost (p < alpha) A velikost ucinku (|d| >= min_effect_size); "
            "viz docstring."
        )

    def _cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Vrati Cohenovo ``d`` — velikost ucinku pro dva nezavisle vybery.

        Definice: ``d = (mean(a) - mean(b)) / pooled_std``, kde ``pooled_std``
        je sdruzena (pooled) smerodatna odchylka

            ``pooled_std = sqrt(((n1 - 1) * var(a) + (n2 - 1) * var(b))
                                / (n1 + n2 - 2))``

        s vyberovymi rozptyly ``var`` pocitanymi s ``ddof=1``.

        Tento pomocny je zamerne pripraven, aby ses mohl(a) soustredit na
        napojeni obou kriterii ve ``select``, ne na odvozovani vzorce.

        Parameters
        ----------
        a:
            Hodnoty prvni skupiny (1D pole).
        b:
            Hodnoty druhe skupiny (1D pole).

        Returns
        -------
        float
            Hodnota ``d``. Kladna, pokud ma ``a`` vyssi prumer nez ``b``;
            velka absolutni hodnota znaci vyrazne oddelene skupiny,
            hodnota blizka nule znaci prekryvajici se skupiny.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n1 = a.size
        n2 = b.size
        pooled_var = (
            (n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)
        ) / (n1 + n2 - 2)
        pooled_std = float(np.sqrt(pooled_var))
        return float((a.mean() - b.mean()) / pooled_std)


class WrapperSelector(FeatureSelector, ABC):
    """Obalovy (wrapper) vyber priznaku hladovou doprednou selekci.

    Navrhovy vzor Template Method: SPOLECNA je hladova smycka nize
    (``select``) — postupne pridava priznaky a sleduje skore podmnoziny.
    LISI SE pouze zpusob vypoctu skore (``_score``), ktery dodavaji
    potomci. Tim je smycka napsana jen jednou a jednotlive strategie
    skorovani jsou zamenitelne.
    """

    def __init__(self, target: float) -> None:
        """Inicializuje selektor.

        Parameters
        ----------
        target:
            Cilova hodnota skore. Smycka konci, jakmile skore podmnoziny
            priznaku dosahne alespon ``target``.
        """
        self.target = target

    def select(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Hladove pridava priznaky, dokud skore nedosahne ``self.target``.

        Pro ``i`` od 1 do ``n_features`` vezme prvnich ``i`` sloupcu
        ``x[:, :i]``, spocita ``score = self._score(x[:, :i], y)`` a jakmile
        ``score >= self.target``, vrati ``np.arange(i)``. Pokud cile nikdy
        nedosahne, vrati vsechny indexy ``np.arange(n_features)``.

        Smycka je spolecna vsem potomkum; meni se pouze ``_score``
        (Template Method).

        Parameters
        ----------
        x:
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y:
            Binarni cilovy vektor delky ``n_samples``.

        Returns
        -------
        np.ndarray
            Souvisly prefix indexu ``np.arange(i)`` (pripadne vsechny
            indexy, neni-li cile dosazeno).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        n_features = x.shape[1]
        for i in range(1, n_features + 1):
            score = self._score(x[:, :i], y)
            if score >= self.target:
                return np.arange(i)
        return np.arange(n_features)

    @abstractmethod
    def _score(self, x_subset: np.ndarray, y: np.ndarray) -> float:
        """Vrati skore podmnoziny priznaku ``X_subset`` (vetsi = lepsi).

        Parameters
        ----------
        x_subset:
            Podmnozina sloupcu ``x`` tvaru ``(n_samples, i)``.
        y:
            Binarni cilovy vektor delky ``n_samples``.

        Returns
        -------
        float
            Skore, ktere smycka ``select`` porovnava s ``self.target``.
        """


class SilhouetteWrapperSelector(WrapperSelector):
    """Wrapper skorujici podmnozinu priznaku siluetou k-means shlukovani.

    Kriterium bez ucitele (unsupervised): dobra podmnozina priznaku je ta,
    na ktere k-means najde dobre oddelene shluky (vysoka silueta).
    """

    def __init__(
        self,
        target: float,
        n_clusters: int = 2,
        random_state: int = 0,
    ) -> None:
        """Inicializuje selektor.

        Parameters
        ----------
        target:
            Cilova hodnota siluety (typicky v intervalu ``(0, 1]``).
        n_clusters:
            Pocet shluku pro ``KMeans``.
        random_state:
            Seed pro reprodukovatelnost ``KMeans``.
        """
        super().__init__(target)
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _score(self, x_subset: np.ndarray, y: np.ndarray) -> float:
        """Vrati siluetu k-means shlukovani na ``x_subset``.

        Postup:

        1. Nafituj ``sklearn.cluster.KMeans(n_clusters=self.n_clusters,
           random_state=self.random_state, n_init=10)`` na ``x_subset``.
        2. Vrat ``sklearn.metrics.silhouette_score(x_subset, labels)``,
           kde ``labels`` jsou prirazeni shluku z bodu 1.

        Knihovni volani jsou zde zamerna: siluetu jsme implementovali
        v cv3, tezistem cv5 je vyber priznaku, ne re-derivace siluety.
        ``y`` se ve skore NEPOUZIVA (kriterium je bez ucitele) — je v podpisu
        jen kvuli spolecnemu rozhrani.

        Parameters
        ----------
        x_subset:
            Podmnozina sloupcu ``x`` tvaru ``(n_samples, i)``.
        y:
            Binarni cilovy vektor (zde nevyuzity).

        Returns
        -------
        float
            Prumerna silueta (v intervalu ``[-1, 1]``).
        """
        # assert  Ověřte, že X_subset je 2D pole (X_subset.ndim == 2)
        # assert  Ověřte, že X_subset.shape[0] > self.n_clusters
        raise NotImplementedError(
            "Úkol: implementujte siluetove skore — nafitujte KMeans na "
            "X_subset a vratte silhouette_score(X_subset, labels); "
            "viz docstring."
        )


class KNNWrapperSelector(WrapperSelector):
    """Wrapper skorujici podmnozinu priznaku testovaci presnosti kNN.

    Kriterium s ucitelem (supervised): dobra podmnozina priznaku je ta, na
    ktere klasifikator k nejblizsich sousedu dobre predpovida diagnozu
    ``y`` na oddelenych testovacich datech.
    """

    def __init__(
        self,
        target: float,
        n_neighbors: int = 3,
        test_size: float = 0.2,
        random_state: int = 0,
    ) -> None:
        """Inicializuje selektor.

        Parameters
        ----------
        target:
            Cilova testovaci presnost (v intervalu ``(0, 1]``).
        n_neighbors:
            Pocet sousedu ``k`` pro ``KNeighborsClassifier``.
        test_size:
            Podil dat vyclenenych na test v ``train_test_split``.
        random_state:
            Seed pro reprodukovatelne rozdeleni na train/test.
        """
        super().__init__(target)
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state

    def _score(self, x_subset: np.ndarray, y: np.ndarray) -> float:
        """Vrati testovaci presnost kNN klasifikatoru na ``x_subset``.

        Postup:

        1. Rozdel data:
           ``X_tr, X_te, y_tr, y_te = sklearn.model_selection.train_test_split(
               x_subset, y, test_size=self.test_size,
               random_state=self.random_state, stratify=y)``.
        2. Nafituj ``sklearn.neighbors.KNeighborsClassifier(
           n_neighbors=self.n_neighbors)`` na ``X_tr, y_tr``.
        3. Vrat testovaci presnost ``clf.score(X_te, y_te)``.

        Knihovni volani jsou zde zamerna — tezistem cv5 je vyber priznaku,
        ne implementace kNN.

        Parameters
        ----------
        x_subset:
            Podmnozina sloupcu ``x`` tvaru ``(n_samples, i)``.
        y:
            Binarni cilovy vektor delky ``n_samples``.

        Returns
        -------
        float
            Podil spravne klasifikovanych testovacich vzorku (``[0, 1]``).
        """
        # assert  Ověřte, že X_subset je 2D pole (X_subset.ndim == 2)
        # assert  Ověřte, že len(y) == X_subset.shape[0]
        # assert  Ověřte, že y obsahuje aspon 2 tridy
        raise NotImplementedError(
            "Úkol: implementujte skore jako testovaci presnost kNN — "
            "train_test_split, fit KNeighborsClassifier, vratte clf.score; "
            "viz docstring."
        )
