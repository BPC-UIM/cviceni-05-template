"""Induktivni PCA (analyza hlavnich komponent) implementovana od zakladu.

PCA vybira smery s NEJVETSIM rozptylem (nejvetsi vlastni cisla kovariancni
matice) — opacny konec spektra nez spektralni shlukovani v cv4, ktere naopak
bere vlastni cisla nejmensi. Model je induktivni: ``fit`` nauci vlastni vektory
a ``transform`` je aplikuje na nova data, diky cemuz lze naucený model ulozit
a pozdeji znovu pouzit.
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Induktivni PCA ve stylu sklearn (nededi z niceho).

    ``fit`` z trenovacich dat nauci prumer, vlastni cisla a vlastni vektory
    (hlavni komponenty) kovariancni matice; ``transform`` timto naucenym
    pravidlem promita libovolna nova data. To je zamerny kontrast s
    transduktivnim shlukovanim z cv4, ktere zadne prenositelne pravidlo nema.

    Znamenko vlastniho vektoru je libovolne: ``eigh`` muze vratit komponentu
    i s obracenym znamenkem a obe varianty jsou spravne. Testy proto nikdy
    neporovnavaji surove vlastni vektory, ale vysvetleny rozptyl, rekonstrukcni
    chybu nebo hodnoty v absolutni hodnote.

    Atributy (naplneny az v ``fit``, do te doby ``None``):
        components_: np.ndarray tvaru (n_features, n_components_) — hlavni
            komponenty ulozene ve sloupcich.
        mean_: np.ndarray tvaru (n_features,) — prumer trenovacich dat.
        eigenvalues_: np.ndarray — vlastni cisla serazena SESTUPNE.
        explained_variance_ratio_: np.ndarray — podil vysvetleneho rozptylu
            pripadajici na jednotliva vlastni cisla (soucet = 1).
        n_components_: int — skutecny pocet ponechanych komponent.
    """

    def __init__(
        self,
        variance_threshold: float | None = 95.0,
        n_components: int | None = None,
    ) -> None:
        """Nastavi hyperparametry PCA.

        Pocet komponent rídí prave jeden z parametru: pokud je ``n_components``
        None, pouzije se prah kumulativniho rozptylu ``variance_threshold``
        (v procentech); jinak se pouzije pevny pocet ``n_components``.

        Args:
            variance_threshold: Pozadovany podil kumulativniho vysvetleneho
                rozptylu v procentech (napr. 95.0). Uplatni se pouze tehdy,
                je-li ``n_components`` None.
            n_components: Pevny pocet hlavnich komponent, nebo None.
        """
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        # Atributy naucene ve fit(); pred zavolanim fit() jsou None.
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def _covariance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Spocita kovariancni matici priznaku ve vektorizovane podobe.

        Args:
            x: np.ndarray tvaru (m, n_features) — m vzorku v radcich,
                n_features priznaku ve sloupcich.

        Returns:
            np.ndarray tvaru (n_features, n_features) — symetricka kovariancni
            matice.

        Kriterium: sloupce nejprve vycentrujte (odectete prumer kazdeho
        sloupce), oznacte je ``xc`` a vratte ``(xc.T @ xc) / (m - 1)`` — tedy
        jednim maticovym soucinem, bez pythonovskych smycek pres priznaky.
        Delitel ``m - 1`` odpovida vyberovemu (nevychylenemu) odhadu rozptylu
        a shoduje se s ``np.cov(x, rowvar=False)``.
        """
        # assert  Overte, ze x je typu np.ndarray.
        # assert  Overte, ze x ma 2 rozmery (matice vzorky x priznaky).
        # assert  Overte, ze x obsahuje alespon 2 vzorky (m >= 2).
        raise NotImplementedError(
            "Úkol: spoctete kovariancni matici (n_features x n_features) "
            "vektorizovane — vycentrujte sloupce a vratte (xc.T @ xc) / (m - 1)."
        )

    def _select_components(self, eigenvalues: np.ndarray) -> int:
        """Urci pocet komponent podle prahu kumulativniho rozptylu.

        Args:
            eigenvalues: np.ndarray — vlastni cisla JIZ SERAZENA SESTUPNE.

        Returns:
            int — nejmensi pocet komponent, jejichz kumulativni podil
            vysvetleneho rozptylu dosahne ``self.variance_threshold`` procent.

        Kriterium: spoctete podil ``eigenvalues / eigenvalues.sum()``, projdete
        jej od nejvetsi hodnoty, kumulativne scitejte a vratte prvni pocet
        komponent, u nehoz kumulativni soucet (v procentech) dosahne prahu.

        PCA si zamerne nechava vlastni cisla NEJVETSI — tedy smery s nejvetsim
        rozptylem. To je opacny konec spektra nez spektralni shlukovani v cv4,
        ktere stejnou vlastni dekompozici pouziva, ale bere vlastni cisla
        NEJMENSI. Znamenko odpovidajiciho vlastniho vektoru pritom nehraje
        roli, do vyberu vstupuje jen velikost vlastniho cisla.
        """
        # assert  Overte, ze eigenvalues je 1D np.ndarray.
        # assert  Overte, ze eigenvalues jsou serazena sestupne.
        # assert  Overte, ze 0 < self.variance_threshold <= 100.
        raise NotImplementedError(
            "Úkol: vratte nejmensi pocet komponent, jejichz kumulativni "
            "vysvetleny rozptyl dosahne self.variance_threshold procent."
        )

    def fit(self, x: np.ndarray) -> "PCA":
        """Nauci PCA model z trenovacich dat ``x``.

        Args:
            x: np.ndarray tvaru (m, n_features) — trenovaci data.

        Returns:
            PCA — tato instance (``self``) s naplnenymi atributy
            ``mean_``, ``eigenvalues_``, ``explained_variance_ratio_``,
            ``n_components_`` a ``components_``.

        Postup: ulozi prumer, sestavi kovariancni matici, provede jeji vlastni
        dekompozici (``np.linalg.eigh``), sesradi vysledky SESTUPNE podle
        vlastnich cisel, urci pocet komponent a ulozi prislusne vlastni vektory
        jako sloupce ``components_``.
        """
        self.mean_ = x.mean(axis=0)

        cov = self._covariance_matrix(x)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # === Klicovy krok: razeni SESTUPNE — "nejvetsi rozptyl prvni" ===
        # PCA si nechava vlastni cisla NEJVETSI (nejvic vysvetleneho rozptylu).
        # Kontrast s cv4: spektralni shlukovani radi stejnou matici, ale bere
        # vlastni cisla NEJMENSI (nejhladsi smery grafu). Stejna mechanika,
        # opacny konec spektra.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.eigenvalues_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()

        if self.n_components is not None:
            self.n_components_ = self.n_components
        else:
            self.n_components_ = self._select_components(self.eigenvalues_)

        # Hlavni komponenty jsou vlastni vektory ve sloupcich:
        # tvar (n_features, n_components_).
        self.components_ = eigenvectors[:, : self.n_components_]
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Nauci model a rovnou vrati promitnuta data.

        Args:
            x: np.ndarray tvaru (m, n_features) — trenovaci data.

        Returns:
            np.ndarray tvaru (m, n_components_) — data v prostoru hlavnich
            komponent.
        """
        return self.fit(x).transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Promitne data ``x`` do prostoru naucenych hlavnich komponent.

        Args:
            x: np.ndarray tvaru (m, n_features) — data se stejnym poctem
                priznaku jako trenovaci mnozina.

        Returns:
            np.ndarray tvaru (m, n_components_) — souradnice vzorku v prostoru
            hlavnich komponent.

        Kriterium: data nejprve vycentrujte naucenym prumerem a pak je
        promitnete na komponenty: ``(X - self.mean_) @ self.components_``.
        Protoze znamenko vlastniho vektoru je libovolne, muze se znamenko
        jednotlivych sloupcu vysledku lisit od jine implementace (napr.
        sklearn) — to je v poradku.
        """
        # assert  Overte, ze model je naucen (self.components_ neni None).
        # assert  Overte, ze X.shape[1] == self.mean_.shape[0].
        raise NotImplementedError(
            "Úkol: promitnete vycentrovana data na self.components_, "
            "tedy vratte (X - self.mean_) @ self.components_."
        )

    def inverse_transform(self, x_pca: np.ndarray) -> np.ndarray:
        """Rekonstruuje data z prostoru hlavnich komponent zpet do puvodniho.

        Args:
            X_pca: np.ndarray tvaru (m, n_components_) — data v prostoru
                hlavnich komponent.

        Returns:
            np.ndarray tvaru (m, n_features) — priblizna rekonstrukce v
            puvodnim prostoru priznaku.

        Kriterium: vratte ``X_pca @ self.components_.T + self.mean_``. Pokud
        jsou ponechany vsechny komponenty, plati
        ``inverse_transform(transform(X)) ~= X`` az na numerickou chybu; pri
        redukci poctu komponent je rekonstrukce ztratova.
        """
        # assert  Overte, ze model je naucen (self.components_ neni None).
        # assert  Overte, ze X_pca.shape[1] == self.n_components_.
        raise NotImplementedError(
            "Úkol: rekonstruujte data zpet do puvodniho prostoru, "
            "tedy vratte X_pca @ self.components_.T + self.mean_."
        )

    def save(self, path: str) -> None:
        """Ulozi naucený model do souboru ``.npz`` pomoci ``np.savez``.

        Args:
            path: Cesta k vystupnimu souboru ``.npz``.

        Returns:
            None.

        Naucený model NENI program — je to hrstka naucenych poli, ktera
        prezije beh programu. Prave to umoznuje rozdeleni "nauc ted / pouzij
        pozdeji": model se jednou nauci, ulozi a kdykoli pozdeji nacte a
        pouzije na nova data bez opakovaneho trenovani.

        Je na vas rozhodnout, ktera pole model tvori. Minimalne ``components_``
        a ``mean_`` (bez nich nelze ``transform`` ani ``inverse_transform``
        provest); dale ulozte tolik, aby sel plne obnovit stav modelu —
        napr. ``explained_variance_ratio_``, ``eigenvalues_`` a
        ``n_components_``.

        Nepouzivejte ``pickle``: je neprehledny (nevidite, co soubor obsahuje)
        a nebezpecny (nacteni cizi pickle soubor muze spustit libovolny kod).
        Format ``.npz`` uklada jen ciselna pole.
        """
        # assert  Overte, ze model je naucen (self.components_ neni None).
        raise NotImplementedError(
            "Úkol: ulozte naucena pole modelu do souboru .npz pomoci np.savez "
            "(minimalne components_ a mean_, plus dalsi pole pro obnovu stavu)."
        )

    # @classmethod: metoda dostane misto instance (self) samotnou tridu (cls).
    # Diky tomu ji lze volat primo na tride bez existujici instance
    # (PCA.load("model.npz")) a uvnitr pres cls(...) vyrobit novou instanci —
    # jde o tzv. alternativni konstruktor (druhy zpusob, jak vytvorit PCA).
    @classmethod
    def load(cls, path: str) -> "PCA":
        """Nacte model ulozený metodou ``save`` a vrati hotovou instanci.

        Args:
            path: Cesta k souboru ``.npz`` vytvorenemu metodou ``save``.

        Returns:
            PCA — nova instance s obnovenymi naucenymi atributy, pripravena
            volat ``transform`` na nova data bez opakovaneho ``fit``.

        Toto je inverzni operace k ``save``: precte pole ze souboru ``.npz``,
        vytvori novou instanci ``PCA`` a nastavi ji naucene atributy
        (``components_``, ``mean_`` a dalsi ulozena pole).

        Funguje to jen proto, ze PCA je INDUKTIVNI — ``fit`` nauci prenositelne
        pravidlo (vlastni vektory), ktere ma smysl ulozit a znovu pouzit.
        Transduktivni shlukovani z cv4 zadny takový znovupouzitelny model nema,
        proto tam obdoba ``load`` neexistuje.
        """
        # assert  Overte, ze soubor path existuje a obsahuje pole 'components_' a 'mean_'.
        # assert  Overte, ze obnovene components_ maji 2 rozmery.
        raise NotImplementedError(
            "Úkol: nactete pole ze souboru .npz, vytvorte novou instanci PCA, "
            "obnovte jeji naucene atributy a vratte ji."
        )
