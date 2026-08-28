"""Vykreslovani vysledku cviceni 05 (kumulativni variance, rekonstrukce,
kompromis PCA, srovnani puvodniho a PCA prostoru).

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy figuru zavrou. Funkce ``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_variance(
    eigenvalues: np.ndarray,
    threshold: float,
    save_path: str | None = None,
) -> None:
    """Vykresli krivku kumulativni vysvetlene variance (v %) s prahem.

    Parametry
    ---------
    eigenvalues:
        Pole vlastnich cisel kovariancni matice. Predpoklada se serazeni
        sestupne; funkce je pro jistotu jeste serazeni.
    threshold:
        Vodorovna cara oznacujici pozadovane % kumulativni variance
        (napr. 95.0).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None`` (pak se figura jen zavre).

    Graf mirni puvodni ``select_components(draw=True)``: na ose x je pocet
    komponent, na ose y kumulativni vysvetlena variance v procentech.
    """
    eig = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    total = eig.sum()
    cumulative = np.cumsum(eig) / total * 100.0
    components = np.arange(1, eig.size + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(components, cumulative, marker="o", color="#1f77b4",
            label="kumulativni vysvetlena variance")
    ax.axhline(threshold, color="#d62728", linestyle="--",
               label=f"prah {threshold:.1f} %")
    ax.set_xlabel("pocet hlavnich komponent")
    ax.set_ylabel("kumulativni vysvetlena variance [%]")
    ax.set_title("Vyber poctu komponent podle kumulativni variance")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    _save_and_close(fig, save_path)


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Vykresli bodovy graf puvodnich vs. rekonstruovanych hodnot.

    Parametry
    ---------
    original:
        Puvodni data (libovolneho tvaru); pred vykreslenim se zplostí.
    reconstructed:
        Rekonstruovana data stejneho tvaru jako ``original``.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Cim blize lezi body na primce ``y = x``, tim vernejsi je rekonstrukce
    (mensi ztrata informace pri redukci dimenze).
    """
    orig_flat = np.asarray(original, dtype=np.float64).ravel()
    rec_flat = np.asarray(reconstructed, dtype=np.float64).ravel()

    lo = float(min(orig_flat.min(), rec_flat.min()))
    hi = float(max(orig_flat.max(), rec_flat.max()))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(orig_flat, rec_flat, s=8, alpha=0.3, color="#1f77b4")
    ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--",
            label="y = x (dokonala rekonstrukce)")
    ax.set_xlabel("puvodni hodnota")
    ax.set_ylabel("rekonstruovana hodnota")
    ax.set_title("Vernost rekonstrukce po PCA")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    _save_and_close(fig, save_path)


def plot_pca_tradeoff(
    thresholds: np.ndarray,
    silhouettes: np.ndarray,
    times: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Vykresli kompromis mezi kvalitou shlukovani a dobou behu podle
    zachovane variance (experiment Cil 2).

    Parametry
    ---------
    thresholds:
        Prahy zachovane variance [%] na ose x.
    silhouettes:
        Silhouette skore pro jednotlive prahy (leva osa y).
    times:
        Doba behu v sekundach pro jednotlive prahy (prava osa y).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Graf ma dve osy y: modre silhouette skore a cervenou dobu behu, obe
    proti prahu zachovane variance.
    """
    thr = np.asarray(thresholds, dtype=np.float64)
    sil = np.asarray(silhouettes, dtype=np.float64)
    tim = np.asarray(times, dtype=np.float64)

    fig, ax_sil = plt.subplots(figsize=(7, 4.5))
    color_sil = "#1f77b4"
    color_time = "#d62728"

    ax_sil.plot(thr, sil, marker="o", color=color_sil, label="silhouette skore")
    ax_sil.set_xlabel("zachovana variance [%]")
    ax_sil.set_ylabel("silhouette skore", color=color_sil)
    ax_sil.tick_params(axis="y", labelcolor=color_sil)
    ax_sil.grid(True, alpha=0.3)

    ax_time = ax_sil.twinx()
    ax_time.plot(thr, tim, marker="s", color=color_time, label="doba behu [s]")
    ax_time.set_ylabel("doba behu [s]", color=color_time)
    ax_time.tick_params(axis="y", labelcolor=color_time)

    ax_sil.set_title("Kompromis PCA: kvalita shlukovani vs. doba behu")

    _save_and_close(fig, save_path)


def plot_feature_vs_pca_space(
    x: np.ndarray,
    x_pca: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    feature_idx: tuple[int, int] = (0, 1),
    save_path: str | None = None,
) -> None:
    """Vykresli vedle sebe dva bodove grafy obarvene diagnozou ``y``:
    vlevo dva puvodni priznaky, vpravo prvni dve hlavni komponenty.

    Parametry
    ---------
    x:
        Puvodni priznakova matice, tvar ``(m, n_features)``.
    x_pca:
        Data po PCA projekci, tvar ``(m, k)`` s ``k >= 2``.
    y:
        Binarni cilova promenna delky ``m`` (1 = maligni, 0 = benigni).
    feature_names:
        Volitelne nazvy priznaku pro popisky os leveho panelu.
    feature_idx:
        Dvojice indexu puvodnich priznaku pro levy panel (vychozi prvni dva).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Poctivy caveat
    --------------
    PCA je **neucici (unsupervised)** metoda: maximalizuje rozptyl dat,
    **nikoli separaci trid**. Lepsi oddeleni malignich a benignich vzorku
    v prostoru hlavnich komponent je casty **vedlejsi efekt** (mezitridni
    rozptyl byva velky), nikoli cil ani zaruka. Obcas se tridy v PC
    prostoru neoddeli lepe -- i to je poucne. Supervizovanym protejskem,
    ktery separaci trid primo optimalizuje, je LDA (linearni
    diskriminacni analyza), ne PCA.
    """
    x = np.asarray(x, dtype=np.float64)
    x_pca = np.asarray(x_pca, dtype=np.float64)
    y = np.asarray(y)

    i, j = feature_idx
    classes = np.unique(y)
    colors = {0: "#2ca02c", 1: "#d62728"}
    labels = {0: "benigni (0)", 1: "maligni (1)"}

    fig, (ax_feat, ax_pc) = plt.subplots(1, 2, figsize=(12, 5))

    for cls in classes:
        mask = y == cls
        color = colors.get(int(cls), None)
        label = labels.get(int(cls), f"trida {cls}")
        ax_feat.scatter(x[mask, i], x[mask, j], s=12, alpha=0.6,
                        color=color, label=label)
        ax_pc.scatter(x_pca[mask, 0], x_pca[mask, 1], s=12, alpha=0.6,
                      color=color, label=label)

    if feature_names is not None:
        xlabel = feature_names[i]
        ylabel = feature_names[j]
    else:
        xlabel = f"priznak {i}"
        ylabel = f"priznak {j}"
    ax_feat.set_xlabel(xlabel)
    ax_feat.set_ylabel(ylabel)
    ax_feat.set_title("Puvodni priznakovy prostor")
    ax_feat.grid(True, alpha=0.3)
    ax_feat.legend(loc="best")

    ax_pc.set_xlabel("1. hlavni komponenta")
    ax_pc.set_ylabel("2. hlavni komponenta")
    ax_pc.set_title("Prostor hlavnich komponent (PCA je unsupervised)")
    ax_pc.grid(True, alpha=0.3)
    ax_pc.legend(loc="best")

    fig.suptitle("Puvodni prostor vs. prostor hlavnich komponent")
    fig.tight_layout()

    _save_and_close(fig, save_path)
