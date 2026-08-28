# Cvičení 5: Výběr příznaků a PCA

Páté praktické cvičení předmětu **Umělá inteligence v medicíně** uzavírá blok učení bez učitele (Cvičení 01 statistika a předzpracování, Cvičení 02 hierarchické shlukování, Cvičení 03 k-means a fuzzy c-means, Cvičení 04 DBSCAN a spektrální shlukování) a otevírá dvě témata, která provázejí zbytek kurzu: **redukci dimenze** a **výběr příznaků**. Vzorovou datovou sadou je **Breast Cancer Wisconsin** — 569 nádorů popsaných 30 číselnými příznaky a binární diagnózou (1 = maligní, 0 = benigní).

Student implementuje **PCA (analýzu hlavních komponent) od základů** — kovarianční matici, vlastní rozklad, výběr počtu komponent, projekci, rekonstrukci a uložení/načtení natrénovaného modelu — a **rodinu selektorů příznaků**: filtrační selektor postavený na dvouvýběrových statistických testech a dva obalové (wrapper) selektory sdílející jednu hladovou smyčku (návrhový vzor Template Method).

Cvičení je zároveň **prvním místem, kde do úlohy vstupuje cílová proměnná** `y`. Filtrační výběr příznaků porovnává obě diagnostické skupiny statistickým testem — používá tedy popisky, aniž by ještě prováděl klasifikaci. Jde o plynulý přechod k učení s učitelem. Druhý podstatný kontrast s předchozím cvičením: zatímco DBSCAN a spektrální shlukování z Cvičení 04 jsou **transduktivní** (produkují pouze rozdělení trénovacích dat), PCA je **induktivní** — `fit()` se naučí přenositelné pravidlo, které `transform()` aplikuje i na data nová. Právě proto zde poprvé dává smysl model **uložit na disk a později znovu použít**.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Doplňkové (papírové) příklady](#doplňkové-papírové-příklady)
9. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Implementuje PCA od základů** — sestaví kovarianční matici vektorově, provede vlastní rozklad, vybere počet komponent podle kumulativní vysvětlené variance a naprogramuje projekci i zpětnou rekonstrukci.
2. **Rozumí tomu, že PCA vybírá směry s největším rozptylem** — tedy vlastní vektory příslušející **největším** vlastním číslům, což je přesně opačný konec spektra než u spektrálního shlukování v Cvičení 04 (nejmenší vlastní čísla Laplaciánu). Tentýž lineárně-algebraický aparát, opačný cíl.
3. **Chápe rozdíl mezi induktivním a transduktivním učením** — a tedy důvod, proč lze natrénovanou PCA uložit do souboru a znovu použít na nová data (`save` / `load` do `.npz`), zatímco shlukování z Cvičení 04 takový znovupoužitelný model nemá.
4. **Osvěží si statistické testování** — nulová hypotéza, p-hodnota, hladina významnosti, předpoklad normality, volba mezi parametrickým a neparametrickým testem a především **velikost účinku** (effect size): malá p-hodnota neznamená velký rozdíl.
5. **Implementuje filtrační výběr příznaků s dvojím kritériem** — příznak se ponechá, jen když se skupiny liší **současně významně** (`p < alpha`) **i podstatně** (`|Cohenovo d| ≥ min_effect_size`).
6. **Implementuje obalový (wrapper) výběr příznaků jako Template Method** — jedna hladová smyčka, dvě záměnná kritéria skóre: silueta k-means (bez učitele) a přesnost kNN (s učitelem).
7. **Názorně si ověří vizuální přínos redukce dimenze** — a zároveň chápe jeho meze: lepší oddělení tříd v prostoru hlavních komponent je častým vedlejším efektem, nikoli cílem ani zárukou (k tomu slouží LDA).
8. **Pracuje s typovanou konfigurací** — čte hyperparametry z `config.yaml` přes dataclassy (`cfg.pca.variance_threshold` místo `cfg["pca"]["variance_threshold"]`), stejný vzor jako v Cvičení 03 a 04.

---

## Struktura repozitáře

```
cviceni-05-template/
├── cviceni_05.py            # Hlavní pipeline — spusťte pro průběžné ověření (PŘEDVYPLNĚNO)
├── config.yaml              # Konfigurace experimentu (YAML)
├── priklady_05.md           # Papírové (teoretické) příklady — BEZ řešení v repozitáři
├── requirements.txt         # Python závislosti (zamčené verze)
├── .gitignore
├── src/
│   ├── __init__.py          # Re-exporty balíčku (neupravujte)
│   ├── pca.py               # PCA — ÚKOL: 6 metod (stavební bloky); __init__ a fit() předvyplněny
│   └── feature_selection.py # FeatureSelector (ABC) + FilterSelector + WrapperSelector — ÚKOL: 3 metody
├── dataio/
│   ├── __init__.py          # Re-exporty balíčku (neupravujte)
│   ├── loader.py            # load_breast_cancer_data() — načtení a standardizace dat (předvyplněno)
│   ├── config_manager.py    # Dataclassy + load_config + validate_config (předvyplněno)
│   └── plotting.py          # Vizualizace: kumulativní variance, rekonstrukce, srovnání prostorů (předvyplněno)
├── models/                  # Sem pipeline uloží natrénovaný PCA model (models/pca_model.npz); negitováno
│   └── .gitkeep
├── graphs/                  # Výstupní složka pro grafy (generuje se automaticky)
└── test_cviceni_05.py       # Automatické testy (pytest)
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`, `dataio/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje veřejné API. Díky tomu lze psát `from src import PCA` místo `from src.pca import PCA`. **Tyto soubory neupravujte.**

> **Celý balíček `dataio/` je předvyplněn** — načítání dat, konfigurace i vykreslování. Žádný `NotImplementedError` v `dataio/` nenajdete. Studentská práce je soustředěna výhradně do `src/pca.py` a `src/feature_selection.py`.

> **Data se nenačítají ze souboru** — `load_breast_cancer_data()` je bere přímo z `scikit-learn`, repozitář proto žádnou složku `data/` neobsahuje. Jediný artefakt, který běh vytvoří vedle grafů, je natrénovaný model `models/pca_model.npz` (metoda `PCA.save`). Složky `graphs/` i `models/` zůstávají ve verzování prázdné (přes `.gitkeep`), jejich obsah je v `.gitignore`.

> **Žádný `src/distance.py`.** Na rozdíl od Cvičení 02–04 se v tomto cvičení nekopíruje třída `Distance` z Cvičení 01. PCA staví na kovarianci, k-means a kNN uvnitř wrapperů volají scikit-learn — nikde se nepočítá žádná párová vzdálenost, kterou by student implementoval. Z téhož důvodu nejsou v testech ani žádné `DummyDistance`.

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

Cvičení používá `numpy`, `scipy` (statistické testy), `scikit-learn` (dataset, k-means, kNN, silueta), `matplotlib` (grafy), `pyyaml` (konfigurace) a `pytest` (testy). Verze jsou v `requirements.txt` zamčené.

### 3. Spuštění

```bash
python cviceni_05.py
```

Pipeline načte konfiguraci a data, spustí výběr příznaků, natrénuje PCA, uloží a znovu načte model, vykreslí srovnání původního a redukovaného prostoru a nakonec spustí experiment o kompromisu mezi zachovanou variancí a kvalitou shlukování. Každá fáze je obalena samostatným blokem `try / except NotImplementedError`, takže **nedokončený úkol jednu fázi přeskočí, ale nezablokuje ostatní** — dostanete co nejvíce zpětné vazby o tom, co ještě zbývá doplnit. Nedokončená fáze vypíše hlášku začínající `[NENI HOTOVO] Úkol: …` a pipeline pokračuje dál. Ve stavu kostry tedy `python cviceni_05.py` **nikdy neskončí nezpracovaným tracebackem**.

> **Jediná výjimka — načtení konfigurace.** `load_config()` volá `validate_config()`; kdyby v `config.yaml` byla nesmyslná hodnota, pipeline se korektně ukončí hláškou `[CHYBA KONFIGURACE]` hned na začátku. Obě funkce jsou ale předvyplněné, takže při nezměněné konfiguraci tento stav nenastane.

> **Jen jeden druh `NotImplementedError`.** V Cvičení 04 existovaly dva — studentský úkol (`Úkol: …`) a trvalé architektonické omezení (`predict()` u transduktivních metod). **V tomto cvičení je druh jediný:** každá zpráva začíná `Úkol:` a každou je potřeba doplnit. PCA je induktivní, takže `transform()` na nová data je dobře definovaný — není co trvale zakazovat.

Jednotlivé metody lze mezitím ověřovat přes `pytest`, viz [Lokální testování](#lokální-testování).

---

## Teoretický základ

### 1. Proč výběr příznaků a redukce dimenze?

Reálná medicínská data mají často desítky až tisíce příznaků, přičemž mnohé jsou navzájem korelované, zašuměné nebo pro daný úkol nepodstatné. Vysoká dimenze přináší tři problémy:

- **Prokletí dimenzionality** — s rostoucím počtem příznaků roste objem prostoru exponenciálně, data se stávají řídkými a pojmy jako „blízkost" ztrácejí rozlišovací sílu (přímý dopad na k-means, kNN, DBSCAN).
- **Přeučení a šum** — nepodstatné příznaky přidávají volnost, kterou model použije k zapamatování šumu místo signálu.
- **Výpočetní cena a interpretovatelnost** — méně příznaků znamená rychlejší trénink a srozumitelnější model.

Existují dvě komplementární strategie, jak dimenzi snížit:

| Přístup | Princip | V tomto cvičení |
|:---|:---|:---|
| **Extrakce příznaků** | Vytvoří **nové** příznaky jako kombinace původních (a ponechá jen několik nejinformativnějších). | **PCA** — nové osy jsou lineární kombinace původních 30 příznaků. |
| **Výběr příznaků** | Ponechá **podmnožinu původních** příznaků, zbytek zahodí. Zachovává interpretovatelnost. | **Filtrační** a **obalový** selektor. |

Výběr příznaků se dále dělí podle toho, jak úzce spolupracuje s modelem:

| Typ | Jak funguje | Výhody | Nevýhody |
|:---|:---|:---|:---|
| **Filtrační (filter)** | Ohodnotí každý příznak samostatně statistikou (zde: rozdíl mezi diagnostickými skupinami). Nezávislý na modelu. | Rychlý, škálovatelný, model-agnostický. | Ignoruje interakce mezi příznaky. |
| **Obalový (wrapper)** | Opakovaně trénuje model na různých podmnožinách a vybírá tu, která maximalizuje výkon. | Zohledňuje interakce, optimalizuje přímo cílovou metriku. | Výpočetně drahý, riziko přeučení na validační metriku. |
| **Vestavěný (embedded)** | Výběr je součástí tréninku (L1/Lasso regularizace, důležitost příznaků ve stromech). | Kompromis rychlosti a kvality. | Mimo rozsah tohoto cvičení. |

---

### 2. PCA — analýza hlavních komponent

Mějme datovou matici $X \in \mathbb{R}^{m \times n}$ — $m$ vzorků v řádcích, $n$ příznaků ve sloupcích. PCA hledá novou ortonormální bázi prostoru příznaků takovou, že **první osa nese největší možný rozptyl dat, druhá největší z rozptylu zbývajícího** (a je kolmá na první) a tak dále. Tyto osy se nazývají **hlavní komponenty**.

#### 2.1 Centrování a kovarianční matice

Nejprve spočítáme průměr každého příznaku a data vycentrujeme:

$$\boldsymbol{\mu} = \frac{1}{m}\sum_{i=1}^{m}\mathbf{x}_i, \qquad X_c = X - \mathbf{1}\,\boldsymbol{\mu}^{\!\top}$$

**Kovarianční matice** vycentrovaných dat je symetrická matice $n \times n$:

$$C = \frac{1}{m-1}\,X_c^{\!\top} X_c$$

Prvek $C_{jk}$ je kovariance mezi příznakem $j$ a $k$; na diagonále jsou rozptyly jednotlivých příznaků. Dělitel $m-1$ (nikoli $m$) dává nestranný výběrový odhad a shoduje se s `numpy.cov(X, rowvar=False)`. Výpočet provádějte **vektorově** — jedním maticovým součinem, bez smyček přes příznaky.

#### 2.2 Vlastní rozklad

Protože $C$ je symetrická a pozitivně semidefinitní, má reálný vlastní rozklad

$$C = V \Lambda V^{\!\top}, \qquad C\mathbf{v}_j = \lambda_j \mathbf{v}_j$$

kde sloupce $V$ jsou navzájem kolmé jednotkové **vlastní vektory** (hlavní směry) a $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$ obsahuje **vlastní čísla** — rozptyly dat podél příslušných směrů. Používáme `numpy.linalg.eigh` (varianta pro symetrické matice: rychlejší, stabilnější, vrací reálná vlastní čísla).

`eigh` vrací vlastní čísla **vzestupně**, proto je ve `fit()` seřadíme **sestupně**:

$$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$$

> **Znaménko vlastního vektoru je libovolné.** Pokud $\mathbf{v}$ je vlastní vektor, je jím i $-\mathbf{v}$ se stejným vlastním číslem. Různé knihovny (a různé běhy) proto vrací komponenty s náhodně obrácenými znaménky — to je stejný druh nejednoznačnosti jako arbitrární ID slučování v Cvičení 02 nebo permutace popisků shluků v Cvičení 03 a 04. **Testy nikdy neporovnávají syrové vlastní vektory** se scikit-learn — porovnávají podíly vysvětlené variance, chybu rekonstrukce nebo hodnoty v absolutní hodnotě.

> **Proč se výsledek nepatrně liší od `sklearn.decomposition.PCA`.** scikit-learn nepočítá vlastní rozklad kovarianční matice, ale **singulární rozklad (SVD)** vycentrované matice $X_c = U S V^{\!\top}$. Obě cesty jsou matematicky ekvivalentní — platí $\lambda_j = s_j^2 / (m-1)$ a sloupce $V$ jsou tytéž hlavní komponenty — ale SVD je numericky stabilnější, protože vůbec nesestavuje součin $X_c^{\!\top} X_c$, a tím nezhoršuje podmíněnost úlohy. Rozdíly v posledních platných číslicích a obrácená znaménka komponent jsou proto očekávané; testy z téhož důvodu srovnávají podíly rozptylu a chybu rekonstrukce, ne surové hodnoty.

#### 2.3 Podíl vysvětlené variance a výběr počtu komponent

Celkový rozptyl dat je $\sum_{l=1}^{n}\lambda_l$. **Podíl vysvětlené variance** $j$-té komponenty a **kumulativní** vysvětlená variance prvních $k$ komponent jsou

$$\mathrm{EVR}_j = \frac{\lambda_j}{\sum_{l=1}^{n}\lambda_l}, \qquad \mathrm{CEV}_k = \sum_{j=1}^{k}\mathrm{EVR}_j$$

Počet komponent zvolíme jako **nejmenší $k$**, pro které kumulativní vysvětlená variance dosáhne zadaného prahu $\tau$ (v procentech, např. $\tau = 95$):

$$k^\star = \min\bigl\{\, k \;:\; 100 \cdot \mathrm{CEV}_k \ge \tau \,\bigr\}$$

> **PCA vybírá NEJVĚTŠÍ vlastní čísla — opak spektrálního shlukování.** Obě metody staví na vlastním rozkladu symetrické matice, ale míří na opačné konce spektra:
>
> | | Matice | Vybírá vlastní vektory pro… | Proč |
> |:---|:---|:---|:---|
> | **Spektrální shlukování (Cvičení 04)** | Laplacián grafu $L = D - W$ | …**nejmenší** vlastní čísla | směry „nejslabších řezů" grafu = hranice mezi shluky |
> | **PCA (Cvičení 05)** | Kovarianční matice $C$ | …**největší** vlastní čísla | směry s největším rozptylem = nejvíce informace |
>
> Záměna pořadí metodu tiše naruší — nevyvolá se žádná výjimka, výsledek je však nesmyslný.

#### 2.4 Projekce a rekonstrukce

Ponecháme prvních $k^\star$ vlastních vektorů jako sloupce matice $V_k \in \mathbb{R}^{n \times k^\star}$. **Projekce** (metoda `transform`) přemapuje data do prostoru hlavních komponent:

$$T = X_c V_k = (X - \boldsymbol{\mu})\,V_k \;\in\; \mathbb{R}^{m \times k^\star}$$

**Zpětná rekonstrukce** (metoda `inverse_transform`) vrací data z prostoru komponent do původního prostoru příznaků:

$$\hat{X} = T\,V_k^{\!\top} + \boldsymbol{\mu}$$

Jsou-li ponechány **všechny** komponenty ($k^\star = n$), platí $V_k V_k^{\!\top} = I$ a rekonstrukce je přesná: $\hat{X} = X$ až na numerickou chybu. Při $k^\star < n$ je rekonstrukce ztrátová — zahozená „energie" odpovídá součtu vynechaných vlastních čísel.

#### 2.5 Standardizace před PCA

PCA je **citlivá na měřítko**: příznak měřený v tisících (např. `mean area`) by bez úpravy dominoval kovarianční matici a první komponenty by kopírovaly jen jeho jednotky, ne jeho informační obsah. Proto data předem **z-skórujeme** (v `dataio/loader.py`, řízeno `config.yaml`):

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Po standardizaci má každý příznak nulový průměr a jednotkový rozptyl a **kovarianční matice standardizovaných dat je rovna korelační matici dat původních**. PCA nad standardizovanými daty je tedy PCA nad korelační maticí — měřítkově neutrální a v tomto cvičení jednotně zvolená varianta.

#### 2.6 Meze PCA — linearita a interpretovatelnost

Hlavní komponenty jsou **lineární kombinace** původních příznaků. Leží-li data na zakřivené varietě — například dvě soustředné kružnice z Cvičení 04 — žádná lineární projekce je neoddělí a PCA v tomto smyslu neuspěje (první komponenty zachytí rozpětí prstence, ne příslušnost ke kružnici). Na nelineární strukturu se používá kernel PCA, t-SNE, UMAP nebo autoenkodéry; ty jsou mimo rozsah tohoto cvičení, PCA je však jejich lineární základ.

Druhou cenou je **ztráta interpretovatelnosti**. Komponenta `0.31·mean radius − 0.28·worst texture + …` nemá přímý klinický význam, na rozdíl od jednotlivého příznaku ponechaného filtrem. Volba mezi extrakcí příznaků (PCA) a jejich výběrem (filtr, wrapper) je proto i volbou mezi kompresí a srozumitelností modelu.

---

### 3. Induktivní model a jeho persistence

| | k-means (Cvičení 03) | DBSCAN / spektrální (Cvičení 04) | **PCA (Cvičení 05)** |
|:---|:---|:---|:---|
| Co se „naučí" | těžiště v prostoru příznaků | jen rozdělení trénovacích dat | **průměr `mean_` a hlavní komponenty `components_`** |
| Použití na nová data | ano (`predict`) | ne | **ano (`transform`)** |
| Charakter | induktivní | transduktivní | **induktivní** |

PCA se ve `fit()` naučí **přenositelné pravidlo** — vektor průměrů a matici vlastních vektorů. Jde o několik polí čísel; jakmile jsou známa, lze jimi promítnout libovolná nová data bez opětovného trénování. Proto zde poprvé v kurzu dává smysl model **uložit na disk a později znovu použít**:

- **`save(path)`** zapíše naučená pole do souboru `.npz` funkcí `numpy.savez`. Student sám rozhodne, která pole model tvoří — minimálně `components_` a `mean_`, dále tolik, aby šel plně obnovit stav (`explained_variance_ratio_`, `eigenvalues_`, `n_components_`).
- **`load(path)`** je `@classmethod`: přečte `.npz`, vytvoří novou instanci `PCA`, nastaví jí naučené atributy a vrátí ji připravenou k `transform`.

> **Nikdy nepoužívejte `pickle`.** Je neprůhledný (z binárního souboru nepoznáte, co obsahuje) a nebezpečný (načtení cizího pickle souboru může spustit libovolný kód). Formát `.npz` ukládá pouze pojmenovaná číselná pole a je bezpečný ke sdílení.

Pipeline `cviceni_05.py` tento postup předvádí: po `fit()` model uloží do `models/pca_model.npz`, znovu jej načte třídní metodou `PCA.load` a ověří, že `restored.transform(X)` dává tentýž výsledek jako původní instance. Toto rozdělení „natrénuj nyní, použij později" je jádrem 4. cíle cvičení: model **je** uložená množina naučených parametrů.

---

### 4. Statistické testování — filtrační výběr příznaků

Filtrační selektor projde příznaky jeden po druhém a u každého se ptá: **liší se jeho hodnoty mezi maligní a benigní skupinou natolik, že to stojí za pozornost?** Odpověď skládá ze dvou nezávislých kritérií.

#### 4.1 Nulová hypotéza, p-hodnota, hladina významnosti

**Nulová hypotéza** $H_0$ tvrdí, že daný příznak má v obou skupinách **stejné rozdělení** (resp. stejnou střední hodnotu). **p-hodnota** je pravděpodobnost, že bychom za platnosti $H_0$ pozorovali data alespoň tak extrémní jako naše. Malá p-hodnota znamená, že pozorovaný rozdíl je **nepravděpodobně dílem náhody**, a $H_0$ zamítáme.

Rozhodovací práh je **hladina významnosti** $\alpha$ (zde `alpha = 0.05`):

$$p < \alpha \;\Rightarrow\; \text{rozdíl je statisticky významný (zamítáme } H_0\text{).}$$

#### 4.2 Předpoklad normality — parametrický, nebo neparametrický test

Parametrický dvouvýběrový **t-test** předpokládá, že hodnoty v každé skupině pocházejí přibližně z **normálního rozdělení**. Tento předpoklad ověříme **Shapiro–Wilkovým testem** (`scipy.stats.shapiro`), jehož $H_0$ zní „data jsou normální":

- je-li Shapiro `p > alpha` u **obou** skupin → normalitu nezamítáme → použijeme **t-test** (viz odd. 4.3);
- jinak → použijeme neparametrický **Mann–Whitneyův U test** (`scipy.stats.mannwhitneyu`), který pracuje s pořadími hodnot a žádné rozdělení nepředpokládá.

> **Shapiro–Wilk na velkých vzorcích.** Test normality podléhá témuž jevu jako každý jiný: při stovkách vzorků ve skupině (což je případ dat Breast Cancer) zamítne normalitu i u nepatrné odchylky od Gaussova tvaru. Na reálných datech tohoto cvičení proto větev s t-testem projde jen zřídka a filtr téměř vždy sáhne po Mann–Whitneyově testu. Je to důsledek téhož vlivu velkého $n$, který o kus dál motivuje zavedení velikosti účinku.

#### 4.3 Shoda rozptylů — Studentův, nebo Welchův t-test

Rozhodneme-li se pro parametrický test, zbývá otázka, zda mají obě skupiny **shodný rozptyl**. Podle odpovědi má t-test dvě varianty.

**Studentův (sdružený) t-test** — předpokládá $\sigma_1^2 = \sigma_2^2$:

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\,\sqrt{\dfrac{1}{n_1} + \dfrac{1}{n_2}}}, \qquad
s_p = \sqrt{\frac{(n_1-1)\,s_1^2 + (n_2-1)\,s_2^2}{n_1 + n_2 - 2}}, \qquad
\nu = n_1 + n_2 - 2$$

**Welchův t-test** — shodu rozptylů nepředpokládá:

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}, \qquad
\nu \approx \frac{\left(\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}\right)^{\!2}}
{\dfrac{(s_1^2/n_1)^2}{n_1-1} + \dfrac{(s_2^2/n_2)^2}{n_2-1}}$$

Druhý vztah je **Welchova–Satterthwaiteova** aproximace stupňů volnosti (obecně necelé číslo). Sdružená směrodatná odchylka $s_p$ ve Studentově testu je **totéž $s_p$**, které vystupuje v Cohenově $d$ v odd. 4.4.

**Test shody rozptylů** má $H_0: \sigma_1^2 = \sigma_2^2$:

- **F-test**: statistika $F = s_1^2 / s_2^2$ se za platnosti $H_0$ řídí **Fisherovým–Snedecorovým $F$-rozdělením** se stupni volnosti $(n_1-1,\ n_2-1)$. $F$-rozdělení je poměr dvou nezávislých $\chi^2$ rozdělení, každého děleného svými stupni volnosti: výběrová veličina $(n-1)s^2/\sigma^2$ se sama řídí $\chi^2_{\,n-1}$, teprve jejich **poměr** je $F$. (Samotné $\chi^2$ rozdělení se uplatní u *jednovýběrového* testu rozptylu proti známé hodnotě $\sigma_0^2$, kde statistika je $(n-1)s^2/\sigma_0^2$.)
- F-test i příbuzný **Bartlettův test** jsou ovšem **velmi citlivé na porušení normality** — na nenormálních datech falešně zamítají. V praxi se proto dává přednost **Leveneho testu** (`scipy.stats.levene`), který je vůči nenormalitě robustní, případně Fligner–Killeenovu.

**Co dělá `scipy.stats.ttest_ind`.** Ve výchozím nastavení (`equal_var=True`) počítá **Studentův sdružený** test; `equal_var=False` přepne na **Welchův**. Welch je dnes doporučovaná výchozí volba — je-li rozptyl skutečně shodný, ztrácí jen zanedbatelně síly, a je-li rozdílný (nebo se výrazně liší velikosti skupin), chrání před nadhodnocenou hladinou významnosti.

#### 4.4 Velikost účinku (effect size)

Klíčové omezení p-hodnoty: **říká, zda rozdíl existuje, ne jak je velký.** Při dostatečně velkém počtu vzorků vyjde jako „významný" i zcela triviální rozdíl — stačí zvětšit $n$ a p-hodnota klesne. **Velikost účinku** měří magnitudu rozdílu nezávisle na $n$. Používáme **Cohenovo $d$**:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}, \qquad s_p = \sqrt{\frac{(n_1-1)\,s_1^2 + (n_2-1)\,s_2^2}{n_1 + n_2 - 2}}$$

kde $s_p$ je táž sdružená (pooled) směrodatná odchylka jako ve Studentově t-testu (odd. 4.3). Hodnota $d$ vyjadřuje rozdíl průměrů **v jednotkách směrodatné odchylky**:

| $\lvert d \rvert$ | Interpretace (Cohen, 1988) |
|:---:|:---|
| $\approx 0{,}2$ | malý účinek |
| $\approx 0{,}5$ | střední účinek |
| $\approx 0{,}8$ a více | velký účinek |

Pomocná metoda `_cohens_d` je **předvyplněna** — studentským úkolem je zapojit obě kritéria, ne odvozovat vzorec.

#### 4.5 Dvojí kritérium

Filtrační selektor ponechá příznak **právě tehdy**, když projde **obě** kritéria zároveň:

$$\bigl(p < \alpha\bigr) \;\wedge\; \bigl(\lvert d \rvert \ge d_{\min}\bigr)$$

kde $d_{\min}$ je `min_effect_size` z konfigurace (výchozí `0.5`). Tato kombinace je pojmovým jádrem filtračního bloku — pojistka proti příznakům, jejichž rozdíl mezi skupinami je statisticky prokazatelný, ale prakticky zanedbatelný. Selektor postavený pouze na p-hodnotě je pro toto cvičení nesprávný.

---

### 5. Obalový (wrapper) výběr příznaků — Template Method

Obalový selektor nehodnotí příznaky izolovaně, ale podle toho, **jak dobře funguje model postavený nad danou podmnožinou**. Sdílená **hladová dopředná** smyčka (`WrapperSelector.select`, předvyplněna) postupně rozšiřuje množinu příznaků a v každém kroku volá skórovací funkci:

```
pro i = 1, 2, …, n_příznaků:
    skóre = self._score(X[:, :i], y)      # prvních i příznaků
    pokud skóre ≥ self.target:
        vrať indexy 0 .. i-1
vrať všechny indexy                        # cíle nebylo dosaženo
```

Smyčka je napsaná **jednou** v bázové třídě; potomci přepisují jedinou metodu `_score`. To je návrhový vzor **Template Method**, stejný jako iterační smyčka `IterativeClustering` v Cvičení 03 nebo `_cluster_distance` v Cvičení 02.

| Potomek | `_score` vrací | Charakter | Knihovna |
|:---|:---|:---|:---|
| `SilhouetteWrapperSelector` | siluetu k-means shlukování na podmnožině | bez učitele | `sklearn.cluster.KMeans`, `sklearn.metrics.silhouette_score` |
| `KNNWrapperSelector` | testovací přesnost kNN klasifikátoru | s učitelem | `sklearn.model_selection.train_test_split`, `sklearn.neighbors.KNeighborsClassifier` |

> **Proč se zde využívá knihovní implementace?** Siluetu jste implementovali v Cvičení 03, k-means rovněž. Předmětem tohoto cvičení je **výběr příznaků**, nikoli reimplementace shlukování — periferní výpočty proto obstará scikit-learn. Jádro (kovariance, výběr komponent, projekce, rekonstrukce, logika filtračního testu) naopak implementujete sami.

Připomenutí vzorce siluety z Cvičení 03 — pro bod $i$ ve shluku s průměrnou vnitroshlukovou vzdáleností $a(i)$ a průměrnou vzdáleností k nejbližšímu jinému shluku $b(i)$:

$$s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i),\, b(i)\bigr)}, \qquad \bar{s} = \frac{1}{n}\sum_{i} s(i)$$

> **Poznámka k této zjednodušené variantě.** Smyčka bere vždy **prvních `i` sloupců** v zadaném pořadí, nikoli „nejlepší dosud nepřidaný příznak" jako plná dopředná selekce. Pro demonstraci vzoru Template Method je to dostačující — v pipeline se navíc obvykle spouští nad příznaky, které již prošly filtrem.

---

### 6. Vizualizace a meze její interpretace

Funkce `plot_feature_vs_pca_space` vykreslí vedle sebe dva bodové grafy obarvené diagnózou: vlevo dva původní příznaky, vpravo první dvě hlavní komponenty. V PC prostoru bývají maligní a benigní skupiny často **lépe oddělené**, což by se dalo mylně považovat za výsledek, o který PCA usiluje.

> **PCA je metoda bez učitele.** Maximalizuje **rozptyl dat**, nikoli **oddělení tříd** — o existenci `y` vůbec neví. Lepší separace v PC prostoru je **častým vedlejším efektem** (mezitřídní rozptyl bývá velký, a propisuje se proto do prvních komponent), **nikoli cílem ani zárukou**. V některých případech k lepšímu oddělení tříd nedojde, což je rovněž informativní výsledek. Metodou, která oddělení tříd optimalizuje přímo, je **LDA (lineární diskriminační analýza)** — hledá projekci maximalizující poměr mezitřídního a vnitrotřídního rozptylu, a k tomu popisky `y` potřebuje. PCA je nezná.

---

### 7. Rozdělení implementace mezi studenta a knihovny

| Komponenta | Kdo počítá | Poznámka |
|:---|:---|:---|
| Kovarianční matice, vlastní rozklad seřazení | **student** (`eigh` je povolené volání) | jádro PCA |
| Výběr počtu komponent, projekce, rekonstrukce | **student** | jádro PCA |
| Uložení / načtení modelu (`.npz`) | **student** | téma „model = parametry" |
| Volba testu, p-hodnota, dvojí kritérium filtru | **student** | jádro filtračního výběru |
| Shapiro–Wilk, t-test, Mann–Whitney U | `scipy.stats` | volané uvnitř `select` |
| Cohenovo $d$ (`_cohens_d`) | **předvyplněno** | student jen zapojuje |
| k-means, silueta, kNN, train/test split | `scikit-learn` | periferie wrapperů |
| Načtení dat, standardizace, konfigurace, grafy | **předvyplněno** (`dataio/`) | není předmětem cvičení |

---

## Konfigurace projektu

### Soubor `config.yaml`

Hyperparametry experimentu jsou v kořenovém `config.yaml`:

```yaml
data:
  standardize: true
  random_state: 42

feature_selection:
  alpha: 0.05                 # hladina významnosti pro filtrační test
  min_effect_size: 0.5        # minimální |Cohenovo d|, aby filtr příznak ponechal
  silhouette_target: 0.67     # zastavovací kritérium wrapperu (silueta)
  knn_accuracy_target: 0.93   # zastavovací kritérium wrapperu (přesnost kNN)
  knn_neighbors: 3
  test_size: 0.2

pca:
  variance_threshold: 95.0    # % kumulativní vysvětlené variance k zachování
  n_components: null          # alternativa: pevný počet komponent (null = použij práh)

clustering:
  n_clusters: 2               # navazující porovnání s k-means
```

Změna prahu variance, cílů wrapperů nebo hladiny významnosti je pouhá úprava konfigurace, ne kódu. Parametr `n_components` ukazuje běžný vzor „buď–anebo": je-li `null`, počet komponent řídí `variance_threshold`; je-li to celé číslo, použije se přímo.

### Typovaná konfigurace (dataclassy)

Konfigurace se načítá funkcí `load_config()` a vrací jako instance dataclassy `ExperimentConfig`. Přístup k hodnotám je přes **atributy**, ne slovníkové klíče:

```
# Místo:   cfg["pca"]["variance_threshold"]   ← runtime chyba při překlepu
# Správně: cfg.pca.variance_threshold          ← editor odhalí překlep okamžitě
```

Struktura dataclassů zrcadlí sekce YAML:

```
ExperimentConfig
    ├── data: DataConfig
    │       ├── standardize: bool
    │       └── random_state: int
    ├── feature_selection: FeatureSelectionConfig
    │       ├── alpha: float
    │       ├── min_effect_size: float
    │       ├── silhouette_target: float
    │       ├── knn_accuracy_target: float
    │       ├── knn_neighbors: int
    │       └── test_size: float
    ├── pca: PCAConfig
    │       ├── variance_threshold: float
    │       └── n_components: int | None
    └── clustering: ClusteringConfig
            └── n_clusters: int
```

`load_config()` volá `validate_config()`, která ověří rozsahy hodnot (`0 < alpha < 1`, `0 < variance_threshold ≤ 100`, `n_clusters ≥ 2`, …) a při neplatné konfiguraci vyvolá srozumitelný `ValueError`. Obě funkce jsou **předvyplněné** — není potřeba do nich zasahovat.

---

## Pokyny k vypracování

Otevřete `src/pca.py` a `src/feature_selection.py` a nahraďte všechny výskyty `raise NotImplementedError("Úkol: …")` funkčním kódem. Bloky implementujte v uvedeném pořadí — pipeline i testy na něm závisí.

Komentáře ve tvaru `# assert  Ověřte, že …` jsou **nápovědy pro validaci vstupů**. Napište odpovídající příkazy `assert` na daná místa — chrání vás před záhadnými chybami při špatném vstupu. (Nikdy nenechávejte aktivní `assert` uvnitř nedokončené kostry.)

> **Předpoklad z předchozích cvičení: žádný.** Na rozdíl od Cvičení 02–04 se sem nekopíruje `src/distance.py` ani nic jiného. Vše potřebné je v repozitáři.

---

### Blok I: PCA — `src/pca.py`

Třída `PCA` je samostatná (nedědí z ničeho). **Předvyplněny** jsou `__init__`, `fit` a `fit_transform` — smyčku `fit` volá vaše dvě metody `_covariance_matrix` a `_select_components`, takže dokud nejsou hotové, `fit` korektně vyvolá jejich `Úkol:`.

> **Časté chyby v tomto bloku:**
> - **Centrování jiným průměrem.** `transform` musí odečítat `self.mean_` naučený ve `fit()`, ne průměr právě předaných dat.
> - **`np.linalg.eig` místo `eigh`.** `eig` u symetrické matice nezaručuje reálný výsledek ani pořadí; použijte `eigh`.
> - **Seřazení jen vlastních čísel.** Permutaci podle sestupných vlastních čísel je nutné aplikovat toutéž `argsort` i na **sloupce** matice vlastních vektorů.
> - **Zapomenuté obrácení.** `eigh` vrací vzestupně — bez otočení dostanete komponenty s **nejmenším** rozptylem.
> - **Záměna tvaru `components_`.** Komponenty jsou **sloupce** matice `(n, n_components_)`; projekce je pak `X_c @ components_`.
> - **Off-by-one v `_select_components`.** Vrací se **počet** komponent, ne index prahové komponenty (obvykle `index + 1`).

#### `_covariance_matrix(X)`

```
# Vstup:  X tvaru (m, n) — m vzorků, n příznaků
# Výstup: kovarianční matice tvaru (n, n)
#
# 1. Vycentrujte sloupce:  Xc = X - průměr každého sloupce
# 2. Vraťte  (Xc.T @ Xc) / (m - 1)          ← jeden maticový součin, žádná smyčka
#
# Výsledek se musí shodovat s numpy.cov(X, rowvar=False).
```

#### `_select_components(eigenvalues)`

```
# Vstup:  eigenvalues — vlastní čísla JIŽ SEŘAZENÁ SESTUPNĚ
# Výstup: int — nejmenší počet komponent, jehož kumulativní vysvětlená
#         variance dosáhne self.variance_threshold procent
#
# 1. ratio = eigenvalues / eigenvalues.sum()          (podíl na komponentu)
# 2. cumulative = kumulativní součet(ratio) * 100
# 3. Vraťte index prvního prvku cumulative >= self.variance_threshold, plus 1
#
# Připomeňte si: PCA bere NEJVĚTŠÍ vlastní čísla (nejvíce rozptylu) —
# opak spektrálního shlukování v Cvičení 04.
```

#### `transform(X)`

```
# Ověřte (assert), že model je natrénován (self.components_ není None)
# a že X.shape[1] odpovídá self.mean_.shape[0].
#
# Vraťte  (X - self.mean_) @ self.components_        → tvar (m, n_components_)
#
# Znaménko jednotlivých sloupců výsledku se může lišit od scikit-learn — to je v pořádku.
```

#### `inverse_transform(X_pca)`

```
# Ověřte (assert), že model je natrénován a X_pca.shape[1] == self.n_components_.
#
# Vraťte  X_pca @ self.components_.T + self.mean_    → tvar (m, n)
#
# Při zachování všech komponent platí inverse_transform(transform(X)) ≈ X.
```

#### `save(path)`

```
# Ověřte (assert), že model je natrénován.
#
# Uložte naučená pole do souboru .npz (cesta `path`) pomocí numpy.savez:
#   - povinně: components_, mean_
#   - dále tolik, aby šel plně obnovit stav modelu:
#     explained_variance_ratio_, eigenvalues_, n_components_
#
# Pipeline volá save() s cestou "models/pca_model.npz".
# NIKDY nepoužívejte pickle (neprůhledný, nebezpečný).
```

#### `load(path)` — `@classmethod`

```
# Přečtěte .npz pomocí numpy.load.
# Vytvořte novou instanci: obj = cls()
# Nastavte jí naučené atributy z načtených polí (n_components_ přetypujte na int).
# Vraťte obj — připraven k transform() bez opětovného fit().
#
# Toto funguje jen proto, že PCA je induktivní (viz teorie, odd. 3).
```

---

### Blok II: Filtrační výběr — `src/feature_selection.py`

**Předvyplněny** jsou ABC `FeatureSelector` (abstraktní `select`) a pomocná metoda `FilterSelector._cohens_d`.

#### `FilterSelector.select(X, y)`

```
# Ověřte (assert): X je 2D, len(y) == X.shape[0], y je binární ({0, 1}).
#
# Pro každý příznak j = 0 .. n-1:
#   a = X[y == 0, j]      # hodnoty v benigní skupině
#   b = X[y == 1, j]      # hodnoty v maligní skupině
#
#   1. Test normality obou skupin: scipy.stats.shapiro(a), shapiro(b)
#   2. Volba testu:
#        obě p_shapiro > self.alpha  → p = scipy.stats.ttest_ind(a, b).pvalue
#        jinak                        → p = scipy.stats.mannwhitneyu(a, b).pvalue
#      (ttest_ind je ve výchozím stavu Studentův sdružený test;
#       equal_var=False by dal Welchův — viz teorie, odd. 4.3)
#   3. Velikost účinku: d = self._cohens_d(a, b)
#   4. Ponechej j PRÁVĚ KDYŽ:  p < self.alpha  AND  abs(d) >= self.min_effect_size
#
# Vraťte np.ndarray celočíselných indexů ponechaných příznaků
# (prázdné pole np.array([], dtype=int), pokud neprojde žádný).
```

Pojmové těžiště bloku: **obě** kritéria současně. Selektor postavený pouze na `p < alpha` je nesprávný — neochrání před příznakem, který je významný, ale odděluje skupiny jen zanedbatelně (viz teorie, odd. 4.4–4.5).

> **Zjednodušení.** Referenční řešení test shody rozptylů (odd. 4.3) vynechává a volá `ttest_ind` ve výchozím nastavení — Blok II se soustředí na propojení dvou kritérií, ne na úplný rozhodovací strom. Přidat `scipy.stats.levene` a podle jeho výsledku přepínat `equal_var` je vítané rozšíření.

---

### Blok III: Obalový výběr — `src/feature_selection.py`

**Předvyplněna** je bázová třída `WrapperSelector` s hladovou smyčkou `select` a abstraktní `_score`. Implementujete pouze skórovací metody obou potomků.

#### `SilhouetteWrapperSelector._score(X_subset, y)`

```
# Ověřte (assert): X_subset je 2D; X_subset.shape[0] > self.n_clusters.
#
# 1. Nafitujte KMeans(n_clusters=self.n_clusters,
#                      random_state=self.random_state, n_init=10) na X_subset.
# 2. Vraťte silhouette_score(X_subset, labels).
#
# y se zde NEPOUŽÍVÁ (kritérium je bez učitele) — je v podpisu jen kvůli
# společnému rozhraní.
```

#### `KNNWrapperSelector._score(X_subset, y)`

```
# Ověřte (assert): X_subset je 2D; len(y) == X_subset.shape[0]; y má aspoň 2 třídy.
#
# 1. X_tr, X_te, y_tr, y_te = train_test_split(
#        X_subset, y, test_size=self.test_size,
#        random_state=self.random_state, stratify=y)
# 2. Nafitujte KNeighborsClassifier(n_neighbors=self.n_neighbors) na X_tr, y_tr.
# 3. Vraťte testovací přesnost clf.score(X_te, y_te).
```

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_05.py -v
```

| Třída testů | Co ověřuje |
|:---|:---|
| `TestPCA` | `_covariance_matrix` se shoduje s `numpy.cov`; `explained_variance_ratio_` se shoduje se `sklearn.decomposition.PCA`; `_select_components` je monotónní v prahu; `inverse_transform(transform(X)) ≈ X` při všech komponentách; `transform` má správný tvar |
| `TestPCAPersistence` | `save` → `load` round-trip: obnovený model dává stejné `transform(X)` a stejná pole `components_`, `mean_`, `n_components_` (přes `tmp_path` fixturu) |
| `TestFeatureSelection` | `FilterSelector` vybere oddělující příznaky, zahodí šumové **a zahodí příznak statisticky významný, ale s nepatrnou velikostí účinku** (test cílí na dvojí kritérium); `_cohens_d` funguje i ve stavu kostry; oba wrappery dosáhnou cíle na separovatelných datech a vrátí souvislý prefix indexů |

Testy se **nespoléhají na syrové vlastní vektory** — porovnávají podíly vysvětlené variance, chybu rekonstrukce a znaménkově necitlivé veličiny (nejednoznačnost znaménka, viz teorie, odd. 2.2). Používají malá syntetická data, kde správná implementace musí projít bez ohledu na numerické detaily.

Dokud nejsou příslušné metody hotové, testy, které je volají, se hlásí jako **`xfail`** (očekávané selhání na `NotImplementedError`) a celá sada skončí s návratovým kódem 0 — to je záměrné, ne chyba. Jakmile metodu doplníte, stejný test začne procházet (`xpass` → `pass`).

Průběžně ověřujte i celou pipeline:

```bash
python cviceni_05.py
```

Kroky s neimplementovanými metodami se přeskočí s hláškou `[NENI HOTOVO] Úkol: …`; ostatní proběhnou normálně, uloží grafy do `graphs/` a natrénovaný model do `models/pca_model.npz`.

---

## Doplňkové (papírové) příklady

Soubor `priklady_05.md` obsahuje pět číselných příkladů na papír:

1. **Kovarianční matice ručně** — průměry, centrování, matice $2 \times 2$.
2. **Vlastní čísla matice $2 \times 2$** — charakteristická rovnice a % vysvětlené variance.
3. **Výběr počtu komponent** — kumulativní variance a práh.
4. **Projekce bodu** — souřadnice v nové bázi a ztrátová rekonstrukce.
5. **Filtrační výběr (pojmově)** — volba testu, čtení p-hodnoty vůči `alpha`, role velikosti účinku.

Příklady 1–4 odpovídají krokům, které programujete v `src/pca.py` (kovarianční matice, vlastní čísla, výběr počtu komponent, projekce); příklad 5 je pojmový k filtračnímu testu z Bloku II. Slouží k procvičení výpočtů z teoretického základu bez psaní kódu. **Řešení nejsou součástí repozitáře** — zkontrolujte si výsledky s vyučujícím nebo dopočtem v `numpy` / `scipy`.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení implementace proveďte:

```bash
git add src/pca.py src/feature_selection.py
git commit -m "Implementace cvičení 5"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).

> **Soubory, které se neodevzdávají:** `src/__init__.py`, `dataio/__init__.py`, `dataio/loader.py`, `dataio/config_manager.py`, `dataio/plotting.py`, `cviceni_05.py` a `test_cviceni_05.py` jsou předvyplněny nebo se nemají měnit. Systém tyto soubory ignoruje a hodnotí pouze `src/pca.py` a `src/feature_selection.py`.
