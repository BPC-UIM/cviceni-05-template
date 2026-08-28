# Papírové příklady 05 — Výběr příznaků a PCA

Tyto příklady si spočítejte **tužkou na papír**. Slouží k procvičení mechaniky,
kterou pak v kódu implementujete. Čísla jsou volena tak, aby vycházela „hezky“.
Řešení nejsou součástí repozitáře — zkontrolujte si je s vyučujícím nebo dopočtem
v numpy.

Značení: dataset `X` má vzorky v řádcích a příznaky ve sloupcích.

---

## Příklad 1 — Kovarianční matice ručně

Dán dataset se 3 vzorky a 2 příznaky:

```
X = | 2   1 |
    | 4   3 |
    | 6   8 |
```

Spočítejte:

1. průměr každého sloupce,
2. vycentrovaná data `Xc` (od každého sloupce odečtěte jeho průměr),
3. kovarianční matici `2 × 2` s dělitelem `m − 1` (tj. `C = (Xcᵀ Xc) / (m − 1)`).

Ověřte, že vám vyjde symetrická matice.

---

## Příklad 2 — Vlastní čísla matice 2 × 2

Dána kovarianční matice

```
C = | 4   2 |
    | 2   7 |
```

1. Napište charakteristickou rovnici `det(C − λI) = 0` a vyřešte ji
   (kvadratická rovnice — pro `2 × 2` je ruční výpočet povolen).
2. Seřaďte vlastní čísla sestupně.
3. Určete, kolik procent celkového rozptylu vysvětluje každá hlavní komponenta
   (podíl `λᵢ / Σλ`).

---

## Příklad 3 — Výběr počtu komponent

PCA vrátila následující vlastní čísla (už seřazená sestupně):

```
λ = [ 6.0,  3.0,  2.0,  0.8,  0.2 ]
```

1. Spočítejte podíl vysvětleného rozptylu (v %) pro každou komponentu.
2. Spočítejte **kumulativní** vysvětlený rozptyl (v %) po jednotlivých komponentách.
3. Kolik komponent je potřeba zachovat, aby kumulativní vysvětlený rozptyl
   dosáhl alespoň **90 %**? A kolik pro práh **95 %**?

---

## Příklad 4 — Projekce bodu do prostoru hlavních komponent

Máte dva ortonormální vlastní vektory (zapsané jako sloupce):

```
v1 = ( 0.6,  0.8 )      v2 = ( -0.8,  0.6 )
```

a jeden vycentrovaný datový bod `x = ( 2,  1 )`.

1. Ověřte, že `v1` a `v2` jsou ortonormální (jednotková délka, kolmé).
2. Spočítejte souřadnice bodu `x` v nové bázi, tj. `[ v1ᵀx , v2ᵀx ]`
   (maticově `Vᵀ x`, kde `V = [v1  v2]`).
3. Kdybyste ponechali jen první komponentu, jaká by byla rekonstrukce bodu
   zpět v původních souřadnicích (`souřadnice₁ · v1`)? O kolik se liší od `x`?

---

## Příklad 5 — Filtrační výběr příznaku (pojmově)

Hodnoty jednoho příznaku ve dvou diagnostických skupinách:

```
skupina A (y = 0):  5.1  4.9  5.0  5.2  4.8
skupina B (y = 1):  6.0  6.2  5.9  6.1  6.3
```

K dispozici máte i výsledek testu normality (Shapiro–Wilk) — obě skupiny mají
`p ≈ 0.97`. Významová hladina je `alpha = 0.05`.

Odpovězte slovně (bez ručního počítání p-hodnoty testu):

1. Který dvouvýběrový test je pro tento příznak vhodný — parametrický
   (`ttest_ind`), nebo neparametrický (`mannwhitneyu`)? Zdůvodněte přes
   předpoklad normality.
2. Test vrátí `p = 4.1 · 10⁻⁶`. Jak to interpretujete vůči `alpha`? Ponechá
   filtr tento příznak podle **prvního** kritéria (významnost)?
3. Představte si jiný příznak, u kterého vyjde `p = 0.002`, ale rozdíl průměrů
   obou skupin je jen `0.03` směrodatné odchylky (`|Cohenovo d| ≈ 0.03`).
   Co v tomto případě řekne **druhé** kritérium (velikost účinku) a proč je
   dobře, že filtr používá obě kritéria najednou?
