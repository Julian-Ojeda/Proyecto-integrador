"""Análisis univariado: cómo se distribuye cada variable por separado."""
import matplotlib.pyplot as plt
import numpy as np

from codigo.configuracion import CARPETA_GRAFICOS, CLASES, COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, OBJETIVO
from codigo.eda.estilo import AZUL, COLOR_INCOME, FONDO, GRIS, TEXTO, guardar, poner_titulo

CARPETA = CARPETA_GRAFICOS / "02_univariado"

# Variables que son casi todo ceros: un histograma común sería una sola barra en 0
CASI_TODO_CEROS = ["capital.gain", "capital.loss"]
# Variables enteras con pocos valores: un bin por cada valor entero
ENTERAS = ["age", "education.num", "hours.per.week"]
# Para native.country mostramos solo los 10 países más frecuentes
MOSTRAR_TOP = {"native.country": 10}


def resumen_numerico(serie):
    return (f"n={serie.count():,}  ·  mean={serie.mean():,.1f}  ·  median={serie.median():,.1f}"
            f"  ·  std={serie.std():,.1f}  ·  skew={serie.skew():.2f}")


def contar_outliers(serie):
    """Regla del rango intercuartílico (IQR): fuera de [Q1 - 1.5·IQR, Q3 + 1.5·IQR]."""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return int(((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).sum())


def grafico_numerica(df, columna):
    """Histograma arriba y boxplot abajo, compartiendo el eje X."""
    serie = df[columna].dropna()
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05}
    )

    if columna in ENTERAS:
        bins = np.arange(serie.min(), serie.max() + 2) - 0.5  # un bin por cada entero
    else:
        bins = 60
    ax_hist.hist(serie, bins=bins, color=AZUL, edgecolor=FONDO, linewidth=0.5)
    ax_hist.axvline(serie.mean(), color=TEXTO, lw=1.2, ls="--", label="mean")
    ax_hist.axvline(serie.median(), color=TEXTO, lw=1.2, ls=":", label="median")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(axis="x", visible=False)
    ax_hist.legend(loc="upper right")

    ax_box.boxplot(serie, vert=False, widths=0.6, patch_artist=True,
                   boxprops={"facecolor": "#d6e5f7", "edgecolor": AZUL},
                   medianprops={"color": TEXTO},
                   whiskerprops={"color": AZUL},
                   capprops={"color": AZUL},
                   flierprops={"marker": "o", "markersize": 2, "markerfacecolor": GRIS, "markeredgecolor": "none"})
    ax_box.set_yticks([])
    ax_box.grid(axis="y", visible=False)
    ax_box.set_xlabel(columna)

    poner_titulo(ax_hist, f"Distribution of {columna}",
                 resumen_numerico(serie) + f"  ·  IQR outliers={contar_outliers(serie):,}")
    return guardar(fig, CARPETA / f"num_{columna}.png")


def grafico_casi_todo_ceros(df, columna):
    """Izquierda: cuántos son 0 y cuántos no. Derecha: histograma de los que no son 0, en escala log."""
    serie = df[columna]
    no_cero = serie[serie > 0]
    ceros = len(serie) - len(no_cero)

    fig, (ax_ceros, ax_hist) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 3]})

    ax_ceros.bar(["0", "> 0"], [ceros, len(no_cero)], color=[GRIS, AZUL], width=0.6)
    ax_ceros.text(0, ceros, f"{ceros / len(serie) * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax_ceros.text(1, len(no_cero), f"{len(no_cero) / len(serie) * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax_ceros.set_ylabel("Count")
    ax_ceros.grid(axis="x", visible=False)
    poner_titulo(ax_ceros, "Zero vs non-zero")

    # Bins con escala logarítmica; el máximo se agranda un poco para que entre el último valor
    bins = np.logspace(np.log10(no_cero.min()), np.log10(no_cero.max() * 1.05), 40)
    ax_hist.hist(no_cero, bins=bins, color=AZUL, edgecolor=FONDO, linewidth=0.5)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(f"{columna} (log scale, non-zero only)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(axis="x", visible=False)

    filas_en_maximo = (serie == serie.max()).sum()
    poner_titulo(ax_hist, f"Distribution of {columna} (non-zero)",
                 f"{ceros / len(serie) * 100:.1f}% of rows are 0  ·  non-zero median={no_cero.median():,.0f}"
                 f"  ·  max={serie.max():,} ({filas_en_maximo} rows)")
    return guardar(fig, CARPETA / f"num_{columna}.png")


def grafico_categorica(df, columna):
    """Barras horizontales ordenadas por frecuencia. Los nulos se muestran en gris."""
    conteo = df[columna].fillna("NaN ('?')").value_counts()

    # Si hay demasiadas categorías, dejamos las más frecuentes y el resto va a "Other"
    top = MOSTRAR_TOP.get(columna)
    if top:
        nulos = conteo.pop("NaN ('?')") if "NaN ('?')" in conteo else 0
        cantidad_otras = len(conteo) - top
        otras = conteo.iloc[top:].sum()
        conteo = conteo.iloc[:top]
        conteo[f"Other ({cantidad_otras} categories)"] = otras
        if nulos:
            conteo["NaN ('?')"] = nulos

    conteo = conteo.iloc[::-1]  # la más frecuente arriba
    porcentaje = conteo / len(df) * 100
    colores = [GRIS if nombre.startswith(("NaN", "Other")) else AZUL for nombre in conteo.index]

    fig, ax = plt.subplots(figsize=(9, max(2.6, 0.38 * len(conteo) + 1.2)))
    ax.barh(conteo.index, conteo.values, color=colores, height=0.68)
    for y, nombre in enumerate(conteo.index):
        ax.text(conteo[nombre], y, f"  {conteo[nombre]:,} ({porcentaje[nombre]:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, conteo.max() * 1.25)
    ax.set_xlabel("Count")
    ax.grid(axis="y", visible=False)

    poner_titulo(ax, f"Distribution of {columna}",
                 f"{df[columna].nunique()} categories  ·  missing={df[columna].isna().sum():,}"
                 "  ·  grey = missing / grouped")
    return guardar(fig, CARPETA / f"cat_{columna}.png")


def grafico_objetivo(df):
    conteo = df[OBJETIVO].value_counts()[CLASES]
    porcentaje = conteo / len(df) * 100

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(conteo.index, conteo.values, color=[COLOR_INCOME[c] for c in CLASES], width=0.55)
    for x, clase in enumerate(CLASES):
        ax.text(x, conteo[clase], f"{conteo[clase]:,}\n{porcentaje[clase]:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, conteo.max() * 1.2)
    ax.set_ylabel("Count")
    ax.set_xlabel(OBJETIVO)
    ax.grid(axis="x", visible=False)

    proporcion = conteo["<=50K"] / conteo[">50K"]
    poner_titulo(ax, f"Target distribution ({OBJETIVO})", f"Imbalanced: ratio {proporcion:.1f} : 1")
    return guardar(fig, CARPETA / f"target_{OBJETIVO}.png")


def ejecutar(df):
    rutas = [grafico_objetivo(df)]
    for columna in COLUMNAS_NUMERICAS:
        if columna in CASI_TODO_CEROS:
            rutas.append(grafico_casi_todo_ceros(df, columna))
        else:
            rutas.append(grafico_numerica(df, columna))
    for columna in COLUMNAS_CATEGORICAS:
        rutas.append(grafico_categorica(df, columna))
    return rutas
