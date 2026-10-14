"""Análisis bivariado: cada variable comparada contra income."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codigo.configuracion import CARPETA_GRAFICOS, CLASES, COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, OBJETIVO
from codigo.eda.estilo import COLOR_INCOME, barras_100, guardar, poner_titulo

CARPETA = CARPETA_GRAFICOS / "03_contra_income"

# Numéricas continuas -> curvas de densidad por clase
CONTINUAS = ["age", "fnlwgt", "hours.per.week"]
# Capital -> la agrupamos en rangos
RANGOS_CAPITAL = [-1, 0, 2000, 5000, 10000, 50000, np.inf]
NOMBRES_RANGOS = ["0", "1-2k", "2k-5k", "5k-10k", "10k-50k", "50k+"]
MOSTRAR_TOP = {"native.country": 10}


def tasa_general(df):
    """% de personas que ganan >50K en todo el dataset (la línea punteada de referencia)."""
    return (df[OBJETIVO] == ">50K").mean() * 100


def grafico_densidad(df, columna):
    """Histogramas de densidad superpuestos: cada clase se normaliza por separado,
    así se comparan las FORMAS aunque una clase tenga 3 veces más filas que la otra."""
    if columna == "fnlwgt":
        bins = 50
    else:
        # age y hours.per.week son enteros: bins de 2 unidades alineados a los enteros
        bins = np.arange(df[columna].min(), df[columna].max() + 3, 2) - 0.5
    medianas = df.groupby(OBJETIVO)[columna].median()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for clase in CLASES:
        valores = df.loc[df[OBJETIVO] == clase, columna]
        color = COLOR_INCOME[clase]
        ax.hist(valores, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=color)
        ax.hist(valores, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"{clase} (median {medianas[clase]:,.0f})")
        ax.axvline(medianas[clase], color=color, lw=1.2, ls="--")

    ax.set_xlabel(columna)
    ax.set_ylabel("Density")
    ax.grid(axis="x", visible=False)
    ax.legend(title=OBJETIVO)
    diferencia = medianas[">50K"] - medianas["<=50K"]
    poner_titulo(ax, f"{columna} by {OBJETIVO}", f"Density per class  ·  median gap: {diferencia:,.0f}")
    return guardar(fig, CARPETA / f"num_{columna}.png")


def grafico_proporciones(df, grupos, columna, nombre_archivo, subtitulo, ordenar_por_tasa=True):
    """Barras al 100%: para cada grupo, qué % gana >50K y qué % <=50K."""
    porcentajes = pd.crosstab(grupos, df[OBJETIVO], normalize="index")[CLASES] * 100
    cantidades = grupos.value_counts()

    if ordenar_por_tasa:
        porcentajes = porcentajes.sort_values(">50K")
    else:
        porcentajes = porcentajes.iloc[::-1]  # orden natural, el primero arriba

    fig, ax = plt.subplots(figsize=(10, max(2.8, 0.42 * len(porcentajes) + 1.3)))
    barras_100(ax, porcentajes, linea_referencia=tasa_general(df))
    etiquetas = [f"{g}  (n={cantidades[g]:,})" for g in porcentajes.index]
    ax.set_yticks(range(len(porcentajes)), etiquetas)
    ax.set_ylabel(columna)
    poner_titulo(ax, f"Share of {OBJETIVO} by {columna}", subtitulo)
    return guardar(fig, CARPETA / nombre_archivo)


def grafico_capital(df, columna):
    rangos = pd.cut(df[columna], bins=RANGOS_CAPITAL, labels=NOMBRES_RANGOS)
    rangos = rangos.cat.remove_unused_categories()
    return grafico_proporciones(df, rangos, columna, f"num_{columna}.png",
                                "Grouped in ranges  ·  dashed line = overall >50K rate", ordenar_por_tasa=False)


def grafico_education_num(df, columna):
    return grafico_proporciones(df, df[columna], columna, f"num_{columna}.png",
                                "One bar per value  ·  dashed line = overall >50K rate", ordenar_por_tasa=False)


def grafico_categorica(df, columna):
    grupos = df[columna].fillna("NaN ('?')")

    top = MOSTRAR_TOP.get(columna)
    if top:
        mas_frecuentes = grupos[grupos != "NaN ('?')"].value_counts().index[:top]
        quedan = grupos.isin(mas_frecuentes) | (grupos == "NaN ('?')")
        grupos = grupos.where(quedan, "Other")

    return grafico_proporciones(df, grupos, columna, f"cat_{columna}.png",
                                "Sorted by >50K share  ·  dashed line = overall >50K rate")


def ejecutar(df):
    rutas = []
    for columna in COLUMNAS_NUMERICAS:
        if columna in CONTINUAS:
            rutas.append(grafico_densidad(df, columna))
        elif columna.startswith("capital"):
            rutas.append(grafico_capital(df, columna))
        else:
            rutas.append(grafico_education_num(df, columna))
    for columna in COLUMNAS_CATEGORICAS:
        rutas.append(grafico_categorica(df, columna))
    return rutas
