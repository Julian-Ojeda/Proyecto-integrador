"""Relación entre variables.

- Numéricas: correlación de Pearson (de -1 a 1).
- Categóricas: V de Cramér (de 0 a 1). Mide qué tan asociadas están dos categóricas,
  basándose en la tabla de contingencia y el test chi-cuadrado.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codigo.configuracion import CARPETA_GRAFICOS, COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS, OBJETIVO
from codigo.eda.estilo import ESCALA_AZUL, ESCALA_DIVERGENTE, guardar, poner_titulo

CARPETA = CARPETA_GRAFICOS / "04_correlaciones"


def v_de_cramer(x, y):
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    tabla = pd.crosstab(x, y).to_numpy()
    n = tabla.sum()

    # Chi-cuadrado: compara lo observado con lo esperado si fueran independientes
    esperado = np.outer(tabla.sum(axis=1), tabla.sum(axis=0)) / n
    chi2 = ((tabla - esperado) ** 2 / esperado).sum()

    filas, cols = tabla.shape
    phi2 = max(0, chi2 / n - (cols - 1) * (filas - 1) / (n - 1))
    filas_corr = filas - (filas - 1) ** 2 / (n - 1)
    cols_corr = cols - (cols - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / min(filas_corr - 1, cols_corr - 1))


def mapa_de_calor(ax, matriz, escala, minimo, maximo):
    """Dibuja solo el triángulo inferior (el superior es igual, espejado)."""
    triangulo_superior = np.triu(np.ones(matriz.shape, dtype=bool), k=1)
    imagen = ax.imshow(np.ma.array(matriz.to_numpy(), mask=triangulo_superior), cmap=escala, vmin=minimo, vmax=maximo)

    for i in range(len(matriz)):
        for j in range(i + 1):
            valor = matriz.iat[i, j]
            # Texto blanco sobre celdas oscuras, negro sobre celdas claras
            r, g, b, _ = escala((valor - minimo) / (maximo - minimo))
            color_texto = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "#0b0b0b"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8, color=color_texto)

    ax.set_xticks(range(len(matriz)), matriz.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(matriz)), matriz.index)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    return imagen


def grafico_pearson(df):
    numericas = df[COLUMNAS_NUMERICAS].copy()
    numericas[f"{OBJETIVO} (>50K=1)"] = (df[OBJETIVO] == ">50K").astype(int)  # income como 0/1 de referencia
    matriz = numericas.corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    imagen = mapa_de_calor(ax, matriz, ESCALA_DIVERGENTE, -1, 1)
    fig.colorbar(imagen, ax=ax, shrink=0.7, label="Pearson r")
    poner_titulo(ax, "Pearson correlation (numeric)", f"{OBJETIVO} encoded as 0/1 for reference")
    return guardar(fig, CARPETA / "pearson_numeric.png")


def grafico_cramer(df):
    columnas = COLUMNAS_CATEGORICAS + [OBJETIVO]
    datos = df[columnas].fillna("NaN")

    matriz = pd.DataFrame(1.0, index=columnas, columns=columnas)
    for a in columnas:
        for b in columnas:
            if a != b:
                matriz.loc[a, b] = v_de_cramer(datos[a], datos[b])

    fig, ax = plt.subplots(figsize=(8, 6.8))
    imagen = mapa_de_calor(ax, matriz, ESCALA_AZUL, 0, 1)
    fig.colorbar(imagen, ax=ax, shrink=0.7, label="Cramér's V")
    poner_titulo(ax, "Cramér's V (categorical)", "0 = no association, 1 = one variable determines the other")
    return guardar(fig, CARPETA / "cramers_v_categorical.png")


def ejecutar(df):
    return [grafico_pearson(df), grafico_cramer(df)]
