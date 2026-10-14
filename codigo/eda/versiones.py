"""Comparación de las tres versiones limpias contra el dataset original."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from codigo.configuracion import CARPETA_GRAFICOS, CLASES, OBJETIVO
from codigo.eda.estilo import AZUL, COLOR_VERSION, GRIS, barras_100, guardar, poner_titulo

CARPETA = CARPETA_GRAFICOS / "05_versiones"


def grafico_filas(original, versiones):
    """Cuántas filas conserva cada estrategia."""
    todas = {"original": original, **versiones}
    nombres = list(todas)[::-1]  # la original arriba
    filas = [len(todas[n]) for n in nombres]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(nombres, filas, color=[GRIS if n == "original" else AZUL for n in nombres], height=0.6)
    for y, cantidad in enumerate(filas):
        perdidas = len(original) - cantidad
        texto = f"  {cantidad:,}"
        if perdidas:
            texto += f"  (-{perdidas:,}, -{perdidas / len(original) * 100:.1f}%)"
        ax.text(cantidad, y, texto, va="center", fontsize=9)

    ax.set_xlim(0, len(original) * 1.35)
    ax.set_xlabel("Rows")
    ax.grid(axis="y", visible=False)
    poner_titulo(ax, "Rows kept by each cleaning strategy")
    return guardar(fig, CARPETA / "rows_per_version.png")


def grafico_income(original, versiones):
    """¿La limpieza cambia el balance de income?"""
    todas = {"original": original, **versiones}
    porcentajes = pd.DataFrame({
        nombre: datos[OBJETIVO].value_counts(normalize=True)[CLASES] * 100
        for nombre, datos in todas.items()
    }).T.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    barras_100(ax, porcentajes, linea_referencia=(original[OBJETIVO] == ">50K").mean() * 100)
    ax.set_yticks(range(len(porcentajes)), porcentajes.index)
    poner_titulo(ax, f"{OBJETIVO} share by version", "Does the cleaning strategy change the target balance?")
    return guardar(fig, CARPETA / "income_by_version.png")


def grafico_mezcla_categorias(original, versiones, columna):
    """% de cada categoría en cada versión, barras agrupadas."""
    todas = {"original": original.fillna({columna: "NaN ('?')"}), **versiones}
    porcentajes = pd.DataFrame({
        nombre: datos[columna].value_counts(normalize=True) * 100
        for nombre, datos in todas.items()
        if columna in datos
    }).fillna(0).sort_values("original")

    cantidad_versiones = porcentajes.shape[1]
    alto = 0.8 / cantidad_versiones
    posiciones = np.arange(len(porcentajes))

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(porcentajes) + 1.6))
    for i, nombre in enumerate(porcentajes.columns):
        desplazamiento = (i - (cantidad_versiones - 1) / 2) * alto
        ax.barh(posiciones - desplazamiento, porcentajes[nombre], height=alto * 0.9,
                color=COLOR_VERSION[nombre], label=nombre)

    ax.set_yticks(posiciones, porcentajes.index)
    ax.set_xlabel("% of rows")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    poner_titulo(ax, f"{columna} mix by version", "v1 drops the missing rows; v2/v3 keep them as 'Unknown' / 'None'")
    return guardar(fig, CARPETA / f"mix_{columna}.png")


def ejecutar(original, versiones):
    return [
        grafico_filas(original, versiones),
        grafico_income(original, versiones),
        grafico_mezcla_categorias(original, versiones, "workclass"),
        grafico_mezcla_categorias(original, versiones, "occupation"),
    ]
