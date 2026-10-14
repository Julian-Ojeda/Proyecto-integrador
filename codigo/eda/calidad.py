"""Calidad de datos: dónde están los nulos, si aparecen juntos y si afectan a income."""
import matplotlib.pyplot as plt

from codigo.configuracion import CARPETA_GRAFICOS, CLASES, OBJETIVO
from codigo.eda.estilo import AZUL, GRILLA, GRIS, TEXTO, barras_100, guardar, poner_titulo

CARPETA = CARPETA_GRAFICOS / "01_calidad"


def grafico_nulos_por_columna(df):
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values()
    porcentaje = nulos / len(df) * 100

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(nulos.index, nulos.values, color=AZUL, height=0.55)
    for y, columna in enumerate(nulos.index):
        ax.text(nulos[columna], y, f"  {nulos[columna]:,}  ({porcentaje[columna]:.1f}%)", va="center", fontsize=9)

    ax.set_xlim(0, nulos.max() * 1.3)
    ax.set_xlabel("Missing values ('?' in the raw file)")
    ax.grid(axis="y", visible=False)
    completas = len(df.columns) - len(nulos)
    poner_titulo(ax, "Missing values by column", f"{completas} of {len(df.columns)} columns are complete")
    return guardar(fig, CARPETA / "missing_by_column.png")


def grafico_patrones_de_nulos(df):
    """¿Qué columnas faltan JUNTAS en la misma fila?

    A la izquierda: cada fila es un patrón (punto negro = falta ese dato).
    A la derecha: cuántas filas tienen ese patrón.
    """
    columnas = df.columns[df.isna().any()].tolist()
    es_nulo = df[columnas].isna()
    es_nulo = es_nulo[es_nulo.any(axis=1)]  # solo filas con algún nulo
    patrones = es_nulo.value_counts().sort_values()

    fig, (ax_puntos, ax_barras) = plt.subplots(
        1, 2, figsize=(10, 3.4), sharey=True, gridspec_kw={"width_ratios": [1, 2.4], "wspace": 0.05}
    )

    for y, (patron, cantidad) in enumerate(patrones.items()):
        for x, falta in enumerate(patron):
            ax_puntos.scatter(x, y, s=110, color=TEXTO if falta else GRILLA, zorder=3)
        faltantes = [x for x, falta in enumerate(patron) if falta]
        if len(faltantes) > 1:
            ax_puntos.plot([min(faltantes), max(faltantes)], [y, y], color=TEXTO, lw=2)

    ax_puntos.set_xticks(range(len(columnas)), columnas, rotation=30, ha="right")
    ax_puntos.set_xlim(-0.5, len(columnas) - 0.5)
    ax_puntos.set_yticks([])
    ax_puntos.grid(False)
    ax_puntos.spines[["left", "bottom"]].set_visible(False)

    ax_barras.barh(range(len(patrones)), patrones.values, color=AZUL, height=0.55)
    for y, cantidad in enumerate(patrones.values):
        ax_barras.text(cantidad, y, f"  {cantidad:,}", va="center", fontsize=9)
    ax_barras.set_xlim(0, patrones.max() * 1.2)
    ax_barras.set_xlabel("Rows with this pattern")
    ax_barras.grid(axis="y", visible=False)

    poner_titulo(ax_puntos, "Missing-value patterns",
                 f"{len(es_nulo):,} rows have at least one missing value  ·  ● = missing")
    return guardar(fig, CARPETA / "missing_patterns.png")


def grafico_income_segun_nulos(df):
    """Comparamos el % de >50K entre filas completas y filas con nulos.
    Si son distintos, borrar las filas con nulos cambia el dataset (sesgo)."""
    grupo = df.isna().any(axis=1).map({False: "Complete rows", True: "Rows with missing"})
    porcentajes = df.groupby(grupo)[OBJETIVO].value_counts(normalize=True).unstack()[CLASES] * 100
    cantidades = grupo.value_counts()
    tasa_general = (df[OBJETIVO] == ">50K").mean() * 100

    fig, ax = plt.subplots(figsize=(8, 2.8))
    barras_100(ax, porcentajes, linea_referencia=tasa_general)
    ax.set_yticks(range(len(porcentajes)), [f"{g}\n(n={cantidades[g]:,})" for g in porcentajes.index])
    poner_titulo(ax, "Income share: complete vs incomplete rows",
                 "Dropping rows with missing values removes mostly <=50K people")
    return guardar(fig, CARPETA / "income_by_missing.png")


def grafico_duplicados(df):
    copias_extra = int(df.duplicated().sum())
    unicas = len(df) - copias_extra
    perfiles_repetidos = df[df.duplicated(keep=False)].drop_duplicates().shape[0]

    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(["Unique rows", "Extra copies"], [unicas, copias_extra], color=[AZUL, GRIS], height=0.5)
    ax.text(unicas, 0, f"  {unicas:,}", va="center", fontsize=9)
    ax.text(copias_extra, 1, f"  {copias_extra:,}", va="center", fontsize=9)
    ax.set_xlim(0, len(df) * 1.15)
    ax.grid(axis="y", visible=False)
    poner_titulo(ax, "Exact duplicated rows",
                 f"{perfiles_repetidos} profiles appear more than once  ·  {copias_extra / len(df) * 100:.2f}% of rows")
    return guardar(fig, CARPETA / "duplicates.png")


def ejecutar(df):
    return [
        grafico_nulos_por_columna(df),
        grafico_patrones_de_nulos(df),
        grafico_income_segun_nulos(df),
        grafico_duplicados(df),
    ]
