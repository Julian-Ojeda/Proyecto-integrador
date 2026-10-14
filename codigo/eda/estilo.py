"""Colores y estilo común para que todos los gráficos se vean iguales."""
import matplotlib

matplotlib.use("Agg")  # genera las imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from codigo.configuracion import OBJETIVO

# Colores base
FONDO = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SUAVE = "#52514e"
GRILLA = "#e4e3df"
GRIS = "#b5b4ae"  # para nulos, "Other" y la versión original

AZUL = "#2a78d6"
NARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"

# Cada clase de income tiene SIEMPRE el mismo color en todos los gráficos
COLOR_INCOME = {"<=50K": AZUL, ">50K": NARANJA}
COLOR_VERSION = {"original": GRIS, "v1_sin_nulos": AZUL, "v2_imputado": NARANJA, "v3_transformado": VERDE_AGUA}

# Escala de un solo color (de claro a oscuro) para valores de 0 a 1
ESCALA_AZUL = LinearSegmentedColormap.from_list("azules", ["#f1f6fd", "#9cc3ef", AZUL, "#123f78"])
# Escala de dos colores con gris en el medio, para valores de -1 a 1
ESCALA_DIVERGENTE = LinearSegmentedColormap.from_list("azul_naranja", [AZUL, "#f2f1ee", NARANJA])


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO,
        "axes.facecolor": FONDO,
        "savefig.facecolor": FONDO,
        "axes.edgecolor": GRILLA,
        "axes.labelcolor": TEXTO_SUAVE,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRILLA,
        "xtick.color": TEXTO_SUAVE,
        "ytick.color": TEXTO_SUAVE,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def poner_titulo(ax, titulo, subtitulo=None):
    """Título en negrita y, debajo, una línea gris con el dato clave."""
    if subtitulo:
        ax.set_title(titulo, pad=26)
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9, color=TEXTO_SUAVE, va="bottom")
    else:
        ax.set_title(titulo, pad=12)


def guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


def barras_100(ax, porcentajes, linea_referencia=None):
    """Barras horizontales apiladas al 100% con el reparto de income.

    `porcentajes` tiene una fila por grupo y una columna por clase (">50K", "<=50K").
    >50K va a la izquierda para leerlo directo contra el 0.
    """
    izquierda = np.zeros(len(porcentajes))
    posiciones = range(len(porcentajes))

    for clase in [">50K", "<=50K"]:
        valores = porcentajes[clase].values
        ax.barh(posiciones, valores, left=izquierda, height=0.68, color=COLOR_INCOME[clase],
                edgecolor=FONDO, linewidth=1.5, label=clase)
        # Escribimos el % dentro de la barra solo si entra
        for y in posiciones:
            if valores[y] >= 7:
                ax.text(izquierda[y] + valores[y] / 2, y, f"{valores[y]:.1f}%", ha="center", va="center",
                        fontsize=8, color="white", weight="bold", zorder=6)
        izquierda += valores

    if linea_referencia is not None:
        ax.axvline(linea_referencia, color=TEXTO, lw=1, ls="--", zorder=5,
                   label=f"overall >50K\n({linea_referencia:.1f}%)")

    ax.set_xlim(0, 100)
    ax.set_xlabel("% of rows")
    ax.grid(axis="y", visible=False)
    ax.legend(title=OBJETIVO, loc="lower left", bbox_to_anchor=(1.0, 0.0))
