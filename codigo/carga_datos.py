"""Carga del dataset original."""
import shutil
from pathlib import Path

import pandas as pd

from codigo.configuracion import CSV_ORIGINAL, DATASET_KAGGLE


def descargar_si_falta():
    """Si no tenemos el CSV en datos/crudos, lo bajamos de Kaggle una sola vez y guardamos una copia."""
    if CSV_ORIGINAL.exists():
        return

    import kagglehub

    carpeta_descarga = Path(kagglehub.dataset_download(DATASET_KAGGLE))
    archivo = next(carpeta_descarga.glob("*.csv"))
    CSV_ORIGINAL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(archivo, CSV_ORIGINAL)


def cargar_datos():
    """Devuelve el dataset original.

    En el archivo los valores faltantes vienen escritos como "?".
    Con na_values=["?"] le decimos a pandas que los trate como nulos (NaN).
    """
    descargar_si_falta()
    df = pd.read_csv(CSV_ORIGINAL, na_values=["?"])

    # Por las dudas sacamos espacios sobrantes en los textos
    for columna in df.select_dtypes(include="object").columns:
        df[columna] = df[columna].str.strip()

    return df
