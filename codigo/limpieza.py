"""Tres formas distintas de limpiar el mismo dataset.

v1_sin_nulos    -> Borra los duplicados y todas las filas que tengan algún nulo.
                   Es la opción más simple; sirve como punto de comparación.

v2_imputado     -> No borra nada. Los nulos pasan a ser una categoría más: "Unknown".
                   Caso especial: si workclass es "Never-worked", la ocupación no falta,
                   simplemente no existe -> la completamos con "None".

v3_transformado -> Parte de v2 y además:
                   - agrupa native.country en United-States / Other / Unknown
                   - junta capital.gain y capital.loss en capital.net
                   - agrega has.capital (1 si tiene ganancia o pérdida de capital, 0 si no)
                   - borra education porque repite la información de education.num
"""
import pandas as pd

from codigo.configuracion import CARPETA_PROCESADOS, OBJETIVO


def limpiar_v1_sin_nulos(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def limpiar_v2_imputado(df):
    df = df.copy()

    # Personas que nunca trabajaron: no tienen ocupación
    nunca_trabajo = (df["workclass"] == "Never-worked") & df["occupation"].isna()
    df.loc[nunca_trabajo, "occupation"] = "None"

    # El resto de los nulos pasan a "Unknown"
    df["workclass"] = df["workclass"].fillna("Unknown")
    df["occupation"] = df["occupation"].fillna("Unknown")
    df["native.country"] = df["native.country"].fillna("Unknown")
    return df


def limpiar_v3_transformado(df):
    df = limpiar_v2_imputado(df)

    # native.country: el 91% es United-States, el resto lo agrupamos
    es_usa_o_unknown = df["native.country"].isin(["United-States", "Unknown"])
    df.loc[~es_usa_o_unknown, "native.country"] = "Other"

    # Capital: casi todos son 0, así que lo resumimos en dos columnas
    df["capital.net"] = df["capital.gain"] - df["capital.loss"]
    df["has.capital"] = ((df["capital.gain"] > 0) | (df["capital.loss"] > 0)).astype(int)

    df = df.drop(columns=["education", "capital.gain", "capital.loss"])
    return df


def crear_versiones(df):
    return {
        "v1_sin_nulos": limpiar_v1_sin_nulos(df),
        "v2_imputado": limpiar_v2_imputado(df),
        "v3_transformado": limpiar_v3_transformado(df),
    }


def guardar_versiones(versiones):
    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    for nombre, datos in versiones.items():
        datos.to_csv(CARPETA_PROCESADOS / f"adult_{nombre}.csv", index=False)


def resumen_limpieza(original, versiones):
    """Tabla con filas, nulos, duplicados y % de >50K de cada versión (para citar en el informe)."""
    todas = {"original": original, **versiones}
    filas = []
    for nombre, datos in todas.items():
        filas.append({
            "version": nombre,
            "filas": len(datos),
            "columnas": datos.shape[1],
            "celdas_nulas": int(datos.isna().sum().sum()),
            "duplicados": int(datos.duplicated().sum()),
            "porcentaje_>50K": round((datos[OBJETIVO] == ">50K").mean() * 100, 2),
        })
    return pd.DataFrame(filas)
