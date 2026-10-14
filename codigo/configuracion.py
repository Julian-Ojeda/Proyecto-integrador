"""Rutas y listas de columnas que usa todo el proyecto."""
from pathlib import Path

# Carpeta raíz del proyecto (la que contiene a codigo/)
RAIZ = Path(__file__).resolve().parent.parent

CSV_ORIGINAL = RAIZ / "datos" / "crudos" / "adult.csv"
CARPETA_PROCESADOS = RAIZ / "datos" / "procesados"
CARPETA_INFORMES = RAIZ / "informes"
CARPETA_GRAFICOS = RAIZ / "informes" / "graficos"

DATASET_KAGGLE = "uciml/adult-census-income"

# Variable objetivo y sus dos clases
OBJETIVO = "income"
CLASES = ["<=50K", ">50K"]

COLUMNAS_NUMERICAS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

COLUMNAS_CATEGORICAS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]
