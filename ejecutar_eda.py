"""Corre todo el EDA de punta a punta.

1. Carga el dataset original.
2. Genera los gráficos de calidad, univariado, bivariado y correlaciones.
3. Crea las 3 versiones limpias y las guarda en datos/procesados/.
4. Compara las versiones con gráficos y guarda un resumen en informes/resumen_limpieza.csv.

Uso:  python ejecutar_eda.py
"""
from codigo.carga_datos import cargar_datos
from codigo.configuracion import CARPETA_GRAFICOS, CARPETA_INFORMES
from codigo.eda import bivariado, calidad, correlaciones, univariado, versiones
from codigo.eda.estilo import aplicar_estilo
from codigo.limpieza import crear_versiones, guardar_versiones, resumen_limpieza


def main():
    aplicar_estilo()

    df = cargar_datos()
    print(f"Dataset original: {df.shape[0]:,} filas x {df.shape[1]} columnas\n")

    print("Calidad de datos ...", len(calidad.ejecutar(df)), "gráficos")
    print("Univariado .........", len(univariado.ejecutar(df)), "gráficos")
    print("Bivariado ..........", len(bivariado.ejecutar(df)), "gráficos")
    print("Correlaciones ......", len(correlaciones.ejecutar(df)), "gráficos")

    versiones_limpias = crear_versiones(df)
    guardar_versiones(versiones_limpias)
    print("Versiones ..........", len(versiones.ejecutar(df, versiones_limpias)), "gráficos")

    resumen = resumen_limpieza(df, versiones_limpias)
    resumen.to_csv(CARPETA_INFORMES / "resumen_limpieza.csv", index=False)
    print("\nResumen de limpieza:")
    print(resumen.to_string(index=False))
    print(f"\nGráficos guardados en: {CARPETA_GRAFICOS}")


if __name__ == "__main__":
    main()
