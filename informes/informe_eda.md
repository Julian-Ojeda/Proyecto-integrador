# Informe EDA – Adult Census Income

> **Cómo usar esta plantilla:** cada sección tiene los gráficos enlazados, los números clave
> que ya calculamos y preguntas guía (💬). Respondé las preguntas con tus palabras y después
> borrá las líneas 💬. Los números salen de correr `python ejecutar_eda.py`.

---

## 1. Descripción del dataset

- **Fuente:** UCI / Kaggle `uciml/adult-census-income` (censo de EE.UU. de 1994).
- **Tamaño:** 32,561 filas × 15 columnas.
- **Objetivo:** `income` → `<=50K` / `>50K` (clasificación binaria).
- **Numéricas (6):** `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`.
- **Categóricas (8):** `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`.

![target](graficos/02_univariado/target_income.png)

💬 ¿Qué significa que el dataset esté desbalanceado (76% / 24%)? ¿Qué problema puede traer
al entrenar un modelo, y qué métrica conviene mirar además del accuracy?

---

## 2. Calidad de los datos

### 2.1 Valores faltantes

En el archivo original los faltantes no están vacíos: vienen escritos como `"?"`.
Al cargarlos con `na_values=["?"]` pandas los reconoce como nulos.

![nulos por columna](graficos/01_calidad/missing_by_column.png)

| Columna | Nulos | % |
|---|---|---|
| `occupation` | 1,843 | 5.7% |
| `workclass` | 1,836 | 5.6% |
| `native.country` | 583 | 1.8% |

### 2.2 ¿Los nulos aparecen juntos?

![patrones](graficos/01_calidad/missing_patterns.png)

- Las **1,836** filas sin `workclass` tampoco tienen `occupation`.
- Las **7** filas que solo tienen nulo en `occupation` son personas con `workclass = Never-worked`:
  el dato no falta, **no aplica**.
- `native.country` falta de forma independiente.

💬 ¿Qué te dice que `workclass` y `occupation` falten siempre juntas? ¿Es un error al azar
o probablemente es una persona que no respondió la parte laboral del censo?

### 2.3 ¿Los nulos están relacionados con income? (sesgo)

![income según nulos](graficos/01_calidad/income_by_missing.png)

- Filas completas: **24.9%** gana >50K.
- Filas con algún nulo: **13.9%** gana >50K.

💬 Si borramos las filas con nulos, ¿a qué tipo de personas estamos sacando del dataset?
¿Por qué eso es un problema? *(pista: los nulos **no son al azar**)*

### 2.4 Duplicados

![duplicados](graficos/01_calidad/duplicates.png)

24 filas son copia exacta de otra (0.07%).

💬 Con 15 columnas, ¿pueden ser dos personas reales distintas con el mismo perfil?
¿Por qué decidimos borrarlos solo en v1?

---

## 3. Análisis univariado

### 3.1 Variables numéricas

| Variable | Gráfico | Qué mirar |
|---|---|---|
| `age` | ![](graficos/02_univariado/num_age.png) | asimetría a la derecha, pico en 90 |
| `fnlwgt` | ![](graficos/02_univariado/num_fnlwgt.png) | es un **peso muestral** del censo, no una característica de la persona |
| `education.num` | ![](graficos/02_univariado/num_education.num.png) | valores concentrados en 9, 10 y 13 |
| `hours.per.week` | ![](graficos/02_univariado/num_hours.per.week.png) | pico enorme en 40 |
| `capital.gain` | ![](graficos/02_univariado/num_capital.gain.png) | 91.7% son 0; 159 filas con 99999 |
| `capital.loss` | ![](graficos/02_univariado/num_capital.loss.png) | 95.3% son 0 |

💬 `age`: ¿media o mediana es más grande? ¿Qué te dice eso sobre la forma de la distribución?
¿Qué pasa con el grupo de 90 años?

💬 `hours.per.week`: ¿por qué hay tanta gente exactamente en 40?

💬 `capital.gain`: ¿99999 es un valor real o un **tope** del formulario (dato censurado)?
¿Lo tratarías como outlier y lo borrarías? ¿Por qué no?

💬 `fnlwgt`: ¿tiene sentido usarla para predecir el ingreso de una persona?

### 3.2 Variables categóricas

| | |
|---|---|
| ![](graficos/02_univariado/cat_workclass.png) | ![](graficos/02_univariado/cat_education.png) |
| ![](graficos/02_univariado/cat_marital.status.png) | ![](graficos/02_univariado/cat_occupation.png) |
| ![](graficos/02_univariado/cat_relationship.png) | ![](graficos/02_univariado/cat_race.png) |
| ![](graficos/02_univariado/cat_sex.png) | ![](graficos/02_univariado/cat_native.country.png) |

💬 ¿Qué categorías dominan cada variable? ¿Hay categorías con muy pocos casos
(ej. `Armed-Forces`, `Never-worked`, `Without-pay`) que convenga agrupar?

💬 `native.country`: 91% es United-States. ¿Cuánta información aportan los otros 40 países?

---

## 4. Relación con el objetivo (`income`)

> En todos estos gráficos la **línea punteada** es el % general de >50K (24.1%).
> Una categoría a la derecha de la línea gana más que el promedio; a la izquierda, menos.

### 4.1 Numéricas

| | |
|---|---|
| ![](graficos/03_contra_income/num_age.png) | ![](graficos/03_contra_income/num_hours.per.week.png) |
| ![](graficos/03_contra_income/num_education.num.png) | ![](graficos/03_contra_income/num_fnlwgt.png) |
| ![](graficos/03_contra_income/num_capital.gain.png) | ![](graficos/03_contra_income/num_capital.loss.png) |

💬 `age`: la mediana de >50K es 44 y la de <=50K es 34. ¿Cómo lo explicarías?

💬 `education.num`: ¿a partir de qué nivel el % de >50K supera el promedio?

💬 `fnlwgt`: ¿las dos curvas se diferencian? ¿Qué conclusión sacás?

💬 `capital.gain`: ¿por qué el rango 1-2k tiene 0% de >50K y 5k-10k tiene 84%?

### 4.2 Categóricas

| | |
|---|---|
| ![](graficos/03_contra_income/cat_marital.status.png) | ![](graficos/03_contra_income/cat_relationship.png) |
| ![](graficos/03_contra_income/cat_occupation.png) | ![](graficos/03_contra_income/cat_education.png) |
| ![](graficos/03_contra_income/cat_workclass.png) | ![](graficos/03_contra_income/cat_sex.png) |
| ![](graficos/03_contra_income/cat_race.png) | ![](graficos/03_contra_income/cat_native.country.png) |

💬 ¿Cuáles son las 3 variables donde las barras cambian más de una categoría a otra?
Esas son candidatas a ser las más importantes para el modelo.

💬 `occupation`: ¿dónde queda la categoría `NaN ('?')` respecto del promedio? ¿Coincide con lo de la sección 2.3?

---

## 5. Correlaciones y redundancia

![pearson](graficos/04_correlaciones/pearson_numeric.png)
![cramer](graficos/04_correlaciones/cramers_v_categorical.png)

Valores destacados de V de Cramér:
- `sex` – `relationship`: **0.65** (Husband/Wife ya codifican el sexo)
- `marital.status` – `relationship`: **0.49**
- `income` – `marital.status` / `relationship`: **0.45**
- `education` – `education.num`: misma información (cada nivel educativo tiene un número)

💬 ¿Hay variables numéricas muy correlacionadas entre sí? ¿Y con income?

💬 ¿Por qué tener dos variables que dicen lo mismo puede ser un problema? ¿Cuál borrarías?

---

## 6. Decisiones de limpieza

Armamos **tres versiones** del mismo dataset para comparar después en el modelado.
Están en `datos/procesados/`.

| Versión | Qué hace con los nulos | Duplicados | Otras transformaciones |
|---|---|---|---|
| `v1_sin_nulos` | Borra las filas | Borrados | — |
| `v2_imputado` | `"Unknown"` (y `"None"` para `occupation` de `Never-worked`) | Se mantienen | — |
| `v3_transformado` | Igual que v2 | Se mantienen | `native.country` → US / Other / Unknown · `capital.net` + `has.capital` · se borra `education` |

![filas](graficos/05_versiones/rows_per_version.png)
![income por versión](graficos/05_versiones/income_by_version.png)
![mezcla workclass](graficos/05_versiones/mix_workclass.png)

| version | filas | columnas | celdas_nulas | duplicados | % >50K |
|---|---|---|---|---|---|
| original | 32,561 | 15 | 4,262 | 24 | 24.08 |
| v1_sin_nulos | 30,139 | 15 | 0 | 0 | 24.90 |
| v2_imputado | 32,561 | 15 | 0 | 24 | 24.08 |
| v3_transformado | 32,561 | 14 | 0 | 24 | 24.08 |

💬 Explicá con tus palabras por qué elegimos `"Unknown"` en vez de rellenar con la moda
(la categoría más frecuente). *(pista: sección 2.3, los nulos tienen menos >50K)*

💬 v1 pierde 2,422 filas (7.4%) y sube el % de >50K de 24.1 a 24.9. ¿Es grave? ¿Por qué?

💬 ¿Qué ventaja y qué desventaja tiene cada versión?

---

## 7. Conclusiones

💬 Escribí 4 o 5 puntos con lo más importante que encontraste. Por ejemplo:
1. Sobre la calidad de los datos (nulos, duplicados, valores tope).
2. Sobre el desbalance del objetivo.
3. Las variables que más se relacionan con income.
4. Las variables redundantes o poco útiles.
5. Qué versión del dataset pensás usar primero y por qué.
