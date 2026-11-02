# Binary Classification – Predicción de abandono de clientes bancarios (Churn)

Proyecto para la competencia de Kaggle **Binary Classification with a Synthetic Bank Churn Dataset**. El objetivo es predecir la probabilidad de que un cliente abandone el banco (`Exited`). La métrica de la competencia es el **AUC-ROC**, por lo que los archivos de envío contienen probabilidades, no clases 0/1.

## Contenido del repositorio

| Archivo / carpeta | Descripción |
|---|---|
| `train.csv` | Datos de entrenamiento (15 000 filas, incluye la columna objetivo `Exited`) |
| `test.csv` | Datos de prueba (10 000 filas, sin `Exited`) |
| `sample_submission.csv` | Formato esperado del envío (`id`, `Exited`) |
| `train_models.py` | Script que entrena 10 modelos y genera un envío por cada uno |
| `Clasificacion.ipynb` | Notebook didáctico: caso de negocio, EDA, feature engineering y comparación de modelos |
| `submissions/` | CSV de envío generados y `validation_scores.csv` con las métricas de validación |

## Requisitos

- Python 3.9 o superior
- Librerías: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`
- Para el notebook, además: `matplotlib`, `seaborn`, `jupyter`

## Instalación

```bash
# Desde la carpeta del proyecto
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn jupyter
```

> En macOS, LightGBM puede requerir OpenMP: `brew install libomp`.

## Uso

### 1. Entrenar los modelos y generar los envíos

```bash
python train_models.py
```

El script:

1. Lee `train.csv`, `test.csv` y `sample_submission.csv` desde la carpeta del script.
2. Crea variables nuevas (`AgeSquared`, `CreditScoreAge`, `BalanceSalaryRatio`, `IsZeroBalance`, `ProductsPerAge`) y descarta `id`, `CustomerId` y `Surname`.
3. Separa un 22 % de validación (estratificado, `random_state=42`) y calcula **AUC** y **accuracy** de cada modelo.
4. Reentrena cada modelo con todo `train.csv` y predice sobre `test.csv`.
5. Guarda los resultados en `submissions/`.

Al terminar imprime la tabla de métricas y el número de archivos generados.

### 2. Resultados generados (`submissions/`)

- `01_logistic_regression.csv` … `10_lightgbm.csv`: un envío por modelo, con el formato `id,Exited`.
- `validation_scores.csv`: métricas de validación, ordenadas de menor a mayor AUC. La columna `delivery` indica el orden de entrega sugerido (del peor al mejor).
- `ensemble_top3.csv`: promedio de las probabilidades de los 3 mejores modelos por AUC de validación.

Puntajes de validación de la última ejecución:

| Modelo | AUC | Accuracy |
|---|---|---|
| 09_xgboost | 0.9366 | 0.8979 |
| 08_gradient_boosting | 0.9365 | 0.8988 |
| 10_lightgbm | 0.9357 | 0.8988 |
| 07_hist_gradient_boosting | 0.9357 | 0.9018 |
| 04_random_forest | 0.9329 | 0.8761 |
| 05_extra_trees | 0.9283 | 0.8739 |
| 03_decision_tree | 0.9195 | 0.8582 |
| 02_knn | 0.9168 | 0.8812 |
| 06_adaboost | 0.9165 | 0.8691 |
| 01_logistic_regression | 0.8941 | 0.8130 |

### 3. Subir un envío a Kaggle

Sube cualquier CSV de `submissions/` en la pestaña **Submit Predictions** de la competencia, o con la CLI de Kaggle:

```bash
kaggle competitions submit -c playground-series-s4e1 \
  -f submissions/ensemble_top3.csv -m "Ensemble top 3"
```

> Verifica el identificador de la competencia (`-c`) en la URL de Kaggle; el de arriba es un ejemplo.

### 4. Ejecutar el notebook

```bash
jupyter notebook Clasificacion.ipynb
```

El notebook fue escrito para Google Colab y carga los datos desde `sample_data/train.csv`. Para ejecutarlo en local, cambia esas rutas por `train.csv` / `test.csv`. En Colab, sube los CSV a la carpeta `sample_data/`.

## Personalización

- **Hiperparámetros:** edita `build_models()` en `train_models.py`.
- **Nuevas variables:** agrégalas en `add_features()`; se aplican igual a train y test.
- **Tamaño de validación o semilla:** cambia `test_size` en `main()` o la constante `RANDOM_STATE`.
- **Agregar un modelo:** añade una entrada al diccionario de `build_models()` con un prefijo numérico (`11_nombre`). Se entrenará y exportará automáticamente.

## Notas

- Todas las ejecuciones usan `random_state=42`, por lo que los resultados son reproducibles en la misma versión de las librerías.
- Los datos son sintéticos, generados a partir del dataset real de churn bancario.
