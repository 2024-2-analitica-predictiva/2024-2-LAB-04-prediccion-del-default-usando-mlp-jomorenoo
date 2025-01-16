# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import gzip
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# Paso 1: Cargar y limpiar datos
input_dir = "files/input"
output_dir = "files/output"
model_dir = "files/models"

train_data = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_data = pd.read_csv(os.path.join(input_dir, "test.csv"))

# Renombrar columna objetivo y eliminar columnas no necesarias
for data in [train_data, test_data]:
    data.rename(columns={"default payment next month": "default"}, inplace=True)
    data.drop(columns=["ID"], inplace=True)

# Eliminar registros con información no disponible
train_data = train_data[(train_data != 0).all(axis=1)]
test_data = test_data[(test_data != 0).all(axis=1)]

# Agrupar niveles superiores de educación en "others"
for data in [train_data, test_data]:
    data["EDUCATION"] = data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

# Paso 2: Dividir en conjuntos de características y etiquetas
x_train, y_train = train_data.drop(columns="default"), train_data["default"]
x_test, y_test = test_data.drop(columns="default"), test_data["default"]

# Paso 3: Crear el pipeline
pipeline = Pipeline(
    [
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("scaler", MinMaxScaler()),
        ("pca", PCA()),
        ("select_kbest", SelectKBest(score_func=f_classif, k=10)),
        ("mlp", MLPClassifier(max_iter=500)),
    ]
)

# Paso 4: Optimizar hiperparámetros
param_grid = {
    "mlp__hidden_layer_sizes": [(50, 50), (100,)],
    "mlp__alpha": [0.0001, 0.001, 0.01],
    "mlp__learning_rate_init": [0.001, 0.01],
}

cv_model = GridSearchCV(
    pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1
)
cv_model.fit(x_train, y_train)

# Guardar modelo
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl.gz")
with gzip.open(model_path, "wb") as f:
    pickle.dump(cv_model.best_estimator_, f)

# Paso 5: Calcular métricas
metrics = []
for dataset, x, y, label in [
    ("train", x_train, y_train, "train"),
    ("test", x_test, y_test, "test"),
]:
    y_pred = cv_model.predict(x)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)

    metrics.append(
        {
            "dataset": label,
            "precision": precision,
            "balanced_accuracy": balanced_accuracy,
            "recall": recall,
            "f1_score": f1,
        }
    )

    # Calcular matriz de confusión
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": label,
        "true_0": {"predicted_0": cm[0, 0], "predicted_1": cm[0, 1]},
        "true_1": {"predicted_0": cm[1, 0], "predicted_1": cm[1, 1]},
    }
    metrics.append(cm_dict)

# Guardar métricas
os.makedirs(output_dir, exist_ok=True)
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
