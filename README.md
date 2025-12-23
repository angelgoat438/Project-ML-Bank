# 🏦 Private Bank Project

## Optimización de Campañas Bancarias mediante Machine Learning

> **Objetivo:** Aumentar la eficiencia en la selección de clientes con mayor probabilidad de aceptar una póliza bancaria.  
> **Modelo final:** Easy Ensemble Classifier  
> **Métrica prioritaria:** Recall (clase positiva)  
> **Mejor resultado:** Recall = 0.93 | ROC AUC = 0.72

---

## 💰 Problema de negocio
Un banco portugués desea lanzar una campaña de marketing para la venta de una póliza concreta.  
Dado que el presupuesto es limitado, resulta clave optimizar los recursos de personal y focalizar los contactos en aquellos clientes con mayor probabilidad de aceptación, reduciendo así costes y contactos innecesarios.

---

## 📊 Dataset
- **Fuente:** Bank Marketing Dataset (UCI Machine Learning Repository)
- **Registros:** 45.211 clientes
- **Variable objetivo:** `acepta_deposito` (sí / no)
- **Desbalanceo de clases:** ~11% de clientes aceptan la póliza

---

## 🧠 Enfoque de modelado
1. Análisis Exploratorio de Datos (EDA)
2. Limpieza y preprocesamiento
3. Ingeniería de características
4. Entrenamiento y evaluación de modelos
5. Ajuste de hiperparámetros

---

## 🤖 Modelos utilizados
Se evaluaron distintos clasificadores supervisados, utilizando la regresión logística como modelo base.  
Además, se empleó un modelo no supervisado con fines exploratorios.

**Modelos supervisados:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- CatBoost Classifier
- Easy Ensemble Classifier

**Modelo no supervisado (exploratorio):**
- K-Means

Dado el fuerte desbalanceo del conjunto de datos y el objetivo de negocio, se priorizó el **Recall** de la clase positiva, aceptando un mayor número de falsos positivos con el fin de minimizar la pérdida de clientes potenciales.

---

## 📈 Resultados

| Modelo                     | ROC AUC | Recall |
|---------------------------|--------:|-------:|
| Logistic Regression       | 0.71    | 0.65   |
| Random Forest             | 0.73    | 0.68   |
| XGBoost                   | 0.74    | 0.82   |
| CatBoost Classifier       | 0.73    | 0.82   |
| Easy Ensemble Classifier  | 0.71    | 0.92   |



