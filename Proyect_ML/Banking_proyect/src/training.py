
"""
En este script se centra en :

- Carga los datos procesados desde data/processed.
- Divide en train/test.
- Entrena el modelo.
- Guarda el modelo y los datasets train y test en las carpetas correspondientes. """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


# --------------------------------- Cargo los datos que usan los modelos ---------------------------------

X_train = pd.read_csv("./data/train/X_train.csv")
y_train = pd.read_csv("./data/train/y_train.csv")

X_test = pd.read_csv("./data/test/X_test.csv")
y_test = pd.read_csv("./data/test/y_test.csv")



 # ----------------------------------- Cargo todos los modelos ----------------------------

categoricas = [ 'job', 'marital', 'education', 'default', 'housing','loan', "poutcome","pdays_contacted"]

numerics = ['age', 'campaign', 'pdays', 'previous']

preprocessor = ColumnTransformer([
    
    ("num", StandardScaler(), numerics),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
])


# Se importn todos los modelos: 

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",  # para clasificación binaria
    use_label_encoder=False,
    random_state=42)

pipelines = {
    "pipeline_lr" : Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)) #------------ Logistic Regression -------------
]),
    "pipeline_rfc" : Pipeline([
        ("preprocess",preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced",random_state=42))  #------------- Random Forest Classifier -----------
]),

    "pipeline_xgb" : Pipeline([
        ("preprocess",preprocessor),   # ------------- XGBoost Classifier -----------------
        ("model",xgb)
]),

    "pipeline_cat": Pipeline([
        ("preprocess",preprocessor),
        ("model",CatBoostClassifier(auto_class_weights="Balanced",verbose=0,random_state=42)) # ------------- CatBoost Classifier ------------
]),
    "pipeline_eec" : Pipeline([
        ("preprocess",preprocessor),
        ("model",EasyEnsembleClassifier(n_estimators=50,random_state=42))  # ------------- Easy Emsemble Classifier ----------------
])

}

params_grids = {

    "pipeline_lr" : {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],  
        "model__solver": ["lbfgs"] 
},

    "pipeline_rfc" : {
        "model__n_estimators": [200,300,400],
        "model__max_depth":[None,10,15],
        "model__min_samples_split":[2,4],
        "model__min_samples_leaf":[1,2,3],
        "model__class_weight": ["balanced", {0:1, 1:5}]
},

    "pipeline_xgb" : {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__subsample": [0.8, 1],
        "model__colsample_bytree": [0.8, 1],
        "model__scale_pos_weight": [4, 5, 6]  # clase minoritaria ~1/0.12 = 8 → puedes probar 4-6
},

    "pipeline_cat" : {
        "model__iterations":[300,600],
        "model__depth":[4,6],
        "model__learning_rate":[0.03,0.1]
},

    "pipeline_eec" : {
        "model__n_estimators": [30, 50, 70]
    
}

}

best_models ={} # para captuarar los mejores hiperparámetros


for name, pipe in pipelines.items():
    print(f"\n🔵 Entrenando {name}...")
    
    grid_params = params_grids[name]
    
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=grid_params,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    
    best_models[name] = grid.best_estimator_
    
    print(f"   ✔ Mejores params para {name}: {grid.best_params_}")
    
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    print(f"   ✔ Modelo guardado en models/{name}.pkl") 





