import pickle

# Cargar modelo elegido
with open("models/pipeline_xgb.pkl", "rb") as f:
    final_model = pickle.load(f)

# Guardar como final_model.pkl
with open("models/final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)