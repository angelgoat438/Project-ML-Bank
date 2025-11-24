

# Aqui voy a meetr los pasor para generar de manera automática el csv de la carpeta de data/processed

import pandas as pd

RAW_DIR = "data/raw" #donde está los datos originales
PROCESSED_DIR = "data/processed"  #donde van los datos procesados

# funcion d caragr los datos:

def carga_datos(nombre_file):
    return pd.read_csv(f"{RAW_DIR}/{nombre_file}",sep=";")


def procesado_data(df):   # Se transforman los datos para tenerlos listos para los modelos. (Todo está distribuido en el notebook)
    df = df.copy()

    df["min_duration"] = df["duration"] / 60
    df = df.drop(columns="duration")
    df["pdays_contacted"] = (df["pdays"] != -1).astype(int)   
    df["y"] = df["y"].map({"no": 0, "yes": 1})
    return df  # me devuelve el dataframe limpio y listo para aplicar los modelos


def procesad_guardado(nombre_file, cambio_nombre):  # Se aplican los datos pasos anteriores y se manda a la ruta elegida
    df = carga_datos(nombre_file)
    df = procesado_data(df)

    df.to_csv(f"{PROCESSED_DIR}/{cambio_nombre}",index=False)
    print(f"Limpieza y datos realizada con éxito a {PROCESSED_DIR}/{cambio_nombre}")


if __name__ == "__main__":
    procesad_guardado("bank-full.csv","datos_processed.csv")