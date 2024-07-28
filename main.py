# main.py
import os
import pandas as pd
from datetime import datetime
from src import load_data, preprocess, train_model, save_model, load_model, predict, evaluate, visualize_results
def obtener_ruta_del_script():
    ruta_absoluta = os.path.abspath(__file__)
    directorio_del_script = os.path.dirname(ruta_absoluta)
    return directorio_del_script

def main():
    # Cargar datos
    ruta=obtener_ruta_del_script()
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> Iniciando proceso")
    taxi = load_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet')
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> datos cargados")

    filas, columnas = taxi.shape
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> Filas: {filas}, Columnas: {columnas}")
    # Preprocesar datos
    target_col = "high_tip"
    taxi_train = preprocess(taxi, target_col)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} ->datos procesados")
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} ->Entrenando modelo")
    # Entrenar modelo
    X_train = taxi_train[['pickup_weekday', 'pickup_hour', 'work_hours', 'pickup_minute', 'passenger_count', 'trip_distance', 'trip_time', 'trip_speed', 'PULocationID', 'DOLocationID', 'RatecodeID']]
    y_train = taxi_train[target_col]
    model = train_model(X_train, y_train)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> modelo entrenado")

    # Guardar modelo

    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> guardando modelo en {ruta}\\models\\random_forest.joblib")
    save_model(model, f"{ruta}\\models\\random_forest.joblib")
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> modelo guardado")
    # Cargar modelo y evaluar en nuevos datos
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> cargando modelo para evaluar datos")

    model = load_model(f"{ruta}\\models\\random_forest.joblib")
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> modelo cargado")

    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> cargando datos")
    taxi_test = load_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet')
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> preprocesando datos")
    taxi_test = preprocess(taxi_test, target_col)
    X_test = taxi_test[['pickup_weekday', 'pickup_hour', 'work_hours', 'pickup_minute', 'passenger_count', 'trip_distance', 'trip_time', 'trip_speed', 'PULocationID', 'DOLocationID', 'RatecodeID']]
    y_test = taxi_test[target_col]
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> test")
    preds = predict(model, X_test)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> F1: {evaluate(y_test, preds)}")
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")} -> visualizar datos")
    # Visualizar resultados
    # Lista de características
    features = ['pickup_weekday', 'pickup_hour', 'work_hours', 'pickup_minute', 'passenger_count', 'trip_distance', 'trip_time', 'trip_speed', 'PULocationID', 'DOLocationID', 'RatecodeID']

    # Llamar a la función con la lista correcta de características
    fecha1="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet"
    fecha2="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet"
    titulo1="febrero"
    titulo2="mayo"
    save_path=f"{ruta}\\reports\\figures"
    
    visualize_results(fecha1,fecha2, titulo1, titulo2,save_path)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}-> proceso finalizado")
if __name__ == "__main__":
    main()
