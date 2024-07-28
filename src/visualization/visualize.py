import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.stats import ttest_ind
from datetime import datetime
def plot_data(df, title, save_path):
    df = df.sample(1000)
    # Crear una instancia del escalador Min-Max
    scaler = MinMaxScaler()
    numeric_df = df.select_dtypes(include='number')

    # Ajustar y transformar el DataFrame
    numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

    # Crear gráficos
    num_vars = numeric_df.columns
    n_vars = len(num_vars)
    
    # Determinar el tamaño de la figura en función del número de gráficos
    n_cols = 3  # Número de columnas en los subplots
    n_rows = (n_vars + n_cols - 1) // n_cols  # Calcular el número de filas necesarias

    plt.figure(figsize=(n_cols * 6, n_rows * 4))

    for i, var in enumerate(num_vars):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(numeric_df[var], kde=True)
        plt.title(f'Histograma de {var}')
    
    # Ajustar el layout para que los subplots no se superpongan
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, y=1.02, fontsize=16)
    save_path=save_path+"\\"+title+".jpg"
    # Guardar el gráfico en la ruta especificada
    plt.savefig(save_path, bbox_inches='tight')
        
    plt.close()
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}-> Graficos guardados en {save_path}")



# Leer los datos
#taxi_feb = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-02.parquet')
#taxi_may = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-04.parquet')
def test(fecha1, fecha2):
    # Seleccionar solo las columnas numéricas
    fecha1_num = fecha1.select_dtypes(include='number')
    fecha2_num = fecha2.select_dtypes(include='number')

    # Iterar sobre las columnas numéricas
    for variable in fecha1_num.columns:
        if variable in fecha2_num.columns:
            # Extraer las series de la variable para ambos DataFrames
            fecha1_var = fecha1_num[variable].dropna()
            fecha2_var = fecha2_num[variable].dropna()

            # Realizar la prueba t
            t_stat, p_value = ttest_ind(fecha1_var, fecha2_var)

            if p_value < 0.05:
                print(f"Rechazamos la hipótesis nula para la variable o campo {variable} su p valor = {p_value:.4f} < 0.05: Las medias son significativamente diferentes.")
            else:
                print(f"No rechazamos la hipótesis nula para la variable o campo {variable} su p valor = {p_value:.4f} > 0.05: No hay evidencia suficiente para decir que las medias son diferentes.")
        else:
            print(f'La variable {variable} no se encuentra en ambos DataFrames.')
def visualize_results(fecha1,fecha2, titulo1, titulo2,save_path):
    fecha1=pd.read_parquet(fecha1)
    fecha2=pd.read_parquet(fecha2)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}-> Generando graficos {titulo1}")
    plot_data(fecha1,titulo1, save_path)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}-> Generando graficos {titulo2}")
    plot_data(fecha2, titulo2, save_path)
    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}-> Generando pruebas de hipotesis")

    print(f"{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}-> Iniciando pruebas de hipotesis")
    test(fecha1,fecha2)


