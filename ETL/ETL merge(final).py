#ETL haciendo el merge, agregando la población total y la tasa anueal por cada 100k
import pandas as pd
import numpy as np

df_delitos = pd.read_csv('BD_limpia_final.csv') 
df_poblacion = pd.read_csv('BD_limpia_municipios.csv')

df_poblacion_agrupada = df_poblacion.groupby('Cve_Municipio', as_index=False)['POB_TOTAL'].sum()

df_final = df_delitos.merge(
    df_poblacion_agrupada,
    on='Cve_Municipio',
    how='left',
    validate='m:1'
)

df_final['POB_TOTAL'] = df_final['POB_TOTAL'].fillna(0).astype(int)

# Calcular la tasa delictiva por cada 100,000 habitantes (Total Anual)
# np.where evita que divida entre 0
df_final['Tasa_Anual_100k'] = np.where(
    df_final['POB_TOTAL'] > 0, 
    (df_final['Total_Anual'] / df_final['POB_TOTAL']) * 100000, 
    0).round(2)

nombre_archivo_salida = 'delitos_nivelados_2015.csv'
df_final.to_csv(nombre_archivo_salida, index=False)
