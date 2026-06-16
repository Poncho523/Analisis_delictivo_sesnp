import pandas as pd
from sqlalchemy import create_engine
import csv
from io import StringIO

#Conectamos Postgre con Dcoker y leemos los archivos csv que teniamos(municipios y delitos)
def inyeccion_masiva_copy(table, conn, keys, data_iter):
    #nos deja usar funciones del controlador de Postgre
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO() #se hace un búfer en donde se va a guardar el CSV
        writer = csv.writer(s_buf) #escribe las filas en el búfer
        writer.writerows(data_iter) #escribe todas las filas de data_iter en el búfer
        s_buf.seek(0) #nos vamos al incio para q se lea desde el inicio para el copy

        columnas = ', '.join(f'"{k}"' for k in keys) #cadena con los nombre de las columnas
        nombre_tabla = f'"{table.name}"'
        sql = f'COPY {nombre_tabla} ({columnas}) FROM STDIN WITH CSV'
        
        cur.copy_expert(sql=sql, file=s_buf) #se ejecuta el copy y el contenido del búfer se copia en la tabla

conexion_url = 'postgresql://andy:super_secreto@127.0.0.1:5455/bd_seguridad'

engine = create_engine(conexion_url)

df_delitos = pd.read_csv('BD_limpia_final.csv') 
df_poblacion = pd.read_csv('BD_limpia_municipios.csv') 

df_poblacion.to_sql('poblacion_cruda', engine, if_exists='replace', index=False, method=inyeccion_masiva_copy)

df_delitos.to_sql('delitos_crudos', engine, if_exists='replace', index=False, method=inyeccion_masiva_copy)
