#ETL en donde eliminamos las columnas que no nos sirven antes de hacer el merge
import pandas as pd

BD = pd.read_csv("archivo_final.csv", encoding='latin-1')

BD = BD.drop(columns=['NOM_ENT','NOM_MUN'], errors='ignore')

Base_final = BD[['Clave_Ent','Entidad','Cve_Municipio','Municipio','Bien_juridico_afectado','Tipo_de_delito','Subtipo_de_delito','Modalidad','Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre','Total_Anual','POB_TOTAL']]

Base_final.to_csv("Base_final.csv", index=False, encoding='utf-8')

print("¡ETL terminado con éxito! Revisa el archivo Base_final.csv")