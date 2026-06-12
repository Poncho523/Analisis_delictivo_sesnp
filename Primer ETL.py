#Primer ETL(eliminar acentos, eliminar año y agregar total anual)
import pandas as pd
import unicodedata

municipios = pd.read_csv("BD.csv", encoding='latin-1')

def quitar_acentos(texto):
    if not isinstance(texto, str):
        return texto
    # Normaliza el texto a NFD (separa la letra del acento)
    # Luego codifica a ASCII ignorando los errores y decodifica de nuevo
    return unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')

municipios.columns = [quitar_acentos(col) for col in municipios.columns]

municipios = municipios.map(lambda x: quitar_acentos(x) if isinstance(x, str) else x)

municipios = municipios.drop(columns=['Anio'], errors='ignore')

meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

municipios['Total_Anual'] = municipios[meses].sum(axis=1)

municipios.columns = municipios.columns.str.strip().str.replace(' ', '_')

BD_limpia = municipios[['Clave_Ent', 'Entidad', 'Cve_Municipio', 'Municipio', 
                     'Bien_juridico_afectado', 'Tipo_de_delito', 'Subtipo_de_delito', 
                     'Modalidad', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre','Total_Anual']]
                     
BD_limpia.to_csv("BD_limpia_final.csv", index=False, encoding='utf-8')

print("¡ETL terminado con éxito! Revisa el archivo BD_limpia_final.csv")