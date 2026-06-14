#ETL de municipios, quitar los acentos del csv de municipios, eliminar año, clave_ent, sexo, ano y poblaciones
import pandas as pd
import unicodedata

municipios = pd.read_csv("data.csv", encoding='latin-1')

def quitar_acentos(texto):
    if not isinstance(texto, str):
        return texto
    # Normaliza el texto a NFD (separa la letra del acento)
    # Luego codifica a ASCII ignorando los errores y decodifica de nuevo
    return unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')

municipios.columns = [quitar_acentos(col) for col in municipios.columns]

municipios = municipios.applymap(lambda x: quitar_acentos(x) if isinstance(x, str) else x)

municipios = municipios.drop(columns=['Anio','CLAVE_ENT','SEXO','ANO','POB_00_04','POB_05_09','POB_010_014','POB_015_019','POB_20_24','POB_25_29','POB_30_34','POB_35_39','POB_40_44','POB_45_49','POB_50_54','POB_55_59','POB_60_64','POB_65_69','POB_70_74','POB_75_79','POB_80_84','POB_85_mm','fecha','etiqueta_estado','temporal_fecha','..anio_fecha'], errors='ignore')

municipios.columns = municipios.columns.str.strip().str.replace(' ', '_')

BD_limpia_municipios = municipios[['CLAVE','NOM_ENT','NOM_MUN','POB_TOTAL']]

BD_limpia_municipios.to_csv("BD_limpia_municipios.csv", index=False, encoding='utf-8')

print("¡ETL terminado con éxito! Revisa el archivo BD_limpia_municipios.csv")