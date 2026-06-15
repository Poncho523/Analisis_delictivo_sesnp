import pandas as pd
import sys
from pathlib import Path

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def calcular_pareto_municipios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la concentración delictiva para verificar el Principio de Pareto.
    Agrupa las tasas por municipio y calcula el porcentaje acumulado.
    """

    print("Agrupando tasas de incidencia por municipio...")
    df_agrupado = df.groupby(['Clave_Ent', 'Entidad', 'Cve_Municipio', 'Municipio'])['Tasa_Anual_100k'].sum().reset_index()
    

    df_ordenado = df_agrupado.sort_values(by='Tasa_Anual_100k', ascending=False).reset_index(drop=True)
    

    tasa_total_nacional = df_ordenado['Tasa_Anual_100k'].sum()
    

    df_ordenado['Porcentaje_Aportacion'] = (df_ordenado['Tasa_Anual_100k'] / tasa_total_nacional) * 100
    

    df_ordenado['Porcentaje_Acumulado'] = df_ordenado['Porcentaje_Aportacion'].cumsum()
    
    return df_ordenado

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    
    resultados_pareto = calcular_pareto_municipios(dataset)
    
    print("\n TOP 5 MUNICIPIOS CON MAYOR TASA DE INCIDENCIA:")
    print(resultados_pareto[['Municipio', 'Entidad', 'Tasa_Anual_100k', 'Porcentaje_Acumulado']].head(5))
    

    municipios_80_porciento = resultados_pareto[resultados_pareto['Porcentaje_Acumulado'] <= 80]
    total_municipios = len(resultados_pareto)
    cantidad_critica = len(municipios_80_porciento)
    porcentaje_municipios = (cantidad_critica / total_municipios) * 100
    
    print(f"\nCONCLUSIÓN DE PARETO:")
    print(f"El 80% de la incidencia delictiva se concentra en {cantidad_critica} municipios.")
    print(f"Esto representa el {porcentaje_municipios:.2f}% del total de municipios del país.")

    #Redactar una conclusion con base en lo ya resultado
    