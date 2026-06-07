import pandas as pd
import sys
from pathlib import Path

# Agregar la raíz del proyecto al path para poder importar el módulo ETL
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def calcular_pareto_municipios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la concentración delictiva para verificar el Principio de Pareto.
    Agrupa las tasas por municipio y calcula el porcentaje acumulado.
    """
    # 1. Agrupar por municipio y sumar la tasa total de todos sus delitos
    print("Agrupando tasas de incidencia por municipio...")
    df_agrupado = df.groupby(['Clave_Ent', 'Entidad', 'Cve_Municipio', 'Municipio'])['Tasa_Anual_100k'].sum().reset_index()
    
    # 2. Ordenar de mayor a menor (Los municipios más violentos arriba)
    df_ordenado = df_agrupado.sort_values(by='Tasa_Anual_100k', ascending=False).reset_index(drop=True)
    
    # 3. Calcular porcentajes para Pareto
    tasa_total_nacional = df_ordenado['Tasa_Anual_100k'].sum()
    
    # Porcentaje que aporta cada municipio
    df_ordenado['Porcentaje_Aportacion'] = (df_ordenado['Tasa_Anual_100k'] / tasa_total_nacional) * 100
    
    # Porcentaje acumulado (La clave para la curva de Pareto)
    df_ordenado['Porcentaje_Acumulado'] = df_ordenado['Porcentaje_Aportacion'].cumsum()
    
    return df_ordenado

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    
    resultados_pareto = calcular_pareto_municipios(dataset)
    
    print("\n TOP 5 MUNICIPIOS CON MAYOR TASA DE INCIDENCIA:")
    print(resultados_pareto[['Municipio', 'Entidad', 'Tasa_Anual_100k', 'Porcentaje_Acumulado']].head(5))
    
    # 4. Verificar la regla del 80/20
    municipios_80_porciento = resultados_pareto[resultados_pareto['Porcentaje_Acumulado'] <= 80]
    total_municipios = len(resultados_pareto)
    cantidad_critica = len(municipios_80_porciento)
    porcentaje_municipios = (cantidad_critica / total_municipios) * 100
    
    print(f"\nCONCLUSIÓN DE PARETO:")
    print(f"El 80% de la incidencia delictiva se concentra en {cantidad_critica} municipios.")
    print(f"Esto representa el {porcentaje_municipios:.2f}% del total de municipios del país.")