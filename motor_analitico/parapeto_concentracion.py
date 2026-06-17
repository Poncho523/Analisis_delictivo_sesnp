import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Puente Arquitectónico
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def calcular_pareto_municipios(df: pd.DataFrame) -> dict:
    """
    Calcula la concentración delictiva para verificar el Principio de Pareto.
    CORRECCIÓN MATEMÁTICA: Se utilizan volúmenes absolutos (Total_Anual), no tasas.
    """
    print("Agrupando volumen total de delitos por municipio...")
    
    # 1. Agrupamos sumando los delitos reales (absolutos) y conservando la población
    df_agrupado = df.groupby(['Cve_Municipio', 'Municipio', 'Entidad']).agg(
        Total_Anual=('Total_Anual', 'sum'),
        POB_TOTAL=('POB_TOTAL', 'max') # Tomamos la población máxima registrada para ese municipio
    ).reset_index()
    
    # 2. Recalculamos la Tasa SOLO para mostrarla en la tabla (no para el cálculo de Pareto)
    df_agrupado['Tasa_Anual_100k'] = np.where(
        df_agrupado['POB_TOTAL'] > 0, 
        (df_agrupado['Total_Anual'] / df_agrupado['POB_TOTAL']) * 100000, 
        0
    )
    
    # 3. Ordenamos de mayor a menor basándonos en el VOLUMEN ABSOLUTO de delitos
    df_ordenado = df.sort_values(by='Total_Anual', ascending=False).reset_index(drop=True)
    
    # 4. MATEMÁTICA DE PARETO (Sobre totales reales)
    total_delitos_nacional = df_ordenado['Total_Anual'].sum()
    
    # ¿Cuánto aporta cada municipio a la bolsa total de delitos del país?
    df_ordenado['Porcentaje_Aportacion'] = (df_ordenado['Total_Anual'] / total_delitos_nacional) * 100
    
    # Suma acumulada para crear la curva
    df_ordenado['Porcentaje_Acumulado'] = df_ordenado['Porcentaje_Aportacion'].cumsum()
    
    # 5. EXTRAEMOS LA INTELIGENCIA PARA EL DASHBOARD
    # Filtramos para saber exactamente cuántos municipios forman el 80% de TODOS los delitos
    municipios_80_porciento = df_ordenado[df_ordenado['Porcentaje_Acumulado'] <= 80]
    
    cantidad_critica = len(municipios_80_porciento)
    total_municipios = len(df_ordenado)
    porcentaje_municipios = (cantidad_critica / total_municipios) * 100
    
    return {
        "datos_grafica": df_ordenado,
        "cantidad_critica": cantidad_critica,
        "total_municipios": total_municipios,
        "porcentaje_territorio": porcentaje_municipios
    }

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    resultados = calcular_pareto_municipios(dataset)
    
    print("\nCONCLUSIÓN DE PARETO:")
    print(f"El 80% de TODOS los delitos reales se concentran en {resultados['cantidad_critica']} municipios.")
    print(f"Esto representa el {resultados['porcentaje_territorio']:.2f}% del territorio nacional.")