import pandas as pd
import sys
from pathlib import Path

# Puente Arquitectónico
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def calcular_pareto_municipios(df: pd.DataFrame) -> dict:
    """
    Calcula la concentración delictiva para verificar el Principio de Pareto.
    Devuelve un diccionario con el DataFrame y los metadatos clave para contar la historia.
    """
    print("Agrupando tasas de incidencia por municipio...")
    df_agrupado = df.groupby(['Clave_Ent', 'Entidad', 'Cve_Municipio', 'Municipio'])['Tasa_Anual_100k'].sum().reset_index()
    
    # Ordenamos de mayor a menor (Los peores municipios primero)
    df_ordenado = df_agrupado.sort_values(by='Tasa_Anual_100k', ascending=False).reset_index(drop=True)
    
    # Calculamos el 100% de la incidencia
    tasa_total_nacional = df_ordenado['Tasa_Anual_100k'].sum()
    
    # Calculamos cuánto aporta cada municipio al problema nacional
    df_ordenado['Porcentaje_Aportacion'] = (df_ordenado['Tasa_Anual_100k'] / tasa_total_nacional) * 100
    
    # Suma acumulada para crear la curva que sube hasta llegar al 100%
    df_ordenado['Porcentaje_Acumulado'] = df_ordenado['Porcentaje_Aportacion'].cumsum()
    
    # --- NUEVO: EXTRAEMOS LA INTELIGENCIA PARA EL DASHBOARD ---
    # Filtramos para saber exactamente cuántos municipios forman el 80% del problema
    municipios_80_porciento = df_ordenado[df_ordenado['Porcentaje_Acumulado'] <= 80]
    
    cantidad_critica = len(municipios_80_porciento)
    total_municipios = len(df_ordenado)
    porcentaje_municipios = (cantidad_critica / total_municipios) * 100
    
    # En lugar de solo el dataframe, devolvemos un paquete completo
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
    print(f"El 80% de la incidencia delictiva se concentra en {resultados['cantidad_critica']} municipios.")
    print(f"Esto representa el {resultados['porcentaje_territorio']:.2f}% del total de municipios del país.")