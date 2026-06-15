import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy.stats import chi2_contingency

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from dominio.municipio import Municipio

def analizar_dependencia_demografica(df: pd.DataFrame):
    print("Aplicando reglas de Dominio (POO) sobre el dataset")
    df['Asentamiento'] = df['POB_TOTAL'].apply(Municipio.clasificar_asentamiento)
    
    # TABLA BASE 
    tabla_observada = pd.crosstab(
        index=df['Bien_juridico_afectado'], 
        columns=df['Asentamiento'], 
        values=df['Total_Anual'], 
        aggfunc='sum'
    ).fillna(0)
    
    print("\nCalculando métricas estadísticas...")
    chi2, p_value, grados_libertad, esperados = chi2_contingency(tabla_observada)
    
    # V DE CRAMÉR 
    # Total de delitos reales
    n_total = tabla_observada.sum().sum() 
    # Mínimo entre (filas-1) y (columnas-1)
    min_dim = min(tabla_observada.shape[0] - 1, tabla_observada.shape[1] - 1)
    v_cramer = np.sqrt(chi2 / (n_total * min_dim))
    
    # PROBABILIDAD CONDICIONAL (Teorema de Bayes)
    tabla_condicional = pd.crosstab(
        index=df['Bien_juridico_afectado'], 
        columns=df['Asentamiento'], 
        values=df['Total_Anual'], 
        aggfunc='sum',
        normalize='columns' 
    ) * 100
    
    # RESIDUOS ESTANDARIZADOS 
    residuos = (tabla_observada - esperados) / np.sqrt(esperados)
    
    print(f"\n--- 1. DIAGNÓSTICO GENERAL ---")
    print(f"Estadístico Chi-cuadrada: {chi2:.2f}")
    print(f"Valor p: {p_value:.6e}")
    print(f"Fuerza de Dependencia (V de Cramér): {v_cramer:.4f} (Escala 0 a 1)")
    
    print("\n--- 2. PERFIL CRIMINAL POR ZONA (Probabilidad Condicional %) ---")
    print("Lectura: 'Del 100% de los delitos en zonas X, el Y% fueron contra el bien Z'")
    print(tabla_condicional.round(2).astype(str) + '%')
    
    print("\n--- 3. DETECCIÓN DE ANOMALIAS (Residuos Estandarizados) ---")
    print("Lectura: Valores > +2.0 indican una concentración atípica (Foco Rojo).")
    print("Lectura: Valores < -2.0 indican una ausencia atípica (Zona Segura).")
    print(residuos.round(2))

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    analizar_dependencia_demografica(dataset)