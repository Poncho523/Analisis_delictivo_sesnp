import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy.stats import chi2_contingency

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from dominio.municipio import Municipio

def calcular_dependencia_demografica(df: pd.DataFrame) -> dict:
    """
    Evalúa la relación entre la zona demográfica y el tipo de crimen,
    traduciendo la matemática pesada en alertas legibles para el usuario.
    """
    df_temp = df.copy()
    df_temp['Asentamiento'] = df_temp['POB_TOTAL'].apply(Municipio.clasificar_asentamiento)
    
    tabla_observada = pd.crosstab(
        index=df_temp['Bien_juridico_afectado'], 
        columns=df_temp['Asentamiento'], 
        values=df_temp['Total_Anual'], 
        aggfunc='sum'
    ).fillna(0)
    
    # Matemática Cruda (
    chi2, p_value, _, esperados = chi2_contingency(tabla_observada)
    n_total = tabla_observada.sum().sum() 
    min_dim = min(tabla_observada.shape[0] - 1, tabla_observada.shape[1] - 1)
    
    v_cramer = np.sqrt(chi2 / (n_total * min_dim))
    residuos = (tabla_observada - esperados) / np.sqrt(esperados)
    
    
    # V de Cramér
    if v_cramer < 0.1:
        fuerza_texto = "Débil (El entorno influye poco)"
    elif v_cramer < 0.3:
        fuerza_texto = "Moderada (El entorno es un factor importante)"
    else:
        fuerza_texto = "Fuerte (El entorno determina el crimen)"

    # Convertimos la matriz de residuos en una lista plana para ordenarla
    residuos_planos = residuos.unstack().reset_index()
    residuos_planos.columns = ['Asentamiento', 'Bien_juridico', 'Gravedad']
    
    # Filtramos solo los que superan el umbral estadístico de atención (+2.0)
    focos_rojos_df = residuos_planos[residuos_planos['Gravedad'] > 2.0].sort_values(by='Gravedad', ascending=False)
    
    # Creamos una lista de mensajes legibles
    alertas = []
    for _, fila in focos_rojos_df.head(3).iterrows(): # Solo mostramos el Top 3 para no saturar
        mensaje = f"🔴 En zona {fila['Asentamiento'].upper()}, el delito contra '{fila['Bien_juridico']}' es anormalmente alto."
        alertas.append(mensaje)
        
    # Empaquetado final para la interfaz
    return {
        "diagnostico": {
            "existe_relacion": p_value < 0.05,
            "fuerza_relacion": fuerza_texto,
            "valor_cramer": round(v_cramer, 4)
        },
        "top_focos_rojos": alertas,
        # Devolvemos la probabilidad condicional para graficarla en barras después
        "perfil_criminal_porcentajes": tabla_observada.div(tabla_observada.sum(axis=0), axis=1) * 100 
    }

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    resultado = calcular_dependencia_demografica(dataset)
    
    print(f"¿El tipo de ciudad afecta al crimen?: {'Sí' if resultado['diagnostico']['existe_relacion'] else 'No'}")
    print(f"Fuerza de esta relación: {resultado['diagnostico']['fuerza_relacion']}")
    
    print("\n=== PRINCIPALES FOCOS ROJOS DEMOGRÁFICOS ===")
    for alerta in resultado['top_focos_rojos']:
        print(alerta)