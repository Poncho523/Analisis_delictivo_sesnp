import pandas as pd
import sys
from pathlib import Path
from scipy.stats import chi2_contingency

# Importar capas
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from dominio.municipio import Municipio 

def analizar_dependencia_demografica(df: pd.DataFrame):
    """
    Ejecuta la prueba Chi-cuadrada para encontrar dependencia entre 
    el tipo de asentamiento y el tipo de delito cometido.
    """
    print("Aplicando reglas de Dominio (POO) sobre el dataset...")
    
    # 1. EL PUENTE POO-PANDAS: Usamos el método de tu clase para crear la categoría
    df['Asentamiento'] = df['POB_TOTAL'].apply(Municipio.clasificar_asentamiento)
    
    # 2. Construcción de la Tabla de Contingencia
    # Cruzamos el "Bien Jurídico" (Robo, Homicidio, etc.) vs "Asentamiento"
    print("Construyendo tabla de contingencia...")
    #CÓDIGO NUEVO (Suma los crímenes reales)
    tabla_contingencia = pd.crosstab(
        index=df['Bien_juridico_afectado'], 
        columns=df['Asentamiento'], 
        values=df['Total_Anual'], 
        aggfunc='sum'
    ).fillna(0) # fillna(0) por si alguna combinación está vacía

    print("\n TABLA DE CONTINGENCIA (Frecuencia Absoluta):")
    print(tabla_contingencia)
    
    # 3. La Prueba Matemática (Chi-cuadrada)
    print("\nEjecutando prueba estadística Chi-cuadrada...")
    chi2, p_value, grados_libertad, esperados = chi2_contingency(tabla_contingencia)
    
    # 4. Interpretación de Resultados (Valor Alpha estandar = 0.05)
    print(f"\nResultados de la prueba:")
    print(f"- Estadístico Chi-cuadrada: {chi2:.2f}")
    print(f"- Valor p (p-value): {p_value:.6e}")
    
    print("\n VEREDICTO CRIMINOLÓGICO:")
    if p_value < 0.05: #Nivel de significancia del 5%
        print("Rechazamos la Hipótesis Nula (H0).")
        print(" EXISTE DEPENDENCIA: El tipo de delito que se comete DEPENDE TOTALMENTE de si el lugar es Rural, Urbano o Megalópolis. El comportamiento criminal muta según la demografía.")
    else:
        print("Aceptamos la Hipótesis Nula (H0).")
        print(" NO HAY DEPENDENCIA: Los crímenes se distribuyen de manera similar sin importar el tamaño poblacional del municipio.")

if __name__ == "__main__":
    # Cargar datos e inyectarlos al modelo
    dataset = cargar_datos_incidencia()
    analizar_dependencia_demografica(dataset)