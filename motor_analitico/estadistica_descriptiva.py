import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore') # Para una salida limpia en consola

# Puente Arquitectónico
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def generar_reporte_eda(df: pd.DataFrame) -> dict:
    """
    Genera la estadística descriptiva y las gráficas del EDA.
    Retorna un diccionario con la tabla de datos y la figura de Matplotlib para la web.
    """
    # 1. Agrupación a nivel municipal
    df_muni = df.groupby(['Cve_Municipio', 'Municipio', 'POB_TOTAL'])['Total_Anual'].sum().reset_index()

    # Calculamos la Tasa Global por Municipio
    df_muni['Tasa_Anual_100k'] = np.where(
        df_muni['POB_TOTAL'] > 0, 
        (df_muni['Total_Anual'] / df_muni['POB_TOTAL']) * 100000, 
        0
    )

    # 2. Estadística descriptiva a nivel municipal
    columnas_stats = ['Total_Anual', 'Tasa_Anual_100k', 'POB_TOTAL']
    resultados = {}

    for col in columnas_stats:
        data = df_muni[col]
        resultados[col] = {
            'Media': data.mean(),
            'Mediana': data.median(),
            'Desv. Est.': data.std(),
            'Mínimo': data.min(),
            'Máximo': data.max(),
            'Q1 (25%)': data.quantile(0.25),
            'Q3 (75%)': data.quantile(0.75),
            'Asimetría (Skew)': data.skew(),
            'Curtosis': data.kurtosis()
        }

    df_stats = pd.DataFrame(resultados).round(2)

    # 3. Gráficas (Lienzo de Matplotlib)
    sns.set_theme(style="whitegrid", palette="viridis")
    
    # Creamos la figura principal
    fig = plt.figure(figsize=(20, 24))
    
    # HISTOGRAMA DE TASA
    ax1 = plt.subplot(3, 2, 1)
    sns.histplot(df_muni['Tasa_Anual_100k'], bins=50, kde=True, color='crimson', ax=ax1)
    ax1.set_title('1. Distribución de la Tasa Delictiva (Histograma)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tasa de Delitos por 100k Hab.')
    ax1.set_ylabel('Frecuencia (N° de Municipios)')
    ax1.text(0.5, -0.15, '¿La criminalidad es uniforme? R: No. Se agrupa a la izquierda (baja incidencia).', ha='center', transform=ax1.transAxes, color='darkred')

    # BOXPLOT DE TASA
    ax2 = plt.subplot(3, 2, 2)
    sns.boxplot(x=df_muni['Tasa_Anual_100k'], color='orange', ax=ax2)
    ax2.set_title('2. Detección de Municipios Atípicos (Boxplot)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tasa de Delitos por 100k Hab.')
    ax2.text(0.5, -0.15, 'Los puntos a la derecha son outliers. Requieren análisis focalizado.', ha='center', transform=ax2.transAxes, color='darkred')

    # 10 BIENES JURÍDICOS
    ax3 = plt.subplot(3, 2, 3)
    top_bienes = df.groupby('Bien_juridico_afectado')['Total_Anual'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_bienes.values, y=top_bienes.index, palette='mako', ax=ax3)
    ax3.set_title('3. Top 10 Bienes Jurídicos Más Afectados', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Total de Delitos Anuales')
    ax3.set_ylabel('')

    # 10 MUNICIPIOS POR TASA
    ax4 = plt.subplot(3, 2, 4)
    top_municipios = df_muni[df_muni['POB_TOTAL'] > 5000].sort_values('Tasa_Anual_100k', ascending=False).head(10)
    sns.barplot(x='Tasa_Anual_100k', y='Municipio', data=top_municipios, palette='rocket', ax=ax4)
    ax4.set_title('4. Top 10 Municipios con Mayor Riesgo Relativo', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Tasa por 100k Hab.')
    ax4.set_ylabel('')

    # CURVA DE LORENZ
    ax5 = plt.subplot(3, 2, 5)
    delitos_ordenados = np.sort(df_muni['Total_Anual'].values)
    lorenz_curve = np.cumsum(delitos_ordenados) / np.sum(delitos_ordenados)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)
    x_lorenz = np.linspace(0.0, 1.0, lorenz_curve.size)

    try:
        gini = 1 - 2 * np.trapezoid(y=lorenz_curve, x=x_lorenz)
    except AttributeError:
        gini = 1 - 2 * np.trapz(y=lorenz_curve, x=x_lorenz)

    ax5.plot(x_lorenz, lorenz_curve, color='purple', lw=3, label=f'Curva de Delitos (Gini: {gini:.3f})')
    ax5.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Línea de Equidad Perfecta')
    ax5.fill_between(x_lorenz, x_lorenz, lorenz_curve, color='purple', alpha=0.1)
    ax5.set_title('5. Curva de Lorenz (Concentración del Delito)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('% Acumulado de Municipios')
    ax5.set_ylabel('% Acumulado de Delitos')
    ax5.legend(loc='upper left')
    ax5.text(0.5, -0.15, 'Mientras más se aleja la curva, mayor es la concentración (Pareto).', ha='center', transform=ax5.transAxes, color='darkred')

    # HEATMAP DE CORRELACIÓN
    ax6 = plt.subplot(3, 2, 6)
    matriz_corr = df_muni[['POB_TOTAL', 'Total_Anual', 'Tasa_Anual_100k']].corr()
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=1, ax=ax6)
    ax6.set_title('6. Matriz de Correlación (Heatmap)', fontsize=14, fontweight='bold')
    ax6.text(0.5, -0.15, '¿La población determina la tasa? Observa la correlación.', ha='center', transform=ax6.transAxes, color='darkred')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    # EN LUGAR DE GUARDAR, DEVOLVEMOS LOS OBJETOS A LA INTERFAZ
    return {
        "estadisticas": df_stats,
        "figura_matplotlib": fig
    }

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    reporte = generar_reporte_eda(dataset)
    print(reporte["estadisticas"])