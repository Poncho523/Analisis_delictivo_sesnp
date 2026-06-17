import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def generar_reporte_eda(df: pd.DataFrame) -> dict:
    df_muni = df.groupby(['Cve_Municipio', 'Municipio', 'POB_TOTAL'])['Total_Anual'].sum().reset_index()

    df_muni['Tasa_Anual_100k'] = np.where(
        df_muni['POB_TOTAL'] > 0, 
        (df_muni['Total_Anual'] / df_muni['POB_TOTAL']) * 100000, 
        0
    )

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

    sns.set_theme(style="whitegrid", palette="viridis")
    fig = plt.figure(figsize=(20, 24))
    
    # 1. HISTOGRAMA
    ax1 = plt.subplot(3, 2, 1)
    sns.histplot(df_muni['Tasa_Anual_100k'], bins=50, kde=True, color='crimson', ax=ax1)
    ax1.set_title('1. Distribución de la Tasa Delictiva (Histograma)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tasa de Delitos por 100k Hab.')
    ax1.set_ylabel('Frecuencia')

    # 2. BOXPLOT
    ax2 = plt.subplot(3, 2, 2)
    sns.boxplot(x=df_muni['Tasa_Anual_100k'], color='orange', ax=ax2)
    ax2.set_title('2. Detección de Municipios Atípicos (Boxplot)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tasa de Delitos por 100k Hab.')

    # 3. TOP BIENES JURÍDICOS
    ax3 = plt.subplot(3, 2, 3)
    top_bienes = df.groupby('Bien_juridico_afectado')['Total_Anual'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_bienes.values, y=top_bienes.index, palette='mako', ax=ax3)
    ax3.set_title('3. Top 10 Bienes Jurídicos Más Afectados', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Total de Delitos Anuales')

    # 4. TOP MUNICIPIOS
    ax4 = plt.subplot(3, 2, 4)
    top_municipios = df_muni[df_muni['POB_TOTAL'] > 5000].sort_values('Tasa_Anual_100k', ascending=False).head(10)
    sns.barplot(x='Tasa_Anual_100k', y='Municipio', data=top_municipios, palette='rocket', ax=ax4)
    ax4.set_title('4. Top 10 Municipios con Mayor Riesgo Relativo', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Tasa por 100k Hab.')

    # 5. SCATTER PLOT (Nueva recomendación del experto)
    ax5 = plt.subplot(3, 2, 5)
    df_scatter = df_muni[df_muni['POB_TOTAL'] > 5000] # Filtramos ruido
    sns.scatterplot(x='POB_TOTAL', y='Tasa_Anual_100k', data=df_scatter, alpha=0.5, color='teal', ax=ax5)
    ax5.set_title('5. Relación Población vs Tasa Delictiva', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Población Total')
    ax5.set_ylabel('Tasa por 100k Hab.')
    ax5.text(0.5, -0.15, '¿Más población = Más criminalidad? Observa la dispersión.', ha='center', transform=ax5.transAxes, color='darkred')

    # 6. HEATMAP DE CORRELACIÓN
    ax6 = plt.subplot(3, 2, 6)
    matriz_corr = df_muni[['POB_TOTAL', 'Total_Anual', 'Tasa_Anual_100k']].corr()
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=1, ax=ax6)
    ax6.set_title('6. Matriz de Correlación (Heatmap)', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    return {
        "estadisticas": df_stats,
        "figura_matplotlib": fig
    }