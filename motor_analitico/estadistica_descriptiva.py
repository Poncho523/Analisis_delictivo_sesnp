import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Fix de rutas
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia

def generar_reporte_eda(df: pd.DataFrame) -> dict:
    """
    Genera estadísticas y figuras interactivas de Plotly para el informe narrativo.
    """
    # 1. Agrupación base
    df_muni = df.groupby(['Cve_Municipio', 'Municipio', 'POB_TOTAL'])['Total_Anual'].sum().reset_index()

    df_muni['Tasa_Anual_100k'] = np.where(
        df_muni['POB_TOTAL'] > 0, 
        (df_muni['Total_Anual'] / df_muni['POB_TOTAL']) * 100000, 
        0
    )

    # 2. Tabla de Estadísticas Descriptivas
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
            'Asimetría': data.skew(),
            'Curtosis': data.kurtosis()
        }

    df_stats = pd.DataFrame(resultados).round(2)

    # 3. Construcción de Gráficas Individuales en Plotly

    # A) Histograma
    fig_hist = px.histogram(
        df_muni, x='Tasa_Anual_100k', nbins=50, 
        title="Distribución de la Tasa Delictiva",
        color_discrete_sequence=['#ef553b'],
        labels={'Tasa_Anual_100k': 'Tasa por 100k Hab.'}
    )
    
    # B) Boxplot
    fig_box = px.box(
        df_muni, x='Tasa_Anual_100k', 
        title="Detección de Municipios Atípicos",
        color_discrete_sequence=['#fd8d49'],
        labels={'Tasa_Anual_100k': 'Tasa por 100k Hab.'}
    )

    # C) Top Bienes Jurídicos
    top_bienes = df.groupby('Bien_juridico_afectado')['Total_Anual'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_bienes = px.bar(
        top_bienes, x='Total_Anual', y='Bien_juridico_afectado', orientation='h',
        title="Top 10 Bienes Jurídicos Más Afectados",
        color='Total_Anual', color_continuous_scale='Blues'
    )
    fig_bienes.update_layout(yaxis={'categoryorder':'total ascending'})

    # D) Top Municipios
    top_municipios = df_muni[df_muni['POB_TOTAL'] > 5000].sort_values('Tasa_Anual_100k', ascending=False).head(10)
    fig_munis = px.bar(
        top_municipios, x='Tasa_Anual_100k', y='Municipio', orientation='h',
        title="Top 10 Municipios de Mayor Riesgo Relativo",
        color='Tasa_Anual_100k', color_continuous_scale='Reds'
    )
    fig_munis.update_layout(yaxis={'categoryorder':'total ascending'})

    # E) Scatter (Población vs Tasa)
    df_scatter = df_muni[df_muni['POB_TOTAL'] > 5000]
    fig_scatter = px.scatter(
        df_scatter, x='POB_TOTAL', y='Tasa_Anual_100k', 
        title="Relación Población vs Criminalidad",
        hover_name='Municipio', opacity=0.6, color_discrete_sequence=['#00cc96'],
        labels={'POB_TOTAL': 'Población Total', 'Tasa_Anual_100k': 'Tasa por 100k Hab.'}
    )

    # F) Curva de Lorenz
    delitos_ordenados = np.sort(df_muni['Total_Anual'].values)
    lorenz_curve = np.cumsum(delitos_ordenados) / np.sum(delitos_ordenados)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)
    x_lorenz = np.linspace(0.0, 1.0, lorenz_curve.size)
    
    # Calculo matemático del Gini para el título
    try:
        gini = 1 - 2 * np.trapezoid(y=lorenz_curve, x=x_lorenz)
    except AttributeError:
        gini = 1 - 2 * np.trapz(y=lorenz_curve, x=x_lorenz)

    df_lorenz = pd.DataFrame({'Porcentaje Municipios': x_lorenz * 100, 'Porcentaje Delitos': lorenz_curve * 100})
    fig_lorenz = px.line(
        df_lorenz, x='Porcentaje Municipios', y='Porcentaje Delitos',
        title=f"Curva de Lorenz - Concentración Criminal (Gini: {gini:.3f})",
        color_discrete_sequence=['#ab63fa']
    )
    # Línea de equidad (la diagonal perfecta)
    fig_lorenz.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="gray", dash="dash"))

    # Empaqueto todas las figuras y la tabla para mandarlas a la interfaz
    return {
        "estadisticas": df_stats,
        "graficas": {
            "histograma": fig_hist,
            "boxplot": fig_box,
            "bienes": fig_bienes,
            "municipios": fig_munis,
            "scatter": fig_scatter,
            "lorenz": fig_lorenz
        }
    }