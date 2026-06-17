import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # Para una salida limpia en consola

df = pd.read_csv('delitos_nivelados_2015.csv')

# Datos Generales
print("\n DATOS GENERALES")
print(f"Total de registros (filas en bruto): {len(df):,}")
print(f"Estados analizados: {df['Entidad'].nunique()}")
print(f"Municipios analizados: {df['Cve_Municipio'].nunique()}")
print(f"Bienes jurídicos afectados: {df['Bien_juridico_afectado'].nunique()}")
print(f"Tipos de delito: {df['Tipo_de_delito'].nunique()}")

# Para analizar tasas y outliers criminológicos, necesitamos comprimir los 2M de registros a nivel de Municipio 1 fila x municipio sumando todos sus delitos
df_muni = df.groupby(['Cve_Municipio', 'Municipio', 'POB_TOTAL'])['Total_Anual'].sum().reset_index()

# Calculamos la Tasa Global por Municipio
df_muni['Tasa_Anual_100k'] = np.where(
    df_muni['POB_TOTAL'] > 0, 
    (df_muni['Total_Anual'] / df_muni['POB_TOTAL']) * 100000, 
    0
)

#estadistica descriptiva a nivel municipal
print("\nESTADÍSTICA DESCRIPTIVA")
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
print(df_stats)

print("\nINTERPRETACIÓN CRIMINOLÓGICA:")
print("- Diferencia Media vs Mediana: Si la media es mucho mayor a la mediana, hay concentración extrema en pocos municipios.")
print("- Asimetría positiva alta: Confirma que la mayoría de municipios son pacíficos y una minoría concentra el caos.")
print("- Curtosis alta: Confirma la existencia de valores atípicos severos (Outliers). ¡DBSCAN será ideal aquí!")

#Graficas
sns.set_theme(style="whitegrid", palette="viridis")

fig = plt.figure(figsize=(20, 24))
fig.suptitle('MÓDULO EDA: ANÁLISIS DE INCIDENCIA DELICTIVA Y RIESGO MUNICIPAL', fontsize=22, fontweight='bold', y=0.98)

# HISTOGRAMA DE TASA (para la distribución)
ax1 = plt.subplot(3, 2, 1)
sns.histplot(df_muni['Tasa_Anual_100k'], bins=50, kde=True, color='crimson', ax=ax1)
ax1.set_title('1. Distribución de la Tasa Delictiva (Histograma)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Tasa de Delitos por 100k Hab.')
ax1.set_ylabel('Frecuencia (N° de Municipios)')
ax1.text(0.5, -0.15, '¿La criminalidad es uniforme? R: No. Se agrupa a la izquierda (baja incidencia).', ha='center', transform=ax1.transAxes, color='darkred')

#BOXPLOT DE TASA (detección de Outliers para DBSCAN)
ax2 = plt.subplot(3, 2, 2)
sns.boxplot(x=df_muni['Tasa_Anual_100k'], color='orange', ax=ax2)
ax2.set_title('2. Detección de Municipios Atípicos (Boxplot)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tasa de Delitos por 100k Hab.')
ax2.text(0.5, -0.15, 'Los puntos a la derecha son outliers. Requieren análisis focalizado (DBSCAN).', ha='center', transform=ax2.transAxes, color='darkred')

# 10 BIENES JURÍDICOS (qué se afecta más)
ax3 = plt.subplot(3, 2, 3)
top_bienes = df.groupby('Bien_juridico_afectado')['Total_Anual'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_bienes.values, y=top_bienes.index, palette='mako', ax=ax3)
ax3.set_title('3. Top 10 Bienes Jurídicos Más Afectados', fontsize=14, fontweight='bold')
ax3.set_xlabel('Total de Delitos Anuales')
ax3.set_ylabel('')

#10 MUNICIPIOS POR TASA (localización de riesgo relativo)
ax4 = plt.subplot(3, 2, 4)
top_municipios = df_muni[df_muni['POB_TOTAL'] > 5000].sort_values('Tasa_Anual_100k', ascending=False).head(10) # Filtro de +5000 habs para evitar ruido de pueblos muy chicos
sns.barplot(x='Tasa_Anual_100k', y='Municipio', data=top_municipios, palette='rocket', ax=ax4)
ax4.set_title('4. Top 10 Municipios con Mayor Riesgo Criminal Relativo', fontsize=14, fontweight='bold')
ax4.set_xlabel('Tasa por 100k Hab.')
ax4.set_ylabel('Municipio')

#CURVA DE LORENZ (Justicicación para principio de Pareto)
ax5 = plt.subplot(3, 2, 5)
delitos_ordenados = np.sort(df_muni['Total_Anual'].values)
lorenz_curve = np.cumsum(delitos_ordenados) / np.sum(delitos_ordenados)
lorenz_curve = np.insert(lorenz_curve, 0, 0)
x_lorenz = np.linspace(0.0, 1.0, lorenz_curve.size)

#hacemos q el error por versiones de numpy se controlen
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
ax5.text(0.5, -0.15, 'Mientras más se aleja la curva curva, mayor es la concentración (Pareto).', ha='center', transform=ax5.transAxes, color='darkred')

# --- HEATMAP DE CORRELACIÓN (K-Means) ---
ax6 = plt.subplot(3, 2, 6)
matriz_corr = df_muni[['POB_TOTAL', 'Total_Anual', 'Tasa_Anual_100k']].corr()
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=1, ax=ax6)
ax6.set_title('6. Matriz de Correlación (Heatmap)', fontsize=14, fontweight='bold')
ax6.text(0.5, -0.15, '¿La población determina la tasa? Observa la correlación POB_TOTAL vs Tasa.', ha='center', transform=ax6.transAxes, color='darkred')


plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.subplots_adjust(hspace=0.3)
nombre_imagen = 'Dashboard_EDA_Criminologico.png'
plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')