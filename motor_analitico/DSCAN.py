import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Puente Arquitectónico
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_data_mart import cargar_data_mart

def ejecutar_pipeline_dbscan(df: pd.DataFrame, eps: float = 1.5, min_samples: int = 5) -> dict:
    """
    Pipeline complementario: Tasas -> Escala -> PCA -> DBSCAN (Detección de Ruido)
    """
    # 1. Filtramos municipios muy pequeños (Misma regla que K-Means)
    df_filtrado = df[df['Poblacion_Total'] >= 5000].copy()
    
    # 2. Separar Metadata
    columnas_base = ['Cve_Municipio', 'Municipio', 'Poblacion_Total']
    if 'Entidad' in df_filtrado.columns:
        columnas_base.append('Entidad')
        
    columnas_trampa = ['Tasa_Global_100k', 'Total_Delitos_Absoluto', 'Robos_Totales', 'Total']
    columnas_delitos = [col for col in df_filtrado.columns if col not in columnas_base and col not in columnas_trampa]
    
    # 3. CONVERSIÓN A TASAS
    for delito in columnas_delitos:
        df_filtrado[delito] = (df_filtrado[delito] / df_filtrado['Poblacion_Total']) * 100000

    df_identificadores = df_filtrado[columnas_base].copy()
    X_numerico = df_filtrado[columnas_delitos].copy()

    # 4. ESCALADO
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_numerico)

    # 5. PCA (Retenemos 90% de varianza, igual que K-Means)
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_escalado)
    varianza_retenida = sum(pca.explained_variance_ratio_) * 100
    n_componentes = X_pca.shape[1]

    # --- AQUÍ INICIA LA MAGIA DE DBSCAN ---
    
    # 6. EJECUCIÓN DE DBSCAN
    # Nota: Si el usuario no manda min_samples, una regla de oro es 2 * dimensiones de PCA
    if min_samples == 5: # Si es el default, lo ajustamos inteligentemente a las dimensiones
        min_samples = max(4, n_componentes * 2)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    etiquetas_dbscan = dbscan.fit_predict(X_pca)

    # 7. RE-ENSAMBLAJE
    df_final = pd.concat([df_identificadores, X_numerico], axis=1)
    df_final['Cluster_DBSCAN'] = etiquetas_dbscan
    df_final['PCA_1'] = X_pca[:, 0]
    df_final['PCA_2'] = X_pca[:, 1]

    # 8. SEPARACIÓN DE LA INTELIGENCIA (Ruido vs Clústeres Naturales)
    df_outliers = df_final[df_final['Cluster_DBSCAN'] == -1].copy()
    df_clusters_validos = df_final[df_final['Cluster_DBSCAN'] != -1].copy()
    
    num_outliers = len(df_outliers)
    total_municipios = len(df_final)
    porcentaje_ruido = (num_outliers / total_municipios) * 100
    
    # Cantidad de clústeres reales (excluyendo el -1)
    num_clusters = len(set(etiquetas_dbscan)) - (1 if -1 in etiquetas_dbscan else 0)

    # 9. RESUMEN DE CLÚSTERES (Para la interfaz)
    if num_clusters > 0:
        resumen_clusters = df_clusters_validos.groupby('Cluster_DBSCAN').agg(
            Total_Municipios=('Cve_Municipio', 'count'),
            Poblacion_Promedio=('Poblacion_Total', 'mean')
        ).reset_index()
        resumen_clusters['Tipo'] = 'Agrupación Natural'
    else:
        resumen_clusters = pd.DataFrame(columns=['Cluster_DBSCAN', 'Total_Municipios', 'Poblacion_Promedio', 'Tipo'])

    # 10. EVALUAR ESPECIALIDAD DE LOS OUTLIERS (Z-Score)
    # Para decirle al usuario *por qué* ese municipio es un outlier
    media_global = X_numerico.mean()
    std_global = X_numerico.std()
    
    if num_outliers > 0:
        # Comparamos cada outlier contra la media nacional
        tasas_outliers = df_outliers[columnas_delitos]
        z_scores_outliers = (tasas_outliers - media_global) / (std_global + 1e-9)
        # Encontramos el delito con el pico más alto para cada outlier
        delito_anomalo = z_scores_outliers.idxmax(axis=1)
        df_outliers['Delito_Anomalo_Principal'] = delito_anomalo.str.replace('_', ' ').str.title()
        
        # Limpiar dataframe de outliers para mostrar en tabla
        df_outliers = df_outliers[['Cve_Municipio', 'Municipio', 'Entidad', 'Poblacion_Total', 'Delito_Anomalo_Principal']]

    return {
        "df_completo": df_final,
        "df_outliers": df_outliers.sort_values(by='Poblacion_Total', ascending=False),
        "resumen_clusters": resumen_clusters,
        "metricas": {
            "num_clusters_encontrados": num_clusters,
            "num_outliers": num_outliers,
            "porcentaje_ruido": round(porcentaje_ruido, 2),
            "eps_utilizado": eps,
            "min_samples_utilizado": min_samples,
            "dimensiones_pca": n_componentes,
            "varianza_pca": round(varianza_retenida, 2)
        }
    }

if __name__ == "__main__":
    dataset = cargar_data_mart()
    resultados = ejecutar_pipeline_dbscan(dataset, eps=2.0) # eps de 2.0 suele funcionar bien post-PCA escalado
    
    print("\n--- RESULTADOS DBSCAN ---")
    print(f"Clústeres Naturales: {resultados['metricas']['num_clusters_encontrados']}")
    print(f"Municipios Atípicos (Ruido): {resultados['metricas']['num_outliers']} ({resultados['metricas']['porcentaje_ruido']}%)")
    print("\nTop 5 Municipios Más Extraños (Outliers):")
    print(resultados['df_outliers'].head(5))