import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Puente Arquitectónico
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_data_mart import cargar_data_mart

def ejecutar_pipeline_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Pipeline profesional: Tasas -> Escala -> PCA -> K-Optimo -> K-Means
    Nota: Recibe directamente un Data Mart pivoteado.
    """
    # 1. Filtramos municipios muy pequeños para evitar tasas infladas
    # Usamos 'Poblacion_Total' basado en la estructura de tu nuevo Data Mart
    df_filtrado = df[df['Poblacion_Total'] >= 5000].copy()
    
    # 2. Separar Metadata de las Características (Features)
    columnas_base = ['Cve_Municipio', 'Municipio', 'Poblacion_Total']
    # Por si el Data Mart llega a tener la Entidad
    if 'Entidad' in df_filtrado.columns:
        columnas_base.append('Entidad')
        
    # Asumimos que todas las demás columnas son delitos
    columnas_delitos = df_filtrado.columns.drop(columnas_base)
    
    # 3. CONVERSIÓN A TASAS
    for delito in columnas_delitos:
        df_filtrado[delito] = (df_filtrado[delito] / df_filtrado['Poblacion_Total']) * 100000

    # Guardamos los IDs por un lado y los números crudos por otro
    df_identificadores = df_filtrado[columnas_base].copy()
    X_numerico = df_filtrado[columnas_delitos]

    # 4. ESCALADO
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_numerico)

    # 5. PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_escalado)
    varianza_retenida = sum(pca.explained_variance_ratio_) * 100

    # 6. EVALUACIÓN CIENTÍFICA (Encontrar el K Óptimo)
    inercias = []
    scores_silueta = []
    rango_k = range(2, 9)
    mejor_k = 2
    mejor_score = -1

    for k in rango_k:
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
        etiquetas_temp = modelo_temp.fit_predict(X_pca)
        inercias.append(modelo_temp.inertia_)
        score = silhouette_score(X_pca, etiquetas_temp)
        scores_silueta.append(score)
        
        # Guardamos el K con la mejor silueta (matemáticamente el más puro)
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            
    df_evaluacion = pd.DataFrame({'K': rango_k, 'Inercia': inercias, 'Silhouette': scores_silueta})

    # 7. K-MEANS DEFINITIVO (Con el K que el usuario pidió en el slider)
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    etiquetas_clusters = kmeans_final.fit_predict(X_pca)
    score_final = silhouette_score(X_pca, etiquetas_clusters)

    # 8. RE-ENSAMBLAJE
    df_final = df_identificadores.copy()
    df_final['Cluster'] = etiquetas_clusters
    
    # 2D PCA para visualización frontal
    df_final['PCA_1'] = X_pca[:, 0]
    df_final['PCA_2'] = X_pca[:, 1]

    # 9. CONTEXTO DEL PERFIL (Población y N)
    resumen_clusters = df_final.groupby('Cluster').agg(
        Total_Municipios=('Cve_Municipio', 'count'),
        Poblacion_Promedio=('Poblacion_Total', 'mean')
    ).reset_index()

    # Creación de etiquetas semánticas ricas
    def crear_etiqueta(fila):
        cluster = int(fila['Cluster'])
        n = int(fila['Total_Municipios'])
        pob = fila['Poblacion_Promedio'] / 1000 # En miles
        return f"Perfil {chr(65 + cluster)} (N={n} | Pob. Media: {pob:.0f}k)"

    resumen_clusters['Nombre_Cluster'] = resumen_clusters.apply(crear_etiqueta, axis=1)
    
    # Pegamos los nombres ricos al dataframe final
    df_final = df_final.merge(resumen_clusters[['Cluster', 'Nombre_Cluster']], on='Cluster')

    # 10. PERFILES PROMEDIO PARA HEATMAP
    df_analisis_perfiles = pd.concat([df_final[['Nombre_Cluster']], X_numerico], axis=1)
    perfiles_promedio = df_analisis_perfiles.groupby('Nombre_Cluster').mean()

    return {
        "df_clusters": df_final,
        "perfiles_promedio": perfiles_promedio,
        "datos_evaluacion": df_evaluacion,
        "metricas": {
            "silhouette_actual": score_final,
            "mejor_k_matematico": mejor_k,
            "mejor_score_matematico": mejor_score,
            "varianza_pca": varianza_retenida,
            "dimensiones_pca": X_pca.shape[1]
        }
    }