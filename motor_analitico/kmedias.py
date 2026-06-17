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
    """
    # 1. Filtramos municipios muy pequeños
    df_filtrado = df[df['Poblacion_Total'] >= 5000].copy()
    
    # 2. Separar Metadata
    columnas_base = ['Cve_Municipio', 'Municipio', 'Poblacion_Total']
    if 'Entidad' in df_filtrado.columns:
        columnas_base.append('Entidad')
        
    # Identificamos las columnas que son "Totales" o "Agregados" para que no arruinen el clustering
    columnas_trampa = ['Tasa_Global_100k', 'Total_Delitos_Absoluto', 'Robos_Totales', 'Total']
    
    # Nos quedamos estrictamente con los delitos específicos
    columnas_delitos = [col for col in df_filtrado.columns if col not in columnas_base and col not in columnas_trampa]
    
    # 3. CONVERSIÓN A TASAS
    for delito in columnas_delitos:
        df_filtrado[delito] = (df_filtrado[delito] / df_filtrado['Poblacion_Total']) * 100000

    df_identificadores = df_filtrado[columnas_base].copy()
    X_numerico = df_filtrado[columnas_delitos].copy()

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
        
        if score > mejor_score:
            mejor_score = score
            mejor_k = k
            
    df_evaluacion = pd.DataFrame({'K': rango_k, 'Inercia': inercias, 'Silhouette': scores_silueta})

    # 7. K-MEANS DEFINITIVO
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    etiquetas_clusters = kmeans_final.fit_predict(X_pca)
    score_final = silhouette_score(X_pca, etiquetas_clusters)

    # 8. RE-ENSAMBLAJE
    # --- EL FIX ESTÁ AQUÍ ---
    # Unimos los identificadores y las tasas (X_numerico) ANTES de meter el cluster
    df_final = pd.concat([df_identificadores, X_numerico], axis=1)
    df_final['Cluster'] = etiquetas_clusters
    
    df_final['PCA_1'] = X_pca[:, 0]
    df_final['PCA_2'] = X_pca[:, 1]

    # 9. CONTEXTO DEL PERFIL Y ETIQUETADO INTELIGENTE (Z-SCORES)
    resumen_clusters = df_final.groupby('Cluster').agg(
        Total_Municipios=('Cve_Municipio', 'count'),
        Poblacion_Promedio=('Poblacion_Total', 'mean')
    ).reset_index()

    # Calculamos la especialidad criminal de cada cluster usando estadística (Z-Score)
    # Ahora df_final sí tiene las columnas_delitos, así que ya no tronará
    centroides_crudos = df_final.groupby('Cluster')[columnas_delitos].mean()
    media_global = X_numerico.mean()
    std_global = X_numerico.std()
    
    # Z-Score: Comparamos el cluster contra el promedio nacional
    z_scores = (centroides_crudos - media_global) / (std_global + 1e-9)
    especialidad_cluster = z_scores.idxmax(axis=1)

    def crear_etiqueta(fila):
        # int() arregla el error de Numpy que tuvimos antes
        cluster = int(fila['Cluster'])
        n = int(fila['Total_Municipios'])
        pob = fila['Poblacion_Promedio'] / 1000 
        
        delito_destacado = especialidad_cluster[cluster]
        delito_limpio = str(delito_destacado).replace('_', ' ').title()
        
        # Si el Z-Score es negativo, significa que es un municipio súper seguro
        if z_scores.loc[cluster, delito_destacado] < 0:
            return f"Perfil Pacifico (N={n} | Pob: {pob:.0f}k)"
            
        return f"Foco: {delito_limpio} (N={n} | Pob: {pob:.0f}k)"

    resumen_clusters['Nombre_Cluster'] = resumen_clusters.apply(crear_etiqueta, axis=1)
    
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