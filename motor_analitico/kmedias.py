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

from ETL.carga_datos import cargar_datos_incidencia

def ejecutar_pipeline_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Pipeline profesional: Tasas -> Escala -> PCA -> K-Optimo -> K-Means
    """
    df_filtrado = df[df['POB_TOTAL'] >= 5000].copy()
    
    matriz_delitos = df_filtrado.pivot_table(
        index=['Cve_Municipio', 'Municipio', 'Entidad', 'POB_TOTAL'], 
        columns='Subtipo_de_delito', 
        values='Total_Anual', 
        aggfunc='sum'
    ).fillna(0).reset_index()

    columnas_delitos = matriz_delitos.columns.drop(['Cve_Municipio', 'Municipio', 'Entidad', 'POB_TOTAL'])
    
    # CONVERSIÓN A TASAS
    for delito in columnas_delitos:
        matriz_delitos[delito] = (matriz_delitos[delito] / matriz_delitos['POB_TOTAL']) * 100000

    df_identificadores = matriz_delitos[['Cve_Municipio', 'Municipio', 'Entidad', 'POB_TOTAL']]
    X_numerico = matriz_delitos[columnas_delitos]

    # ESCALADO
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_numerico)

    # PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_escalado)
    varianza_retenida = sum(pca.explained_variance_ratio_) * 100

    # EVALUACIÓN CIENTÍFICA (Encontrar el K Óptimo)
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

    # K-MEANS DEFINITIVO (Con el K que el usuario pidió en el slider)
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    etiquetas_clusters = kmeans_final.fit_predict(X_pca)
    score_final = silhouette_score(X_pca, etiquetas_clusters)

    # RE-ENSAMBLAJE
    df_final = df_identificadores.copy()
    df_final['Cluster'] = etiquetas_clusters
    
    # 2D PCA (La recomendación del Senior)
    df_final['PCA_1'] = X_pca[:, 0]
    df_final['PCA_2'] = X_pca[:, 1]

    # CONTEXTO DEL PERFIL (Población y N)
    resumen_clusters = df_final.groupby('Cluster').agg(
        Total_Municipios=('Cve_Municipio', 'count'),
        Poblacion_Promedio=('POB_TOTAL', 'mean')
    ).reset_index()

    # Creación de etiquetas semánticas ricas
    def crear_etiqueta(fila):
        cluster = fila['Cluster']
        n = fila['Total_Municipios']
        pob = fila['Poblacion_Promedio'] / 1000 # En miles
        return f"Perfil {chr(65 + cluster)} (N={n} | Pob. Media: {pob:.0f}k)"

    resumen_clusters['Nombre_Cluster'] = resumen_clusters.apply(crear_etiqueta, axis=1)
    
    # Pegamos los nombres ricos al dataframe final
    df_final = df_final.merge(resumen_clusters[['Cluster', 'Nombre_Cluster']], on='Cluster')

    # PERFILES PROMEDIO PARA HEATMAP
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