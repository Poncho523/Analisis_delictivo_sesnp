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
    Ejecuta el pipeline completo de clustering: Pivoteo -> Tasas -> Escalado -> PCA -> K-Means.
    Retorna un diccionario listo para ser consumido por Streamlit.
    """
    print("1. Iniciando transformación de datos (Pivot)...")
    
    # Nos quedamos con municipios que tengan población significativa para evitar tasas distorsionadas
    df_filtrado = df[df['POB_TOTAL'] >= 5000].copy()
    
    # 1. PIVOT: Filas = Municipios, Columnas = Tipos de Delito
    matriz_delitos = df_filtrado.pivot_table(
        index=['Cve_Municipio', 'Municipio', 'Entidad', 'POB_TOTAL'], 
        columns='Subtipo_de_delito', 
        values='Total_Anual', 
        aggfunc='sum'
    ).fillna(0).reset_index()

    print("2. Convirtiendo valores absolutos a Tasas por 100k habitantes...")
    # Extraemos la lista de todas las columnas que son delitos (las que agregamos en el pivot)
    columnas_delitos = matriz_delitos.columns.drop(['Cve_Municipio', 'Municipio', 'Entidad', 'POB_TOTAL'])
    
    # 2. CONVERSIÓN A TASAS
    for delito in columnas_delitos:
        matriz_delitos[delito] = (matriz_delitos[delito] / matriz_delitos['POB_TOTAL']) * 100000

    # Separamos el "DNI" del municipio (texto) de los "Datos Matemáticos" (números)
    df_identificadores = matriz_delitos[['Cve_Municipio', 'Municipio', 'Entidad', 'POB_TOTAL']]
    X_numerico = matriz_delitos[columnas_delitos]

    print("3. Escalando datos (StandardScaler)...")
    # 3. ESCALADO: Para que un delito de volumen masivo no eclipse a uno grave pero de poco volumen
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X_numerico)

    print("4. Aplicando Análisis de Componentes Principales (PCA)...")
    # 4. PCA: Le pedimos que retenga el 90% de la varianza criminal original
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_escalado)
    varianza_retenida = sum(pca.explained_variance_ratio_) * 100

    print(f"   -> PCA comprimió de {len(columnas_delitos)} delitos a {X_pca.shape[1]} Súper-Componentes.")
    print(f"   -> Varianza retenida: {varianza_retenida:.2f}%")

    # --- DATOS PARA LA CURVA DEL CODO (Evaluación) ---
    print("5. Calculando Curva del Codo para Streamlit...")
    inercias = []
    rango_k = range(2, 11)
    for k in rango_k:
        modelo_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
        modelo_temp.fit(X_pca)
        inercias.append(modelo_temp.inertia_)
        
    df_codo = pd.DataFrame({'K (Número de Clusters)': rango_k, 'Inercia': inercias})

    print(f"6. Entrenando Modelo K-Means Final (K={n_clusters})...")
    # 5. K-MEANS DEFINITIVO
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    etiquetas_clusters = kmeans_final.fit_predict(X_pca)
    
    # Evaluamos la calidad de la agrupación
    score_silueta = silhouette_score(X_pca, etiquetas_clusters)

    print("7. Ensamblando el paquete de inteligencia...")
    # 6. RE-ENSAMBLAJE
    df_final = df_identificadores.copy()
    df_final['Cluster'] = etiquetas_clusters
    # Le ponemos nombres amigables a los clusters (Ej: Grupo A, Grupo B...)
    df_final['Nombre_Cluster'] = df_final['Cluster'].apply(lambda x: f"Perfil {chr(65 + x)}") 
    
    # Agregamos los primeros 3 componentes de PCA para poder hacer una gráfica 3D en la interfaz
    if X_pca.shape[1] >= 3:
        df_final['PCA_1'] = X_pca[:, 0]
        df_final['PCA_2'] = X_pca[:, 1]
        df_final['PCA_3'] = X_pca[:, 2]

    # 7. PERFILAMIENTO: ¿Qué hace a cada cluster especial?
    # Juntamos los clusters con las tasas originales y sacamos el promedio
    df_analisis_perfiles = pd.concat([df_final[['Nombre_Cluster']], X_numerico], axis=1)
    perfiles_promedio = df_analisis_perfiles.groupby('Nombre_Cluster').mean()

    return {
        "df_clusters": df_final,                    # La tabla con los municipios y su cluster asignado
        "perfiles_promedio": perfiles_promedio,     # Los promedios de tasas para saber qué delito domina
        "datos_codo": df_codo,                      # Para que Streamlit dibuje la gráfica de codo
        "metricas": {
            "silhouette": score_silueta,
            "varianza_pca": varianza_retenida,
            "dimensiones_pca": X_pca.shape[1]
        }
    }

if __name__ == "__main__":
    dataset = cargar_datos_incidencia()
    resultado = ejecutar_pipeline_kmeans(dataset, n_clusters=4)
    print("\n=== RESUMEN DE EJECUCIÓN ===")
    print(f"Silhouette Score: {resultado['metricas']['silhouette']:.3f}")
    print("\nMunicipios del Perfil A (Muestra):")
    print(resultado['df_clusters'][resultado['df_clusters']['Nombre_Cluster'] == 'Perfil A'][['Municipio', 'Entidad']].head())