import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go 

# Setup básico. 'wide' es indispensable, si lo dejo por defecto las tablas de Pandas
# se aplastan horrible en monitores anchos y el usuario tiene que hacer scroll horizontal.
st.set_page_config(page_title="Gestión de Incidentes SESNSP", layout="wide")

# Hack asqueroso pero necesario de Python para encontrar las carpetas hermanas
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from motor_analitico.parapeto_concentracion import calcular_pareto_municipios
from motor_analitico.consultador_con_POO import transformar_dataframe_a_objetos
from motor_analitico.chi2 import calcular_dependencia_demografica
from motor_analitico.estadistica_descriptiva import generar_reporte_eda 
from motor_analitico.kmedias import ejecutar_pipeline_kmeans 
from ETL.carga_data_mart import cargar_data_mart 

# @st.cache_data salva vidas. Evita que lea los 300MB del CSV cada vez que mueven algo.
@st.cache_data
def obtener_datos_cacheados():
    return cargar_datos_incidencia()

@st.cache_data
def obtener_data_mart_cacheado():
    return cargar_data_mart()

dataset_global = obtener_datos_cacheados()
data_mart_global = obtener_data_mart_cacheado() 


def mostrar_pantalla_inicio(df: pd.DataFrame):
    st.title("Sistema de Análisis Delictivo Nacional")
    st.markdown("Plataforma de inteligencia criminal basada en datos del SESNSP.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Incidentes", f"{len(df):,}") 
    col2.metric("Total de Municipios", f"{df['Cve_Municipio'].nunique():,}")
    col3.metric("Municipios Críticos", "1,171", "Pareto 80%")
    col4.metric("Estado de Alerta", "Activo", "Megalópolis", delta_color="inverse")
    
    st.divider()

    st.subheader("Hallazgos Nacionales (Executive Summary)")
    st.info("""
    * **Concentración (Pareto):** El 47% de los municipios concentran el 80% de la incidencia delictiva del país.
    * **Dominancia Patrimonial:** El delito contra el patrimonio (robos, fraudes) representa el núcleo principal del crimen a nivel nacional en entornos urbanos.
    * **Perfiles Criminales (K-Means):** Se identificaron tipologías claramente diferenciadas, separando los focos de violencia interpersonal de las zonas de robo generalizado.
    * **Influencia Demográfica (Chi-Cuadrada):** Existe una comprobación estadística de que la demografía urbana/rural dicta fuertemente el perfil del crimen.
    """)


def mostrar_analisis_exploratorio(df: pd.DataFrame):
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    reporte = generar_reporte_eda(df)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(reporte["estadisticas"], use_container_width=True)
    st.divider()

    st.subheader("Distribución de la Criminalidad")
    st.plotly_chart(reporte["graficas"]["histograma"], use_container_width=True)
    st.info(" La distribución presenta una fuerte asimetría positiva, indicando que la mayoría de municipios tienen tasas relativamente bajas mientras un pequeño grupo concentra niveles excepcionalmente altos de criminalidad.")
    st.divider()

    st.subheader("Municipios Atípicos")
    st.plotly_chart(reporte["graficas"]["boxplot"], use_container_width=True)
    st.info(" La presencia de numerosos valores atípicos sugiere la existencia de focos rojos que requieren análisis y política pública diferenciada.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bienes Jurídicos Más Afectados")
        st.plotly_chart(reporte["graficas"]["bienes"], use_container_width=True)
        st.info(" Los delitos patrimoniales representan la principal carga criminal observada en el país.")
    with col2:
        st.subheader("Municipios de Mayor Riesgo")
        st.plotly_chart(reporte["graficas"]["municipios"], use_container_width=True)
        st.info(" Estos municipios presentan las mayores tasas relativas y constituyen prioridades para la asignación de recursos.")
    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Población vs Criminalidad")
        st.plotly_chart(reporte["graficas"]["scatter"], use_container_width=True)
        st.info(" No se observa una relación lineal fuerte entre tamaño poblacional absoluto y la tasa delictiva per cápita.")
    with col4:
        st.subheader("Concentración Criminal")
        st.plotly_chart(reporte["graficas"]["lorenz"], use_container_width=True)
        st.info(" Una proporción muy reducida de municipios concentra una gran parte de los delitos registrados a nivel nacional.")


def mostrar_clustering(df: pd.DataFrame):
    st.title("Perfilamiento Criminal Avanzado (K-Means)")
    st.markdown("Segmentación de municipios basada en la similitud de sus tasas delictivas mediante Inteligencia Artificial.")
    
    with st.expander("Justificación Metodológica: ¿Por qué K-Means?"):
        st.write("""
        **Objetivo:** Descubrir la estructura subyacente y tipologías criminales generales del país.
        * Se eligió **K-Means** por su capacidad robusta para agrupar variables continuas (tasas) y segmentar todos los municipios en perfiles interpretables mediante el análisis de sus centroides. Esto permite generalizar patrones para el diseño de políticas públicas.
        """)

    resultado_previo = ejecutar_pipeline_kmeans(df, n_clusters=4)
    mejor_k = resultado_previo['metricas']['mejor_k_matematico']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parámetros del Modelo")
    st.sidebar.info(f"Recomendación Algorítmica: Silhouette sugiere K = {mejor_k}.")
    
    k_elegido = st.sidebar.slider("Forzar Número de Perfiles (K)", min_value=2, max_value=8, value=int(mejor_k))
    
    resultado = ejecutar_pipeline_kmeans(df, n_clusters=k_elegido)
    
    st.subheader("1. Diagnóstico del Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dimensiones Retenidas por PCA", f"{resultado['metricas']['dimensiones_pca']}")
    col2.metric("Varianza Retenida", f"{resultado['metricas']['varianza_pca']:.1f}%")
    col3.metric(f"Silhouette Score (K={k_elegido})", f"{resultado['metricas']['silhouette_actual']:.3f}")
    
    st.divider()
    
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.subheader("2. Mapa de Similitud Criminal (PCA 2D)")
        df_plot = resultado['df_clusters']
        
        hover_cols = ['Poblacion_Total']
        if 'Entidad' in df_plot.columns:
            hover_cols.append('Entidad')
            
        fig_2d = px.scatter(
            df_plot, x='PCA_1', y='PCA_2',
            color='Nombre_Cluster',
            hover_name='Municipio', hover_data=hover_cols,
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_2d.update_xaxes(showticklabels=False)
        fig_2d.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_2d, use_container_width=True)
            
    with col_g2:
        st.subheader("Validación Matemática")
        df_eval = resultado['datos_evaluacion'].set_index('K')
        st.line_chart(df_eval['Silhouette'], color="#ff4b4b")
        
    st.divider()
    
    st.subheader("3. ADN Criminal de los Perfiles")
    
    df_perfiles = resultado['perfiles_promedio'].T
    
    st.markdown("**Interpretación Automática de Perfiles:**")
    for cluster_name in df_perfiles.columns:
        top_3 = df_perfiles[cluster_name].nlargest(3).index.tolist()
        top_3_limpio = [str(delito).replace('_', ' ').title() for delito in top_3]
        st.markdown(f"- **{cluster_name}:** Predominan *{top_3_limpio[0]}*, *{top_3_limpio[1]}* y *{top_3_limpio[2]}*.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    top_delitos = df_perfiles.sum(axis=1).sort_values(ascending=False).head(10).index
    df_perfiles_top = df_perfiles.loc[top_delitos]
    
    fig_heat = px.imshow(
        df_perfiles_top,
        text_auto='.0f', 
        aspect="auto",   
        color_continuous_scale='Reds', 
        labels=dict(x="Caracterización del Perfil", y="Tipo de Delito", color="Tasa 100k")
    )
    fig_heat.update_xaxes(side="top") 
    st.plotly_chart(fig_heat, use_container_width=True)


def mostrar_patrones_delictivos(df: pd.DataFrame):
    st.title("Análisis de Patrones (Ley de Pareto)")
    st.info("Ejecutando motor analítico en tiempo real...")
    
    resultado_pareto = calcular_pareto_municipios(df)
    df_pareto = resultado_pareto["datos_grafica"]
    
    st.subheader("La Historia del 80/20 en el Crimen Nacional")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total de Municipios", f"{resultado_pareto['total_municipios']:,}")
    kpi2.metric("Focos Rojos (80%)", f"{resultado_pareto['cantidad_critica']:,}", delta_color="inverse")
    kpi3.metric("% del Territorio", f"{resultado_pareto['porcentaje_territorio']:.1f}%")
    kpi4.metric("Total de Delitos (100%)", f"{resultado_pareto['total_delitos_nacional']:,.0f}")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.subheader("Curva de Concentración Acumulada")
        
        df_grafica = df_pareto.copy()
        df_grafica['Ranking de Gravedad'] = range(1, len(df_grafica) + 1)
        
        fig_pareto = px.line(
            df_grafica, x='Ranking de Gravedad', y='Porcentaje_Acumulado',
            labels={"Porcentaje_Acumulado": "% de Delitos Acumulados", "Ranking de Gravedad": "Municipios (De Peor a Mejor)"}
        )
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Umbral Crítico (80%)")
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    with col2:
        st.subheader("Top Focos Rojos Absolutos")
        columnas_ver = ['Municipio', 'Entidad', 'Total_Anual']
        df_mostrar = df_pareto[columnas_ver].head(10).copy()
        
        df_mostrar['Total_Anual'] = df_mostrar['Total_Anual'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_mostrar, hide_index=True)


def mostrar_tabla_datos(df: pd.DataFrame):
    st.title("Explorador de Inteligencia (Uso de POO)")
    
    # Hardcode a 100k para la muestra para no tronar el navegador.
    TAMANO_MUESTRA = 100000
    if len(df) > TAMANO_MUESTRA:
        df_muestra = df.sample(n=TAMANO_MUESTRA, random_state=42)
        st.caption(f"Operando sobre muestra representativa de {TAMANO_MUESTRA:,} registros.")
    else:
        df_muestra = df

    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        entidades = ["Todas"] + list(df_muestra["Entidad"].unique())
        estado_sel = st.selectbox("1. Entidad Federativa:", entidades)
        
    with col_f2:
        if estado_sel == "Todas":
            municipios = ["Todos"]
        else:
            municipios = ["Todos"] + list(df_muestra[df_muestra["Entidad"] == estado_sel]["Municipio"].unique())
        mun_sel = st.selectbox("2. Municipio:", municipios)
        
    with col_f3:
        bienes = ["Todos"] + list(df_muestra["Bien_juridico_afectado"].unique())
        bien_sel = st.selectbox("3. Bien Jurídico:", bienes)

    df_filtrado = df_muestra.copy()
    if estado_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado["Entidad"] == estado_sel]
    if mun_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Municipio"] == mun_sel]
    if bien_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Bien_juridico_afectado"] == bien_sel]
    
    if len(df_filtrado) == 0:
        st.warning("No se encontraron registros.")
        return
    
    lista_registros_poo = transformar_dataframe_a_objetos(df_filtrado)
    
    st.subheader("Resumen Global")
    
    conteo_riesgos = {"Alto": 0, "Medio": 0, "Bajo": 0}
    lista_meses = []
    
    for reg in lista_registros_poo:
        riesgo = reg.categorizar_nivel_riesgo(umbral_alto=50.0, umbral_medio=20.0)
        conteo_riesgos[riesgo] += 1
        
        mes_obj = reg.obtener_mes_moda()
        if mes_obj and mes_obj.cantidadCasos > 0:
            lista_meses.append(mes_obj.mes.name)
            
    mes_predominante = Counter(lista_meses).most_common(1)[0][0] if lista_meses else "Sin incidentes"
    
    cr1, cr2, cr3, cr4 = st.columns(4)
    cr1.metric("Registros Procesados", f"{len(lista_registros_poo):,}")
    cr2.metric("Municipios Riesgo Alto", f"{conteo_riesgos['Alto']:,}")
    cr3.metric("Municipios Riesgo Medio", f"{conteo_riesgos['Medio']:,}")
    cr4.metric("Mes Crítico Global", mes_predominante)
    
    st.divider()
    
    # Solo muestro los primeros 50 objetos
    for registro in lista_registros_poo[:50]:
        titulo_caja = f"{registro.municipio.nombre} | {registro.clasificacion.subtipoDelito} ({registro.clasificacion.modalidad})"
        
        with st.expander(titulo_caja):
            c1, c2, c3 = st.columns(3)
            
            c1.markdown("**Taxonomía Penal**")
            c1.write(f"**Subtipo:** {registro.clasificacion.subtipoDelito}")
            c1.write(f"**Prioridad de Atención:** Nivel {registro.clasificacion.calcular_peso_estadistico()}")
            c1.write(f"**Índice de Urgencia:** {registro.calcular_indice_urgencia():.2f} pts")

            c2.markdown("**Contexto Demográfico**")
            c2.write(f"**Asentamiento:** {registro.municipio.clasificar_asentamiento(df_filtrado.iloc[0]['POB_TOTAL'])}")
            c2.write(f"**Tasa 100k:** {registro.tasaAnual100k:.2f}")
            riesgo = registro.categorizar_nivel_riesgo(umbral_alto=50.0, umbral_medio=20.0)
            c2.write(f"**Nivel de Riesgo:** {riesgo}")
            
            mes_moda = registro.obtener_mes_moda()
            c3.markdown("**Alerta Temporal**")
            
            if mes_moda and mes_moda.cantidadCasos > 0:
                c3.write(f"**Mes Crítico:** {mes_moda.mes.name} ({mes_moda.cantidadCasos} casos)")
            else:
                c3.write("**Mes Crítico:** Ninguno (0 casos)")
                
            c3.write(f"**Tendencia:** {registro.calcular_tendencia_semestral()}")
            c3.write(f"**Patrón:** {registro.determinar_patron_ocurrencia()}")


def mostrar_analisis_demografico(df: pd.DataFrame):
    st.title("Dependencia Demográfica (Chi-Cuadrada)")
    st.markdown("Analiza estadísticamente si el entorno dicta el tipo de crimen.")
        
    resultado = calcular_dependencia_demografica(df)
    
    st.subheader("1. Resultados de aplicar CHI2")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # --- AQUÍ INYECTAMOS LA PRUEBA REINA (P-VALOR) ---
        p_val = resultado['diagnostico']['p_value']
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        
        if resultado['diagnostico']['existe_relacion']:
            st.success(f" Sí existe una relación estadística. El tipo de entorno modifica el perfil criminal.")
            st.markdown(f"**Significancia (P-Valor):** {p_str} *(Como es menor a 0.05, podemos decir que existe evidencia estadística suficiente para rechazar H₀.)*.")
        else:
            st.error(f"**Sin Relación:** El entorno no dicta el tipo de crimen (P-Valor: {p_str}).")
            
        st.markdown(f"**Fuerza de la Relación:** {resultado['diagnostico']['fuerza_relacion']}")
        
    with col2:
        valor_cramer = resultado['diagnostico']['valor_cramer']
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = valor_cramer,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fuerza de Relación (V de Cramér)"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "rgba(255, 255, 255, 0)"},
                'steps' : [
                    {'range': [0, 0.1], 'color': "#a8e6cf", 'name': 'Débil'},      
                    {'range': [0.1, 0.3], 'color': "#ffd3b6", 'name': 'Moderada'}, 
                    {'range': [0.3, 1.0], 'color': "#ff8b94", 'name': 'Fuerte'}    
                ],
                'threshold' : {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': valor_cramer
                }
            }
        ))
        
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.divider()
    
    st.subheader("2. Perfil Criminal por Zona")
    
    df_porcentajes = resultado['perfil_criminal_porcentajes']
    
    fig = px.imshow(
        df_porcentajes,
        text_auto='.1f', 
        aspect="auto",   
        color_continuous_scale='YlOrRd', 
        labels=dict(x="Tipo de Asentamiento", y="Bien Jurídico Afectado", color="% de Concentración")
    )
    
    fig.update_xaxes(side="top") 
    st.plotly_chart(fig, use_container_width=True)

# ----------------- SIDEBAR Y ROUTING -----------------
st.sidebar.title("Menú")

opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ("Inicio", "Análisis Exploratorio (EDA)", "Clustering (K-Means)", "Consultar Datos", "Patrones y Pareto", "Análisis Demográfico")
)

if opcion == "Inicio":
    mostrar_pantalla_inicio(dataset_global)
elif opcion == "Análisis Exploratorio (EDA)":
    mostrar_analisis_exploratorio(dataset_global)
elif opcion == "Clustering (K-Means)":
    mostrar_clustering(data_mart_global)
elif opcion == "Consultar Datos":
    mostrar_tabla_datos(dataset_global)
elif opcion == "Patrones y Pareto":
    mostrar_patrones_delictivos(dataset_global)
elif opcion == "Análisis Demográfico":
    mostrar_analisis_demografico(dataset_global)