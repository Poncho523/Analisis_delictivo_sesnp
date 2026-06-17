import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go 

# Setup básico. 'wide' es indispensables
st.set_page_config(page_title="Gestión de Incidentes SESNSP", layout="wide")

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

# Mis imports del backend
from ETL.carga_datos import cargar_datos_incidencia
from motor_analitico.parapeto_concentracion import calcular_pareto_municipios
from motor_analitico.consultador_con_POO import transformar_dataframe_a_objetos
from motor_analitico.chi2 import calcular_dependencia_demografica
from motor_analitico.estadistica_descriptiva import generar_reporte_eda 
from motor_analitico.kmedias import ejecutar_pipeline_kmeans 
from ETL.carga_data_mart import cargar_data_mart 

# OJO: @st.cache_data salva vidas. Si no lo pongo, cada vez que el usuario muvea algo del menu tendra que cargar todo
@st.cache_data
def obtener_datos_cacheados():
    return cargar_datos_incidencia()

# Caché separado para el Data Mart de Andy.
@st.cache_data
def obtener_data_mart_cacheado():
    return cargar_data_mart()

dataset_global = obtener_datos_cacheados()
data_mart_global = obtener_data_mart_cacheado() 


def mostrar_pantalla_inicio(df: pd.DataFrame):
    st.title("Sistema de Análisis Delictivo Nacional")
    st.markdown("Plataforma de inteligencia criminal basada en datos del SESNSP.")
    
    # KPIs principales arriba
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
    st.markdown("Radiografía estadística global de la criminalidad a nivel municipal.")
    
    reporte = generar_reporte_eda(df)
    
    st.subheader("Estadística Descriptiva Municipal")
    st.dataframe(reporte["estadisticas"], use_container_width=True)
    st.divider()
    st.subheader("Diagnóstico Visual (Grid)")
    
    st.pyplot(reporte["figura_matplotlib"])


def mostrar_clustering(df: pd.DataFrame):
    st.title("Perfilamiento Criminal Avanzado (K-Means)")
    st.markdown("Segmentación de municipios basada en la similitud de sus tasas delictivas mediante Inteligencia Artificial.")
    
    # para que no estorbe visualmente, pero ahí está por si un profe lo pide.
    with st.expander("Justificación Metodológica: ¿Por qué K-Means y no DBSCAN?"):
        st.write("""
        **Objetivo:** Descubrir tipologías criminales generales (Patrones base).
        * **K-Means** permite segmentar todos los municipios en grupos interpretables mediante centroides.
        * **DBSCAN** se utilizará en fases posteriores para detectar *outliers* o municipios atípicos que no pertenecen a ningún patrón.
        """)

    # Corro el modelo en silencio (en default 4) solo para extraer cuál sería el K perfecto
    resultado_previo = ejecutar_pipeline_kmeans(df, n_clusters=4)
    mejor_k = resultado_previo['metricas']['mejor_k_matematico']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parámetros del Modelo")
    st.sidebar.info(f"Recomendación Algorítmica: Silhouette sugiere K = {mejor_k}.")
    
    # El slider para forzar el modelo. Le pongo el valor default igual al mejor_k matemático.
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
        # Limpieza UX: Quito los números del eje X y Y. Son valores de PCA que no significan nada 
        fig_2d.update_xaxes(showticklabels=False)
        fig_2d.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_2d, use_container_width=True)
            
    with col_g2:
        st.subheader("Validación Matemática")
        df_eval = resultado['datos_evaluacion'].set_index('K')
        st.line_chart(df_eval['Silhouette'], color="#ff4b4b")
        
    st.divider()
    
    st.subheader("3. ADN Criminal de los Perfiles")
    
    # Transpongo (.T) para que los delitos queden en las filas y los clusters en las columnas
    df_perfiles = resultado['perfiles_promedio'].T
    
    st.markdown("**Interpretación Automática de Perfiles:**")
    # Loop para sacar el "Top 3" de delitos de cada cluster y escribirlo como texto para el usuario.
    for cluster_name in df_perfiles.columns:
        top_3 = df_perfiles[cluster_name].nlargest(3).index.tolist()
        top_3_limpio = [str(delito).replace('_', ' ').title() for delito in top_3]
        st.markdown(f"- **{cluster_name}:** Predominan *{top_3_limpio[0]}*, *{top_3_limpio[1]}* y *{top_3_limpio[2]}*.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Solo quiero plotear los 10 delitos más importantes (si meto todos, el heatmap queda gigante)
    top_delitos = df_perfiles.sum(axis=1).sort_values(ascending=False).head(10).index
    df_perfiles_top = df_perfiles.loc[top_delitos]
    
    # text_auto='.0f' redondea a cero decimales. Más limpio.
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
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total de Municipios Evaluados", f"{resultado_pareto['total_municipios']:,}")
    kpi2.metric("Municipios que concentran el 80% del crimen", f"{resultado_pareto['cantidad_critica']:,}", delta_color="inverse")
    kpi3.metric("% del Territorio Nacional Afectado", f"{resultado_pareto['porcentaje_territorio']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.subheader("Curva de Concentración Acumulada")
        
        df_grafica = df_pareto.copy()
        # Sobrescribir el index con un rankingpara que Streamlit
        # no intente ordenar la gráfica alfabéticamente por nombre del municipio. Eso la destruía.
        df_grafica['Ranking de Gravedad'] = range(1, len(df_grafica) + 1)
        
        # px.line es mejor porque me deja meter la línea roja del threshold fácil.
        fig_pareto = px.line(
            df_grafica, x='Ranking de Gravedad', y='Porcentaje_Acumulado',
            labels={"Porcentaje_Acumulado": "% de Delitos Acumulados", "Ranking de Gravedad": "Municipios (De Peor a Mejor)"}
        )
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Umbral Crítico (80%)")
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    with col2:
        st.subheader("Top Focos Rojos Absolutos")
        columnas_ver = ['Municipio', 'Entidad', 'Tasa_Anual_100k']
        df_mostrar = df_pareto[columnas_ver].head(10).copy()
        df_mostrar['Tasa_Anual_100k'] = df_mostrar['Tasa_Anual_100k'].apply(lambda x: f"{x:,.2f}")
        st.dataframe(df_mostrar, hide_index=True)


def mostrar_tabla_datos(df: pd.DataFrame):
    st.title("Explorador de Inteligencia (Motor POO)")
    
    # Hardcode a 100k para la muestra.
    TAMANO_MUESTRA = 100000
    if len(df) > TAMANO_MUESTRA:
        df_muestra = df.sample(n=TAMANO_MUESTRA, random_state=42)
        st.caption(f"Aviso Metodológico: Operando sobre muestra representativa de {TAMANO_MUESTRA:,} registros.")
    else:
        df_muestra = df

    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        entidades = ["Todas"] + list(df_muestra["Entidad"].unique())
        estado_sel = st.selectbox("1. Entidad Federativa:", entidades)
        
    with col_f2:
        # Lógica anidada: Si no eligen estado, los municipios salen de todo el país.
        # Si sí eligen, filtro la lista de municipios para que solo salgan los de ese estado.
        if estado_sel == "Todas":
            municipios = ["Todos"]
        else:
            municipios = ["Todos"] + list(df_muestra[df_muestra["Entidad"] == estado_sel]["Municipio"].unique())
        mun_sel = st.selectbox("2. Municipio:", municipios)
        
    with col_f3:
        bienes = ["Todos"] + list(df_muestra["Bien_juridico_afectado"].unique())
        bien_sel = st.selectbox("3. Bien Jurídico:", bienes)

    # Aplico filtros en cascada al dataset
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
    
    # Aquí llamo a la arquitectura Orientada a Objetos pesada.
    lista_registros_poo = transformar_dataframe_a_objetos(df_filtrado)
    
    st.subheader("Resumen Ejecutivo Global")
    
    conteo_riesgos = {"Alto": 0, "Medio": 0, "Bajo": 0}
    lista_meses = []
    
    for reg in lista_registros_poo:
        riesgo = reg.categorizar_nivel_riesgo(umbral_alto=50.0, umbral_medio=20.0)
        conteo_riesgos[riesgo] += 1
        
        # Logica para tomar el mes más critico
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
    
    st.warning("Aviso Académico: La dependencia estadística encontrada no implica causalidad directa, sino una fuerte correlación asimétrica.")
    
    resultado = calcular_dependencia_demografica(df)
    
    st.subheader("1. Diagnóstico Ejecutivo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True) # Espacio para centrar verticalmente con el Gauge chart
        if resultado['diagnostico']['existe_relacion']:
            st.success("**Comprobado:** Sí existe una relación estadística. El tipo de entorno modifica el perfil criminal.")
        else:
            st.error("**Sin Relación:** El entorno no dicta el tipo de crimen.")
            
        st.markdown(f"**Interpretación:** {resultado['diagnostico']['fuerza_relacion']}")
        
    with col2:
        valor_cramer = resultado['diagnostico']['valor_cramer']
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = valor_cramer,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fuerza de Relación (V de Cramér)"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "rgba(255, 255, 255, 0)"}, # Barra base invisible, uso el threshold como aguja
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
    
    st.subheader("2. Perfil Criminal por Zona (Mapa de Calor %)")
    
    df_porcentajes = resultado['perfil_criminal_porcentajes']
    
    fig = px.imshow(
        df_porcentajes,
        text_auto='.1f', 
        aspect="auto",   
        color_continuous_scale='YlOrRd', 
        labels=dict(x="Tipo de Asentamiento", y="Bien Jurídico Afectado", color="% de Concentración")
    )
    
    fig.update_xaxes(side="top") # Pongo las etiquetas de X arriba para que se lea como tabla de Excel
    st.plotly_chart(fig, use_container_width=True)

# El menú lateral para movernos entre los módulos. 
st.sidebar.title("Menú")

opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ("Inicio", "Análisis Exploratorio (EDA)", "Clustering (K-Means)", "Consultar Datos", "Patrones y Pareto", "Análisis Demográfico")
)

# Aquí hago el ruteo. Dependiendo de qué pique el usuario, le inyecto el dataframe a la función de renderizado.
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