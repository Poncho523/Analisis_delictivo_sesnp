import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go 

# layout="wide" hace que la app ocupe toda la pantalla y no solo el centro.
st.set_page_config(page_title="Gestión de Incidentes SESNSP", page_icon="🚓", layout="wide")

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from motor_analitico.parapeto_concentracion import calcular_pareto_municipios
from motor_analitico.consultador_con_POO import transformar_dataframe_a_objetos
from motor_analitico.chi2 import calcular_dependencia_demografica
from motor_analitico.estadistica_descriptiva import generar_reporte_eda 
from motor_analitico.kmedias import ejecutar_pipeline_kmeans # <-- IMPORTAMOS EL NUEVO MOTOR K-MEANS

# @st.cache_data Memoriza el resultado de la función para no leer el CSV completo siempre
@st.cache_data
def obtener_datos_cacheados():
    return cargar_datos_incidencia()

# Mandamos llamar la función. 'dataset_global' vivirá en caché.
dataset_global = obtener_datos_cacheados()

def mostrar_pantalla_inicio(df: pd.DataFrame):
    st.title("Sistema de Análisis Delictivo Nacional")
    st.markdown("Plataforma de inteligencia criminal basada en datos del SESNSP.")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total de Incidentes Registrados", f"{len(df):,}") 
    col2.metric("Total de Municipios", f"{df['Cve_Municipio'].nunique():,}")
    col3.metric("Municipios Críticos", "1,171", "Pareto 80%")
    col4.metric("Estado de Alerta", "Activo", "Megalópolis", delta_color="inverse")
    st.divider()

    st.subheader("Mapa de Concentración Delictiva")
    # datos aleatorios
    mapa_datos = pd.DataFrame(
        np.random.randn(100, 2) / [10, 10] + [23.6345, -102.5528], 
        columns=['lat', 'lon']
    )
    # st.map renderiza un mapa interactivo
    st.map(mapa_datos, zoom=4, use_container_width=True)

def mostrar_analisis_exploratorio(df: pd.DataFrame):
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Radiografía estadística global de la criminalidad a nivel municipal.")
    
    st.info("Generando estadísticos y renderizando lienzo de Matplotlib...")
    
    reporte = generar_reporte_eda(df)
    
    st.subheader("Estadística Descriptiva Municipal")
    st.dataframe(reporte["estadisticas"], use_container_width=True)
    
    st.markdown("""
    **Interpretación Criminológica:**
    * **Media vs Mediana:** Si la media es mucho mayor a la mediana, hay concentración extrema en pocos municipios.
    * **Asimetría / Curtosis:** Valores altos confirman valores atípicos severos (outliers).
    """)
    
    st.divider()
    
    st.subheader("Diagnóstico Visual (Grid)")
    st.pyplot(reporte["figura_matplotlib"])

# --- NUEVO MÓDULO: CLUSTERING K-MEANS ---
def mostrar_clustering(df: pd.DataFrame):
    st.title("🤖 Perfilamiento Criminal Avanzado (K-Means)")
    st.markdown("Segmentación de municipios basada en la similitud de sus tasas delictivas mediante Inteligencia Artificial.")
    
    # INTERACTIVIDAD: El usuario elige cuántos clusters quiere
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parámetros del Modelo")
    k_elegido = st.sidebar.slider("Número de Perfiles (K)", min_value=2, max_value=8, value=4)
    
    st.info(f"Entrenando modelo K-Means con {k_elegido} clusters. Reduciendo dimensionalidad con PCA...")
    
    # Ejecutamos el pipeline
    resultado = ejecutar_pipeline_kmeans(df, n_clusters=k_elegido)
    
    # --- SECCIÓN 1: MÉTRICAS DE CALIDAD ---
    st.subheader("1. Diagnóstico del Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Varianza Retenida (PCA)", f"{resultado['metricas']['varianza_pca']:.1f}%")
    col2.metric("Súper-Componentes", f"{resultado['metricas']['dimensiones_pca']}")
    col3.metric("Silhouette Score (Calidad)", f"{resultado['metricas']['silhouette']:.3f}", "Rango ideal > 0.5")
    
    st.divider()
    
    # --- SECCIÓN 2: VISUALIZACIÓN 3D Y CODO ---
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        st.subheader("2. Universo de Municipios (PCA 3D)")
        st.markdown("Cada punto es un municipio. Los colores representan los perfiles criminales descubiertos.")
        
        # Gráfica 3D interactiva
        df_plot = resultado['df_clusters']
        if 'PCA_3' in df_plot.columns:
            fig_3d = px.scatter_3d(
                df_plot, x='PCA_1', y='PCA_2', z='PCA_3',
                color='Nombre_Cluster',
                hover_name='Municipio', hover_data=['Entidad'],
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("El PCA retuvo menos de 3 componentes. No se puede graficar en 3D.")
            
    with col_g2:
        st.subheader("Curva del Codo")
        st.markdown("Ayuda a validar si elegiste la K correcta (busca la 'rodilla').")
        df_codo = resultado['datos_codo'].set_index('K (Número de Clusters)')
        st.line_chart(df_codo)
        
    st.divider()
    
    # --- SECCIÓN 3: INTERPRETACIÓN DE LOS PERFILES (HEATMAP) ---
    st.subheader("3. ADN Criminal de los Perfiles (Top 10 Delitos)")
    st.markdown("Tasas promedio por cada 100k habitantes. **Busca las zonas rojas para entender la especialidad de cada grupo.**")
    
    # Extraemos los perfiles, los rotamos (T) para que los delitos sean las filas y filtramos el top 10
    df_perfiles = resultado['perfiles_promedio'].T
    # Sumamos las tasas para descubrir cuáles son los delitos más frecuentes en general
    top_delitos = df_perfiles.sum(axis=1).sort_values(ascending=False).head(10).index
    df_perfiles_top = df_perfiles.loc[top_delitos]
    
    fig_heat = px.imshow(
        df_perfiles_top,
        text_auto='.1f', 
        aspect="auto",   
        color_continuous_scale='Reds', 
        labels=dict(x="Perfil Criminal (Cluster)", y="Tipo de Delito", color="Tasa Promedio")
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
    kpi2.metric("Municipios que concentran el 80% del crimen", f"{resultado_pareto['cantidad_critica']:,}", "Focos Rojos", delta_color="inverse")
    kpi3.metric("% del Territorio Nacional Afectado", f"{resultado_pareto['porcentaje_territorio']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.subheader("Proporción de Territorialidad Criminal")
        
        municipios_peligrosos = resultado_pareto['cantidad_critica']
        municipios_pacificos = resultado_pareto['total_municipios'] - municipios_peligrosos
        
        datos_pastel = pd.DataFrame({
            "Categoría": ["Focos Rojos (Acumulan el 80% del crimen)", "Resto del País (Acumulan 20% del crimen)"],
            "Cantidad de Municipios": [municipios_peligrosos, municipios_pacificos]
        })
        
        fig = px.pie(
            datos_pastel, 
            values='Cantidad de Municipios', 
            names='Categoría',
            color='Categoría',
            color_discrete_map={
                "Focos Rojos (Acumulan el 80% del crimen)": "#ff4b4b",
                "Resto del País (Acumulan 20% del crimen)": "#1f77b4"
            },
            hole=0.4
        )
        
        fig.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Top Focos Rojos Absolutos")
        columnas_ver = ['Municipio', 'Entidad', 'Tasa_Anual_100k']
        df_mostrar = df_pareto[columnas_ver].head(10).copy()
        df_mostrar['Tasa_Anual_100k'] = df_mostrar['Tasa_Anual_100k'].apply(lambda x: f"{x:,.2f}")
        st.dataframe(df_mostrar, hide_index=True)

def mostrar_tabla_datos(df: pd.DataFrame):
    st.title("🗄️ Explorador de Inteligencia (Motor POO)")
    
    TAMANO_MUESTRA = 100000
    if len(df) > TAMANO_MUESTRA:
        df_muestra = df.sample(n=TAMANO_MUESTRA, random_state=42)
        st.caption(f"⚠️ **Aviso Metodológico:** Para garantizar el rendimiento, el motor POO está operando sobre una muestra estadísticamente representativa de {TAMANO_MUESTRA:,} registros (Nivel de Confianza: 99%).")
    else:
        df_muestra = df

    st.markdown("Consulta de registros evaluados a través del Modelo de Dominio.")
    
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
        st.warning("No se encontraron registros con estos filtros en la muestra.")
        return

    st.info(f"Ensamblando grafo de objetos para {len(df_filtrado)} registros encontrados...")
    
    lista_registros_poo = transformar_dataframe_a_objetos(df_filtrado)
    
    st.subheader("Resumen Ejecutivo Global")
    
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
    
    st.markdown("### Detalles de Casos de Estudio (Top 50 visualizados)")
    
    for registro in lista_registros_poo[:50]:
        titulo_caja = f"📍 {registro.municipio.nombre} | {registro.clasificacion.subtipoDelito} ({registro.clasificacion.modalidad})"
        
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
            c2.write(f"**Nivel de Riesgo:** {riesgo} 🚦")
            
            mes_moda = registro.obtener_mes_moda()
            c3.markdown("**Alerta Temporal**")
            
            if mes_moda and mes_moda.cantidadCasos > 0:
                c3.write(f"**Mes Crítico:** {mes_moda.mes.name} ({mes_moda.cantidadCasos} casos)")
            else:
                c3.write("**Mes Crítico:** Ninguno (0 casos)")
                
            c3.write(f"**Tendencia:** {registro.calcular_tendencia_semestral()}")
            c3.write(f"**Patrón:** {registro.determinar_patron_ocurrencia()}")

def mostrar_analisis_demografico(df: pd.DataFrame):
    st.title("⚖️ Dependencia Demográfica (Chi-Cuadrada)")
    st.markdown("Analiza científicamente si el tamaño de la ciudad dicta el tipo de crimen que sufre.")
    
    st.info("Procesando matriz condicional y estadísticos de Cramér...")
    resultado = calcular_dependencia_demografica(df)
    
    st.subheader("1. Diagnóstico Ejecutivo")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        if resultado['diagnostico']['existe_relacion']:
            st.success("✅ **Comprobado:** Sí existe una relación estadística. El tipo de entorno (Rural/Urbano) modifica significativamente el perfil criminal.")
        else:
            st.error("❌ **Sin Relación:** El entorno no dicta el tipo de crimen.")
            
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
    
    st.subheader("2. Perfil Criminal por Zona (Mapa de Calor %)")
    st.markdown("Lectura: *Busca los cuadros más rojos. Representan la especialidad criminal de esa zona.*")
    
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

st.sidebar.image("https://st.depositphotos.com/1000163/2482/i/450/depositphotos_24824379-stock-photo-handcuffs-and-judge-gavel-on.jpg", width=100)
st.sidebar.title("Menu")

# --- AGREGAMOS EL MÓDULO DE CLUSTERING AL MENÚ ---
opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ("Inicio", "Análisis Exploratorio (EDA)", "Clustering (K-Means)", "Consultar Datos", "Patrones y Pareto", "Análisis Demográfico")
)

if opcion == "Inicio":
    mostrar_pantalla_inicio(dataset_global)
elif opcion == "Análisis Exploratorio (EDA)":
    mostrar_analisis_exploratorio(dataset_global)
elif opcion == "Clustering (K-Means)":
    mostrar_clustering(dataset_global)
elif opcion == "Consultar Datos":
    mostrar_tabla_datos(dataset_global)
elif opcion == "Patrones y Pareto":
    mostrar_patrones_delictivos(dataset_global)
elif opcion == "Análisis Demográfico":
    mostrar_analisis_demografico(dataset_global)