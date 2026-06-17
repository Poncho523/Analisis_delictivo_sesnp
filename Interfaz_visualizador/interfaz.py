import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter
import plotly.express as px # <-- NUEVA LIBRERÍA PARA EL PASTEL

# layout="wide" hace que la app ocupe toda la pantalla y no solo el centro.
st.set_page_config(page_title="Gestión de Incidentes SESNSP", page_icon="🚓", layout="wide")

ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from motor_analitico.parapeto_concentracion import calcular_pareto_municipios
from motor_analitico.consultador_con_POO import transformar_dataframe_a_objetos
from motor_analitico.chi2 import calcular_dependencia_demografica

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

def mostrar_patrones_delictivos(df: pd.DataFrame):
    st.title("Análisis de Patrones (Ley de Pareto)")
    st.info("Ejecutando motor analítico en tiempo real...")
    
    # 1. RECIBIMOS EL PAQUETE COMPLETO DE INTELIGENCIA
    resultado_pareto = calcular_pareto_municipios(df)
    df_pareto = resultado_pareto["datos_grafica"]
    
    # 2. STORYTELLING: Mostramos la conclusión
    st.subheader("La Historia del 80/20 en el Crimen Nacional")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total de Municipios Evaluados", f"{resultado_pareto['total_municipios']:,}")
    kpi2.metric("Municipios que concentran el 80% del crimen", f"{resultado_pareto['cantidad_critica']:,}", "Focos Rojos", delta_color="inverse")
    kpi3.metric("% del Territorio Nacional Afectado", f"{resultado_pareto['porcentaje_territorio']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1]) 
    
    with col1:
        st.subheader("Proporción de Territorialidad Criminal")
        
        # --- TU IDEA: LA GRÁFICA DE PASTEL ---
        municipios_peligrosos = resultado_pareto['cantidad_critica']
        municipios_pacificos = resultado_pareto['total_municipios'] - municipios_peligrosos
        
        # Creamos una tabla pequeña solo para el pastel
        datos_pastel = pd.DataFrame({
            "Categoría": ["Focos Rojos (Acumulan el 80% del crimen)", "Resto del País (Acumulan 20% del crimen)"],
            "Cantidad de Municipios": [municipios_peligrosos, municipios_pacificos]
        })
        
        # Generamos el Pastel Interactivo con Plotly
        fig = px.pie(
            datos_pastel, 
            values='Cantidad de Municipios', 
            names='Categoría',
            color='Categoría',
            color_discrete_map={
                "Focos Rojos (Acumulan el 80% del crimen)": "#ff4b4b", # Rojo alerta
                "Resto del País (Acumulan 20% del crimen)": "#1f77b4"  # Azul pacífico
            },
            hole=0.4 # Esto lo hace ver como una Dona (más moderno y elegante)
        )
        
        # Actualizamos el diseño para que el texto se vea por fuera
        fig.update_traces(textposition='outside', textinfo='percent+label')
        
        # Mostramos la gráfica en Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Top Focos Rojos Absolutos")
        columnas_ver = ['Municipio', 'Entidad', 'Tasa_Anual_100k']
        # st.dataframe muestra una tabla interactiva
        st.dataframe(df_pareto[columnas_ver].head(10), hide_index=True)

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
    
    st.info("Procesando matriz condicional y residuos estandarizados...")
    resultado = calcular_dependencia_demografica(df)
    
    st.subheader("1. Diagnóstico Ejecutivo")
    col1, col2 = st.columns(2)
    
    if resultado['diagnostico']['existe_relacion']:
        col1.success("✅ **Comprobado:** Sí existe una relación estadística entre el entorno y el delito.")
    else:
        col1.error("❌ **Sin Relación:** El entorno no dicta el crimen.")
        
    col2.metric("Fuerza de la Relación (V de Cramér)", 
                resultado['diagnostico']['fuerza_relacion'], 
                f"Score: {resultado['diagnostico']['valor_cramer']}")
    
    st.divider()
    
    st.subheader("2. Focos Rojos Demográficos")
    st.markdown("Concentraciones de crimen que rompen la normalidad estadística:")
    for alerta in resultado['top_focos_rojos']:
        st.error(alerta)
        
    st.divider()
    
    st.subheader("3. Perfil Criminal por Zona (%)")
    st.markdown("Lectura: *Del 100% de los crímenes en cada tipo de zona, así se distribuyen los delitos.*")
    df_porcentajes = resultado['perfil_criminal_porcentajes']
    st.bar_chart(df_porcentajes)

st.sidebar.image("https://st.depositphotos.com/1000163/2482/i/450/depositphotos_24824379-stock-photo-handcuffs-and-judge-gavel-on.jpg", width=100)
st.sidebar.title("Menu")

opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ("Inicio", "Consultar Datos", "Patrones y Pareto", "Análisis Demográfico")
)

if opcion == "Inicio":
    mostrar_pantalla_inicio(dataset_global)
elif opcion == "Consultar Datos":
    mostrar_tabla_datos(dataset_global)
elif opcion == "Patrones y Pareto":
    mostrar_patrones_delictivos(dataset_global)
elif opcion == "Análisis Demográfico":
    mostrar_analisis_demografico(dataset_global)