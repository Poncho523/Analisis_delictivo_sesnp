import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# layout="wide" hace que la app ocupe toda la pantalla y no solo el centro.
st.set_page_config(page_title="Gestión de Incidentes SESNSP", page_icon="🚓", layout="wide")

# Puente Arquitectónico
ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))

from ETL.carga_datos import cargar_datos_incidencia
from motor_analitico.parapeto_concentracion import calcular_pareto_municipios
from motor_analitico.consultador_con_POO import transformar_dataframe_a_objetos

# @st.cache_data Memoriza el resultado de la función para no leer el CSV completo siempreAAA
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
    # el dataframe que le pases a st.map() DEBE tener columnas llamadas exactamente 'lat' y 'lon'.
    mapa_datos = pd.DataFrame(
        np.random.randn(100, 2) / [10, 10] + [23.6345, -102.5528], 
        columns=['lat', 'lon']
    )
    # st.map renderiza un mapa interactivo
    st.map(mapa_datos, zoom=4, use_container_width=True)


def mostrar_patrones_delictivos(df: pd.DataFrame):
    st.title("Análisis de Patrones (Ley de Pareto)")
    
    # st.info pone un cuadro azul de aviso. st.warning es amarillo, st.error es rojo.
    st.info("Ejecutando motor analítico en tiempo real...")
    
    # Le pasamos el dataset_global a tu función y guardamos el resultado.
    df_pareto = calcular_pareto_municipios(df)
    
    col1, col2 = st.columns([2, 1]) # La columna 1 será el doble de ancha que la 2
    
    with col1:
        st.subheader("Curva de Concentración Acumulada")
        # Solo le pasamos la columna que queremos graficar (Porcentaje_Acumulado).
        st.line_chart(df_pareto['Porcentaje_Acumulado'].head(100)) # Graficamos solo el top 100 para que se vea claro
        
    with col2:
        st.subheader("Top 5 Focos Rojos")
        # st.dataframe muestra una tabla de Pandas de forma interactiva (permite ordenar y hacer scroll).
        # Seleccionamos solo columnas relevantes para no saturar la vista.
        columnas_ver = ['Municipio', 'Entidad', 'Tasa_Anual_100k']
        st.dataframe(df_pareto[columnas_ver].head(5), hide_index=True)

def mostrar_tabla_datos(df: pd.DataFrame):
    st.title("🗄️ Explorador de Inteligencia (Motor POO)")
    st.markdown("Consulta de registros evaluados a través del Modelo de Dominio.")
    
    # 1. Filtros UI
    bienes_juridicos = ["Todos"] + list(df["Bien_juridico_afectado"].unique())
    tipo_filtro = st.selectbox("Filtrar por Bien Jurídico Afectado:", bienes_juridicos)
    
    df_filtrado = df
    if tipo_filtro != "Todos":
        df_filtrado = df[df["Bien_juridico_afectado"] == tipo_filtro]
    
    # 2. EL PUENTE: De Pandas a Objetos
    # Tomamos solo 50 registros para no saturar la pantalla
    st.info("Ensamblando grafo de objetos POO...")
    lista_registros_poo = transformar_dataframe_a_objetos(df_filtrado.head(50))
    
    # 3. RENDERIZADO DE TARJETAS (Explotando la POO)
    for registro in lista_registros_poo:
        
        # st.expander crea una "caja" desplegable para cada registro
        titulo_caja = f"📍 {registro.municipio.nombre} | {registro.clasificacion.tipoDelito}"
        with st.expander(titulo_caja):
            
            # Dividimos el interior de la caja en 3 columnas
            c1, c2, c3 = st.columns(3)
            
            # Columna 1: Usamos los métodos de la clase Delito
            c1.markdown("**Taxonomía Penal**")
            c1.write(f"**Subtipo:** {registro.clasificacion.subtipoDelito}")
            c1.write(f"**Prioridad de Atención:** Nivel {registro.clasificacion.calcular_peso_estadistico()}")
            
            # Columna 2: Usamos los métodos de la clase Municipio
            c2.markdown("**Contexto Demográfico**")
            c2.write(f"**Asentamiento:** {registro.municipio.clasificar_asentamiento(df_filtrado.iloc[0]['POB_TOTAL'])}")
            c2.write(f"**Tasa 100k:** {registro.tasaAnual100k:.2f}")
            
            # Columna 3: Usamos los métodos de Temporalidad
            mes_moda = registro.obtener_mes_moda()
            c3.markdown("**Alerta Temporal**")
            c3.write(f"**Mes Crítico:** {mes_moda.mes.name} ({mes_moda.cantidadCasos} casos)")
            c3.write(f"**Tendencia:** {registro.calcular_tendencia_semestral()}")



st.sidebar.image("https://st.depositphotos.com/1000163/2482/i/450/depositphotos_24824379-stock-photo-handcuffs-and-judge-gavel-on.jpg", width=100)
st.sidebar.title("Navegación")

# st.sidebar.radio crea opciones excluyentes en la barra lateral.
opcion = st.sidebar.radio(
    "Selecciona un módulo:",
    ("Inicio (Mapa)", "Consultador de Datos", "Patrones y Pareto", "Análisis Demográfico")
)
# A las funciones les pasamos el dataset_global cacheado para que trabajen.
if opcion == "Inicio (Mapa)":
    mostrar_pantalla_inicio(dataset_global)
elif opcion == "Consultador de Datos":
    mostrar_tabla_datos(dataset_global)
elif opcion == "Patrones y Pareto":
    mostrar_patrones_delictivos(dataset_global)
elif opcion == "Análisis Demográfico":
    st.title("Dependencia Demográfica (Chi-Cuadrada)")
    st.warning("Módulo en construcción: Aquí irá la matriz de probabilidad condicional y V de Cramér.")