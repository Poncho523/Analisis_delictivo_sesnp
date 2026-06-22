# **1\. Sistema de Análisis Delictivo Nacional**

**Análisis exploratorio y perfilamiento criminal utilizando datos del SESNSP.**

## **2\. Descripción General**

Aplicación desarrollada en Python para analizar patrones criminales en municipios mexicanos utilizando datos abiertos del Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP).

El sistema integra técnicas de análisis exploratorio de datos, estadística inferencial y aprendizaje no supervisado para identificar concentraciones delictivas, dependencias geográficas y perfiles criminales.

## **3\. Objetivo**

Analizar la distribución de la criminalidad en México e identificar patrones espaciales y estadísticos relevantes que puedan apoyar la toma de decisiones estratégicas relacionadas con la seguridad pública.

## **4\. Tecnologías Utilizadas**

* **Lenguaje:** Python 3.12  
* **Visualización:** Streamlit, Plotly, Matplotlib, Seaborn  
* **Manipulación de Datos:** Pandas, NumPy  
* **Machine Learning:** Scikit-Learn  
* **Estadística:** SciPy

## **5\. Arquitectura del Sistema**

El proyecto opera bajo dos flujos arquitectónicos diferenciados para garantizar la eficiencia (separación de la ingeniería de datos y el software en tiempo de ejecución):

**A. Pipeline de Datos (Fase de Ingeniería \- *Offline*) Este fue el que usamos al momento de preparar los datos que ya estan en el google drive**

Datos Crudos SESNSP ➔ Scripts ETL ➔ Data Mart Municipal ➔ Almacenamiento Cloud (Google Drive)

**B. Pipeline de Ejecución (Fase de Software \- *Runtime*)**

El sistema no procesa el ETL en tiempo real. Implementa un **Patrón de Caché Local**:

Consulta de Módulo ➔ Validación de Caché Local ➔ Descarga (gdown) si es necesario ➔ Carga en Memoria ➔ Motor Analítico ➔ Interfaz Streamlit

## **6\. Estructura del Proyecto**

ANALISIS\_DELICTIVO\_SESNP/  
├── dominio/  
│   ├── clasificacion\_delito.py  
│   ├── EntidadFederativa.py  
│   ├── municipio.py  
│   └── registro\_incidencia.py  
├── ETL/  
│   ├── carga\_data\_mart.py  
│   ├── carga\_datos.py  
│   ├── ETL(merge, agregando la población total y la tasa anual por cada 100k).py  
│   ├── ETL(eliminar acentos, eliminar año y agregar total anual).py  
│   ├── ETL(quitar los acentos del csv de municipios, eliminar año, clave_ent, sexo, ano y poblaciones).py  
│   └── ETL(eliminamos las columnas que no nos sirven antes de hacer el merge).py  
├── Interfaz\_visualizador/  
│   └── interfaz.py  
├── motor\_analitico/  
│   ├── chi2.py  
│   ├── consultador\_con\_POO.py  
│   ├── dbscan.py  
│   ├── estadistica\_descriptiva.py  
│   ├── kmedias.py  
│   └── parapeto\_concentracion.py  
├── .dockerignore  
├── .gitignore  
├── Dockerfile  
├── librerias.txt  
├── LICENSE  
└── README.md

## **7\. ETL y Construcción del Data Mart**

El trabajo de ingeniería de datos es el cimiento de los modelos de Machine Learning implementados en la nube:

* **Fuente de datos:** Registros de incidencia delictiva municipal del SESNSP.  
* **Limpieza:** Tratamiento de valores nulos y estandarización de catálogos.  
* **Transformación y Pivot:** Los datos originales (formato largo) fueron transformados y vectorizados.  
* **Data Mart:** Se construyó una matriz matemática donde:  
  * **Cada fila** representa un Municipio.  
  * **Cada columna** representa un subtipo de delito.  
  * *Resultado:* Estructura numérica pura que permite el cálculo de distancias euclidianas, servida directamente a la aplicación mediante caché.

## **8\. Descripción de Módulos**

* **EDA (Análisis Exploratorio de Datos)**  
  * *Qué hace:* Genera estadística descriptiva.  
  * *Gráficas:* Histogramas, diagramas de caja y dispersión.  
  * *Preguntas:* ¿Cuál es el volumen general y la varianza de la incidencia delictiva?  
* **Pareto**  
  * *Qué calcula:* Frecuencias acumuladas (Regla 80/20).  
  * *Para qué sirve:* Identifica rápidamente la minoría de municipios que concentran la mayoría de los delitos críticos.  
* **Chi-Cuadrada (![][image1])**  
  * *Hipótesis:* Evalúa la independencia entre variables categóricas.  
  * *Métricas:* Cálculo de P-valor para descartar el azar y V de Cramér para medir la fuerza de la relación.  
* **K-Means (Perfiles Globales)**  
  * *Pipeline técnico:* Tasas por cada 100k hab. ➔ StandardScaler ➔ PCA (Reducción de dimensionalidad) ➔ K-Means ➔ Evaluación con Silhouette Score ➔ Perfilamiento con Z-Scores.  
* **DBSCAN (Detector de Anomalías)**  
  * *Pipeline técnico:* Scala -> PCA ➔ DBSCAN (Agrupamiento por densidad) ➔ Aislamiento de Ruido (-1) ➔ Etiquetado de Anomalía Máxima (Z-Score Inverso).  
  * *Para qué sirve:* Encuentra los focos rojos y municipios hiperviolentos que rompen por completo la estadística nacional.  
* **Consultador POO**  
  * *Implementación:* Módulo de consultas diseñado bajo el paradigma de Orientación a Objetos, encapsulando la lógica de filtrado y extrayendo información de manera eficiente y estructurada.

## **9\. Instalación**

Ejecute los siguientes comandos en su terminal:

git clone \<https://github.com/Poncho523/Analisis_delictivo_sesnp.git>  
cd ANALISIS\_DELICTIVO\_SESNP  
pip install \-r librerias.txt

## **10\. Ejecución**

Para levantar la interfaz web interactiva:

streamlit run Interfaz\_visualizador/interfaz.py

## **11\. Dependencias**

La lista exacta de requerimientos se encuentra en librerias.txt y se instala mediante:

pip install \-r librerias.txt

## **12\. Guía de Uso**

1. Abrir la aplicación (automático en el navegador tras la ejecución).  
2. Seleccionar el módulo analítico deseado desde la barra de navegación lateral.  
3. Ingresar los parámetros solicitados (filtros de estado, número de clústeres, sensibilidad EPS, etc.).  
4. Explorar e interpretar los resultados visuales y tabulares generados.

## **13\. Resultados Principales**

* **Concentración:** La criminalidad presenta una fuerte concentración territorial (validación empírica de Pareto).  
* **Dependencia:** Existen diferencias significativas y correlación matemática entre los tipos de delitos y los contextos demográficos.  
* **Anomalías y Ruido Espacial (DBSCAN):** Se demostró que no existe una "normalidad" genérica en el país. El motor topológico catalogó al **43.2% de los municipios (911 en total) como ruido estadístico** (valores atípicos).  
* **Intervención Táctica:** Se aislaron con éxito focos rojos específicos que requieren políticas diferenciadas, detectando anomalías extremas como Iztapalapa (Aborto), Tijuana (Hostigamiento Sexual), Ecatepec (Lesiones) y León (Narcomenudeo).

## **14\. Limitaciones**

* El análisis utiliza datos estáticos de un único periodo específico.  
* No se implementaron modelos de series de tiempo para realizar predicciones futuras.  
* El modelo base K-Means demostró ser altamente sensible a los valores atípicos (limitación que fue resuelta implementando DBSCAN en paralelo).

## **15\. Trabajo Futuro**

* Integración de análisis de series de tiempo geolocalizadas.  
* Desarrollo de modelos predictivos (Forecasting) para anticipar brotes criminales estacionales.  
* Despliegue en la nube mediante contenedores Docker para acceso público.

## **16\. Integrantes del Proyecto**

* Velazquez Rojas Alfonso  
* Barajas Pablo Andrea Raquel

