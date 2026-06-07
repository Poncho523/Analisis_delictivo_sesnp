import pandas as pd
import gdown
import os
from pathlib import Path

def cargar_datos_incidencia(file_id: str = '1vsmdJFPhp_MlypHpiMGS8NpeYQBbBUmk') -> pd.DataFrame:
    """Extrae el dataset de incidencia delictiva desde Google Drive hacia la memoria.
    
    Implementa un patrón de caché local: descarga el archivo desde la nube solo 
    si no existe localmente. Utiliza 'gdown' para sortear la advertencia de 
    análisis de virus de Google Drive en archivos pesados (>100MB).

    Args:
        file_id (str): El identificador único del archivo en Google Drive.

    Returns:
        pd.DataFrame: El dataset cargado, listo para cálculos matemáticos.
    """
    # 1. Definir dónde se guardará la copia caché localmente
    ruta_base = Path(__file__).parent.parent
    ruta_cache = ruta_base / "datos_nivelados_2015_cache.csv"

    # 2. Verificar si ya tenemos el archivo descargado (Patrón Caché)
    if not ruta_cache.exists():
        print(f"☁️ Archivo local no encontrado. Descargando desde Google Drive (ID: {file_id}) ...")
        print("⏳ Esto puede tardar un par de minutos (Aprox. 300MB)...")
        
        # Construir URL directa
        url_drive = f'https://drive.google.com/uc?id={file_id}'
        
        # gdown maneja automáticamente el token de confirmación de archivos grandes
        gdown.download(url_drive, str(ruta_cache), quiet=False)
        
        if not ruta_cache.exists():
            raise Exception("❌ Error crítico: Falló la descarga desde Google Drive.")
        print("✅ Descarga completada y guardada en caché.")
    else:
        print(f"📁 Archivo encontrado en caché local: {ruta_cache.name}. Omitiendo descarga.")

    # 3. Extracción a Pandas
    print("🧠 Cargando dataset en memoria (Pandas)...")
    df = pd.read_csv(ruta_cache, encoding="utf-8", low_memory=False)
    
    return df

# Bloque de prueba rápida
if __name__ == "__main__":
    try:
        dataset = cargar_datos_incidencia()
        print("\n📊 Resumen de los datos cargados:")
        print(f"Registros: {len(dataset)}")
        print(f"Columnas: {len(dataset.columns)}")
        print(dataset.head(3))
    except Exception as e:
        print(f"\n❌ Se produjo un error: {e}")