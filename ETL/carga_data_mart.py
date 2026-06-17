import pandas as pd
import gdown
import os
from pathlib import Path

def cargar_data_mart(file_id: str = '13uJ-sDoZ7NV4TDGfY7JRRxWg3OfGjxfs') -> pd.DataFrame:
    """Extrae el Data Mart pivoteado desde Google Drive hacia la memoria.
    
    Implementa un patrón de caché local para evitar descargas redundantes.
    
    Args:
        file_id (str): El identificador único del archivo en Google Drive.

    Returns:
        pd.DataFrame: El Data Mart listo para K-Means.
    """
    ruta_base = Path(__file__).parent.parent
    ruta_cache = ruta_base / "datamart_municipios_cache.csv"

    if not ruta_cache.exists():
        print(f"Data Mart local no encontrado. Descargando desde Google Drive (ID: {file_id}) ...")
        
        url_drive = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url_drive, str(ruta_cache), quiet=False)
        
        if not ruta_cache.exists():
            raise Exception("Falló la descarga del Data Mart desde Google Drive.")
        print("Descarga completada y guardada en caché.")
    else:
        print(f"Data Mart encontrado en caché local: {ruta_cache.name}.")

    print("Cargando Data Mart en memoria...")
    df = pd.read_csv(ruta_cache, encoding="utf-8", low_memory=False)
    
    return df

if __name__ == "__main__":
    try:
        datamart = cargar_data_mart()
        print("\nResumen del Data Mart:")
        print(f"Municipios: {len(datamart)}")
        print(f"Columnas (Delitos): {len(datamart.columns)}")
        print(datamart.head(3))
    except Exception as e:
        print(f"\n Se produjo un error: {e}")