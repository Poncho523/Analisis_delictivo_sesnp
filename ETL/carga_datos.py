import pandas as pd
from pathlib import Path

def cargar_datos_incidencia(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """Extrae el dataset de incidencia delictiva hacia la memoria (DataFrame).
    
    Esta función actúa como la Capa de Acceso a Datos. Centraliza la lectura 
    para que, al migrar a PostgreSQL, solo se modifique este archivo usando 
    pd.read_sql(), protegiendo así todos los scripts de análisis.

    Args:
        ruta_csv (str | Path | None): Ruta al archivo CSV. Si es None, 
            buscará automáticamente 'datos_nivelados_2015.csv' en la 
            raíz del proyecto.

    Returns:
        pd.DataFrame: El dataset cargado, listo para cálculos matemáticos.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta calculada.
    """
    # 1. Resolución dinámica de la ruta
    if ruta_csv is None:
        # __file__ es este script. .parent es la carpeta ETL, 
        # el segundo .parent es la raíz del proyecto.
        ruta_base = Path(__file__).parent.parent
        ruta_archivo = ruta_base / "datos_nivelados_2015.csv"
    else:
        ruta_archivo = Path(ruta_csv)

    # 2. Validación de seguridad antes de que Pandas colapse
    if not ruta_archivo.exists():
        raise FileNotFoundError(
            f"❌ No se encontró el dataset en: {ruta_archivo}\n"
            "Verifica que el archivo exista y no esté dentro del .gitignore temporalmente."
        )

    # 3. Extracción de datos
    # low_memory=False evita advertencias de tipos de datos en columnas mixtas al leer CSVs grandes
    print(f"Leyendo datos desde: {ruta_archivo.name} ...")
    df = pd.read_csv(ruta_archivo, encoding="utf-8", low_memory=False)
    
    return df

# Bloque de prueba rápida (Solo se ejecuta si corres este script directamente)
if __name__ == "__main__":
    try:
        dataset = cargar_datos_incidencia()
        print(f"✅ ¡Éxito! Dataset cargado con {len(dataset)} registros y {len(dataset.columns)} columnas.")
    except Exception as e:
        print(e)